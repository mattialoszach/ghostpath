"""A bounded writer pipeline around the LeRobot 0.7 dataset API."""

import math
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore

import numpy as np
from PIL import Image

from .cameras import AVFOUNDATION_BUFFER, MAX_FRAME_AGE, Cameras, CameraSyncError, jpeg
from .models import Configuration
from .motion import JOINTS, Pose, validate_pose
from .poses import write_json

MIN_CAMERA_GAP_INTERVAL = 2.0
ARM_EXCURSION_DEG = 3.0


@dataclass(frozen=True)
class Record:
    timestamp: float
    state: Pose
    action: Pose
    manual: bool
    reference: Pose | None = None
    encoder_read: tuple[float, float] | None = None
    command_send: tuple[float, float] | None = None
    stage: str = ""
    path_cursor: float | None = None
    replay_rate: float | None = None

    @property
    def cutoff(self) -> float:
        return self.encoder_read[1] if self.encoder_read is not None else self.timestamp


def _feature(dtype: str, names: list[str]) -> dict:
    return {"dtype": dtype, "shape": (len(names),), "names": names}


@lru_cache(maxsize=1)
def load_dataset_backend():
    """Import the optional encoder before a take requests its first frame."""
    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("Install GhostPath with the [record] or [hardware] extra.") from exc
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Expected LeRobotDataset v3.0; found {CODEBASE_VERSION}.")
    return LeRobotDataset, RGBEncoderConfig


class Datasets:
    def __init__(self, root: Path, config: Configuration, cameras: Cameras):
        self.root, self.config, self.cameras = root, config, cameras
        self.repo_id = f"local/{root.name}"
        self.dataset = None
        self.created = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset")
        self.slots = BoundedSemaphore(18)
        self.error: str | None = None
        self.broken = False
        self.stopped_early = False
        self.frames = 0
        self.accepted = 0
        self.review_dir: TemporaryDirectory | None = None
        self.review_frames = 0
        self.samples = self.skipped_camera_samples = 0
        self.started = self.last_sample = self.first_saved = self.last_saved = 0.0
        self.last_camera_gap: float | None = None
        self.max_gap = self.max_skew = self.max_sensor_offset = 0.0
        self.max_arm_error = self.max_gripper_error = 0.0
        self.path_frames = 0
        self.max_arm_path_error = self.max_gripper_path_error = 0.0
        self.replay_timing: dict[str, float] | None = None
        self.indices: dict[str, int] = {}
        self.camera_modes: dict = {}
        self.camera_offsets: dict[str, list[float]] = {}
        self.discontinuities: list[int] = []
        self.last_saved_index = -1
        self.arm_path_peak: dict | None = None
        self.control_trace: list[Record] = []
        self.control_fps: int | None = None
        self.saved: list[dict] = []

    def prepare(self) -> Future:
        return self.executor.submit(self._prepare)

    def _prepare(self):
        if self.broken:
            raise RuntimeError("The dataset writer failed. Disconnect and choose a new dataset.")
        modes = self.cameras.check_recording(self.config.fps)
        if self.dataset is None:
            LeRobotDataset, RGBEncoderConfig = load_dataset_backend()
            encoder = RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=19, g=30)
            if self.created:
                self.dataset = LeRobotDataset.resume(
                    repo_id=self.repo_id,
                    root=self.root,
                    rgb_encoder=encoder,
                    video_backend="pyav",
                    encoder_threads=2,
                    streaming_encoding=True,
                    encoder_queue_maxsize=8,
                )
            else:
                self._create(LeRobotDataset, encoder)
        self._clear()
        self.camera_modes = modes
        self.camera_offsets = {spec.name: [] for spec in self.cameras.recorded}
        self.review_dir = TemporaryDirectory(prefix="ghostpath-review-")

    def _create(self, dataset_class, encoder):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.root.exists():
            raise FileExistsError(f"Dataset {self.root.name} already exists. Use a new name.")
        motor_features = [f"{joint}.pos" for joint in JOINTS]
        features = {
            "observation.state": _feature("float32", motor_features),
            "action": _feature("float32", motor_features),
            "ghostpath.actual_timestamp": _feature("float64", ["seconds"]),
            "ghostpath.encoder_read": _feature("float64", ["start_seconds", "end_seconds"]),
            "ghostpath.sample_index": _feature("int64", ["index"]),
            "ghostpath.segment_index": _feature("int64", ["index"]),
            "ghostpath.manual": _feature("bool", ["manual"]),
        }
        if self.cameras.recorded:
            for key in (
                "camera_timestamps",
                "camera_arrival_timestamps",
                "camera_available_timestamps",
            ):
                features[f"ghostpath.{key}"] = _feature(
                    "float64", [spec.name for spec in self.cameras.recorded]
                )
        for spec in self.cameras.recorded:
            features[f"observation.images.{spec.name}"] = {
                "dtype": "video",
                "shape": (spec.height, spec.width, 3),
                "names": ["height", "width", "channels"],
            }
        self.dataset = dataset_class.create(
            repo_id=self.repo_id,
            fps=self.config.fps,
            features=features,
            root=self.root,
            robot_type="so_follower" if self.config.device == "so101" else "so101_simulated",
            use_videos=bool(self.cameras.recorded),
            rgb_encoder=encoder,
            video_backend="pyav",
            encoder_threads=2,
            streaming_encoding=True,
            encoder_queue_maxsize=8,
        )
        self.created = True

    def offer(self, record: Record):
        if self.error:
            raise RuntimeError(self.error)
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Dataset writer fell behind. Lower FPS or camera resolution.")
        self.executor.submit(self._consume, record)

    def _consume(self, record: Record):
        try:
            if not self.error:
                self._add(record)
        except Exception as exc:
            self.error = f"Capture validation failed: {exc}"
        finally:
            self.slots.release()

    def _add(self, record: Record):
        period = 1 / self.config.fps
        stamp = record.timestamp
        if not math.isfinite(stamp):
            raise ValueError("invalid telemetry timestamp")
        if not self.samples:
            self.started = stamp
        else:
            gap = stamp - self.last_sample
            scheduled = self.samples * period
            if not 0 < gap <= period * 1.8 or abs(stamp - self.started - scheduled) > period * 1.1:
                raise ValueError("telemetry missed a recording deadline")
        self.last_sample = stamp
        self.samples += 1
        state, action = validate_pose(record.state), validate_pose(record.action)
        try:
            captures = self.cameras.frames(
                record.cutoff,
                state,
                self.indices,
                scheduled=self.started + (self.samples - 1) * period,
            )
        except CameraSyncError as exc:
            if (
                self.last_camera_gap is not None
                and stamp - self.last_camera_gap < MIN_CAMERA_GAP_INTERVAL
            ):
                raise ValueError(f"Repeated camera gaps: {exc}") from exc
            self.last_camera_gap = stamp
            self.skipped_camera_samples += 1
            return

        sample_index = self.samples - 1
        if self.frames:
            self.max_gap = max(self.max_gap, stamp - self.last_saved)
            if sample_index != self.last_saved_index + 1:
                self.discontinuities.append(self.frames)
        else:
            self.first_saved = stamp
        self.last_saved_index = sample_index
        self.last_saved = stamp
        self.max_arm_error = max(
            self.max_arm_error, *(abs(action[j] - state[j]) for j in JOINTS[:-1])
        )
        self.max_gripper_error = max(
            self.max_gripper_error, abs(action["gripper"] - state["gripper"])
        )
        if record.reference is not None:
            reference = validate_pose(record.reference)
            if self._on_path(record):
                self.path_frames += 1
                joint = max(JOINTS[:-1], key=lambda joint: abs(reference[joint] - state[joint]))
                error = abs(reference[joint] - state[joint])
                if self.arm_path_peak is None or error > self.max_arm_path_error:
                    self.max_arm_path_error = error
                    self.arm_path_peak = {"joint": joint, "seconds": round(stamp - self.started, 3)}
                self.max_gripper_path_error = max(
                    self.max_gripper_path_error, abs(reference["gripper"] - state["gripper"])
                )
        frame = {
            "task": self.config.task.strip(),
            "observation.state": np.array(list(state.values()), np.float32),
            "action": np.array(list(action.values()), np.float32),
            "ghostpath.actual_timestamp": np.array([stamp - self.started], np.float64),
            "ghostpath.encoder_read": np.array(
                [time - self.started for time in (record.encoder_read or (stamp, stamp))],
                np.float64,
            ),
            "ghostpath.sample_index": np.array([sample_index], np.int64),
            "ghostpath.segment_index": np.array([len(self.discontinuities)], np.int64),
            "ghostpath.manual": np.array([record.manual], np.bool_),
        }
        times = []
        previews = []
        review_interval = max(1, round(self.config.fps / 5))
        for spec, capture in zip(self.cameras.recorded, captures, strict=True):
            times.append(capture.timestamp)
            self.camera_offsets[spec.name].append((capture.timestamp - stamp) * 1000)
            if capture.pixels.shape != (spec.height, spec.width, 3):
                raise ValueError(f"{spec.name}: unexpected frame dimensions")
            if capture.pixels.dtype != np.uint8:
                raise ValueError(f"{spec.name}: expected 8-bit RGB")
            frame[f"observation.images.{spec.name}"] = capture.pixels
            if self.frames % review_interval == 0:
                previews.append((spec.name, capture.pixels))
        if times:
            skew = max(times) - min(times)
            self.max_skew = max(self.max_skew, skew)
            self.max_sensor_offset = max(self.max_sensor_offset, *(abs(t - stamp) for t in times))
            frame["ghostpath.camera_timestamps"] = np.array(
                [t - self.started for t in times], np.float64
            )
            frame["ghostpath.camera_arrival_timestamps"] = np.array(
                [capture.arrived - self.started for capture in captures], np.float64
            )
            frame["ghostpath.camera_available_timestamps"] = np.array(
                [capture.available - self.started for capture in captures], np.float64
            )
        self.dataset.add_frame(frame)
        self.indices.update(
            (spec.name, capture.sequence)
            for spec, capture in zip(self.cameras.recorded, captures, strict=True)
        )
        for name, pixels in previews:
            path = Path(self.review_dir.name) / f"{name}-{self.review_frames}.jpg"
            path.write_bytes(jpeg(Image.fromarray(pixels), small=True))
        if previews:
            self.review_frames += 1
        self.frames += 1

    def seal(
        self,
        *,
        stopped_early: bool = False,
        replay_timing: dict[str, float] | None = None,
        control_trace: list[Record] | None = None,
        control_fps: int | None = None,
    ) -> Future:
        return self.executor.submit(
            self._seal, stopped_early, replay_timing, control_trace, control_fps
        )

    def _seal(self, stopped_early, replay_timing, control_trace, control_fps) -> dict:
        self.stopped_early = stopped_early
        self.replay_timing = replay_timing
        self.control_trace = control_trace or []
        self.control_fps = control_fps
        if self.control_trace:
            self._summarize_tracking()
        return self._validate()

    @staticmethod
    def _on_path(tick: Record) -> bool:
        return tick.reference is not None and not tick.manual and tick.stage != "aligning"

    def _summarize_tracking(self):
        ticks = [tick for tick in self.control_trace if self._on_path(tick)]
        if not ticks:
            return
        error = np.abs(
            np.array(
                [[tick.reference[joint] - tick.state[joint] for joint in JOINTS] for tick in ticks]
            )
        )
        arm = error[:, :-1]
        row, column = np.unravel_index(arm.argmax(), arm.shape)
        self.max_arm_path_error = peak = float(arm[row, column])
        self.max_gripper_path_error = float(error[:, -1].max())
        self.path_frames = len(ticks)
        self.arm_path_peak = {
            "joint": JOINTS[column],
            "seconds": round(ticks[row].timestamp - self.started, 3),
        }
        if peak >= ARM_EXCURSION_DEG:
            first = last = row
            while first > 0 and arm[first - 1, column] >= ARM_EXCURSION_DEG:
                first -= 1
            while last + 1 < len(ticks) and arm[last + 1, column] >= ARM_EXCURSION_DEG:
                last += 1
            self.arm_path_peak.update(
                excursion_deg=ARM_EXCURSION_DEG,
                excursion_ms=round((ticks[last].timestamp - ticks[first].timestamp) * 1000, 1),
            )

    def _validate(self) -> dict:
        if self.error:
            raise ValueError(self.error)
        if self.frames < 3:
            raise ValueError("The episode has too few frames.")
        return {
            "stopped_early": self.stopped_early,
            "frames": self.frames,
            "seconds": round(self.frames / self.config.fps, 2),
            "capture_span_seconds": round(self.last_saved - self.first_saved, 3),
            "max_gap_ms": round(self.max_gap * 1000, 1),
            "camera_skew_ms": round(self.max_skew * 1000, 1),
            "skipped_camera_samples": self.skipped_camera_samples,
            "discontinuities": self.discontinuities,
            "camera_offset_ms": {
                name: {
                    f"p{p}": round(float(value), 1)
                    for p, value in zip(
                        (1, 50, 95, 99), np.percentile(offsets, (1, 50, 95, 99)), strict=True
                    )
                }
                for name, offsets in self.camera_offsets.items()
                if offsets
            },
            "sensor_offset_ms": round(self.max_sensor_offset * 1000, 1),
            "max_arm_error_deg": round(self.max_arm_error, 2),
            "max_gripper_error_pct": round(self.max_gripper_error, 2),
            "max_arm_path_error_deg": (
                round(self.max_arm_path_error, 2) if self.path_frames else None
            ),
            "arm_path_peak": self.arm_path_peak,
            "max_gripper_path_error_pct": (
                round(self.max_gripper_path_error, 2) if self.path_frames else None
            ),
            "replay_timing": self.replay_timing,
            "review_frames": self.review_frames,
        }

    def save(
        self, path_id: str, mode: str, *, manual_finish: bool = False, firm_carry_grip: bool = False
    ) -> Future:
        return self.executor.submit(self._save, path_id, mode, manual_finish, firm_carry_grip)

    def _save(self, path_id: str, mode: str, manual_finish: bool, firm_carry_grip: bool):
        quality = self._validate()
        record = {
            "episode": self.accepted,
            "path_id": path_id,
            "mode": mode,
            "manual_finish": manual_finish,
            "firm_carry_grip": firm_carry_grip,
            "cameras": self.camera_modes,
            **quality,
        }
        accepted_before = self._output_files()
        trace_path = self.root / "control" / f"episode-{self.accepted:06d}.json"
        if self.control_trace:
            record["control_trace"] = trace_path.relative_to(self.root).as_posix()
        dataset = self.dataset
        # A resumed v3 writer starts new Parquet and video files. Keep the existing
        # metadata so an encoder or disk error cannot append a partial episode.
        with TemporaryDirectory(prefix="ghostpath-checkpoint-", dir=self.root.parent) as backup:
            metadata = Path(backup) / "meta"
            shutil.copytree(self.root / "meta", metadata)
            try:
                dataset.save_episode(parallel_encoding=False)
                dataset.finalize()
                if dataset.meta.total_episodes != self.accepted + 1:
                    raise RuntimeError("LeRobot did not commit the expected episode index.")
                self._verify_videos(self._output_files() - accepted_before)
                if self.control_trace:
                    self._write_control(trace_path)
                write_json(
                    self.root / "ghostpath.json",
                    {
                        "version": 2,
                        "config": self.config.model_dump(mode="json"),
                        "observation_policy": {
                            "cutoff": "encoder_read_end",
                            "avfoundation_buffer_seconds": AVFOUNDATION_BUFFER,
                            "max_frame_age_seconds": MAX_FRAME_AGE,
                        },
                        "episodes": [*self.saved, record],
                    },
                )
            except Exception:
                self.dataset = None
                with suppress(Exception):
                    dataset.finalize()
                # Drop the old writer before restoring, so its destructor cannot
                # flush a partially written episode over the restored metadata.
                del dataset
                try:
                    trace_path.unlink(missing_ok=True)
                    for file in self._output_files() - accepted_before:
                        file.unlink()
                    shutil.rmtree(self.root / "meta")
                    metadata.replace(self.root / "meta")
                    if (self.root / "images").exists():
                        shutil.rmtree(self.root / "images")
                except OSError as exc:
                    self.broken = True
                    raise RuntimeError(f"Dataset rollback failed: {exc}") from exc
                raise
        self.dataset = None
        self.accepted += 1
        self.saved.append(record)
        self.control_trace = []
        self._clean_review()
        self.frames = 0

    def _verify_videos(self, new_files: set[Path]):
        if not self.cameras.recorded:
            return
        import av

        videos = [path for path in new_files if path.suffix == ".mp4"]
        if len(videos) != len(self.cameras.recorded):
            raise RuntimeError("The encoder did not save a video for every recorded camera.")
        for path in videos:
            with av.open(str(path)) as container:
                if container.streams.video[0].frames != self.frames:
                    raise RuntimeError(
                        "The encoder lost video frames. Lower camera resolution or dataset FPS."
                    )

    def _write_control(self, path: Path):
        def relative(window):
            return [stamp - self.started for stamp in window] if window is not None else None

        write_json(
            path,
            {
                "joint_order": JOINTS,
                "nominal_hz": self.control_fps,
                "ticks": [
                    {
                        "timestamp": tick.timestamp - self.started,
                        "encoder_read": relative(tick.encoder_read),
                        "command_send": relative(tick.command_send),
                        "stage": tick.stage,
                        "path_cursor": tick.path_cursor,
                        "replay_rate": tick.replay_rate,
                        "state": [tick.state[joint] for joint in JOINTS],
                        "action": [tick.action[joint] for joint in JOINTS],
                        "reference": (
                            [tick.reference[joint] for joint in JOINTS] if tick.reference else None
                        ),
                    }
                    for tick in self.control_trace
                ],
            },
        )

    def _output_files(self) -> set[Path]:
        return {
            path
            for name in ("data", "videos")
            for path in (self.root / name).rglob("*")
            if path.is_file()
        }

    def discard(self) -> Future:
        return self.executor.submit(self._clear)

    def _clean_review(self):
        if self.review_dir is not None:
            self.review_dir.cleanup()
            self.review_dir = None
        self.review_frames = 0

    def _clear(self):
        self.stopped_early = False
        self._clean_review()
        if self.dataset is not None and not self.broken:
            self.dataset.clear_episode_buffer()
        self.error = None
        self.frames = 0
        self.indices.clear()
        self.camera_modes = {}
        self.camera_offsets = {}
        self.discontinuities = []
        self.last_saved_index = -1
        self.arm_path_peak = None
        self.control_trace = []
        self.control_fps = None
        self.samples = self.skipped_camera_samples = 0
        self.started = self.last_sample = self.first_saved = self.last_saved = 0.0
        self.last_camera_gap = None
        self.max_gap = self.max_skew = self.max_sensor_offset = 0.0
        self.max_arm_error = self.max_gripper_error = 0.0
        self.path_frames = 0
        self.max_arm_path_error = self.max_gripper_path_error = 0.0
        self.replay_timing = None

    def finalize(self) -> Future:
        return self.executor.submit(self._finalize)

    def _finalize(self):
        self._clear()
        if self.dataset is not None:
            self.dataset.finalize()
            self.dataset = None
        if self.created and not self.accepted and not self.broken:
            shutil.rmtree(self.root)
            self.created = False
        if self.broken:
            raise RuntimeError("The dataset needs repair after a failed write.")

    def review(self, name: str, index: int) -> bytes | None:
        directory = self.review_dir
        if not directory or name not in self.indices or not 0 <= index < self.review_frames:
            return None
        try:
            return (Path(directory.name) / f"{name}-{index}.jpg").read_bytes()
        except FileNotFoundError:
            return None

    def close(self):
        try:
            self.finalize().result()
        finally:
            self.executor.shutdown(wait=True)
