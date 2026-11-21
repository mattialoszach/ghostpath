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

from .cameras import Cameras, jpeg
from .models import Configuration
from .motion import JOINTS, Pose, validate_pose
from .poses import write_json


@dataclass(frozen=True)
class Record:
    timestamp: float
    state: Pose
    action: Pose
    manual: bool


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
        self.started = self.last = self.max_gap = self.max_skew = self.max_sensor_offset = 0.0
        self.max_arm_error = self.max_gripper_error = 0.0
        self.indices: dict[str, int] = {}
        self.saved: list[dict] = []

    def prepare(self) -> Future:
        return self.executor.submit(self._prepare)

    def _prepare(self):
        if self.broken:
            raise RuntimeError("The dataset writer failed. Disconnect and choose a new dataset.")
        self.cameras.check_recording(self.config.fps)
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
                )
            else:
                self._create(LeRobotDataset, encoder)
        self._clear()
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
            "ghostpath.manual": _feature("bool", ["manual"]),
        }
        if self.cameras.recorded:
            features["ghostpath.camera_timestamps"] = _feature(
                "float64", [s.name for s in self.cameras.recorded]
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
        if not self.frames:
            self.started = stamp
        else:
            gap = stamp - self.last
            scheduled = self.frames * period
            if not 0 < gap <= period * 1.8 or abs(stamp - self.started - scheduled) > period * 1.1:
                raise ValueError("telemetry missed a recording deadline")
            self.max_gap = max(self.max_gap, gap)
        self.last = stamp
        state, action = validate_pose(record.state), validate_pose(record.action)
        self.max_arm_error = max(
            self.max_arm_error, *(abs(action[j] - state[j]) for j in JOINTS[:-1])
        )
        self.max_gripper_error = max(
            self.max_gripper_error, abs(action["gripper"] - state["gripper"])
        )
        frame = {
            "task": self.config.task.strip(),
            "observation.state": np.array(list(state.values()), np.float32),
            "action": np.array(list(action.values()), np.float32),
            "ghostpath.actual_timestamp": np.array([stamp - self.started], np.float64),
            "ghostpath.manual": np.array([record.manual], np.bool_),
        }
        times = []
        previews = []
        review_interval = max(1, round(self.config.fps / 5))
        captures = self.cameras.frames(stamp, state, period, self.indices)
        for spec, capture in zip(self.cameras.recorded, captures, strict=True):
            times.append(capture.timestamp)
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

    def seal(self, *, stopped_early: bool = False) -> Future:
        return self.executor.submit(self._seal, stopped_early)

    def _seal(self, stopped_early: bool) -> dict:
        self.stopped_early = stopped_early
        return self._validate()

    def _validate(self) -> dict:
        if self.error:
            raise ValueError(self.error)
        if self.frames < 3:
            raise ValueError("The episode has too few frames.")
        return {
            "stopped_early": self.stopped_early,
            "frames": self.frames,
            "seconds": round(self.frames / self.config.fps, 2),
            "max_gap_ms": round(self.max_gap * 1000, 1),
            "camera_skew_ms": round(self.max_skew * 1000, 1),
            "sensor_offset_ms": round(self.max_sensor_offset * 1000, 1),
            "max_arm_error_deg": round(self.max_arm_error, 2),
            "max_gripper_error_pct": round(self.max_gripper_error, 2),
            "review_frames": self.review_frames,
        }

    def save(self, path_id: str, mode: str, *, manual_finish: bool = False) -> Future:
        return self.executor.submit(self._save, path_id, mode, manual_finish)

    def _save(self, path_id: str, mode: str, manual_finish: bool):
        quality = self._validate()
        record = {
            "episode": self.accepted,
            "path_id": path_id,
            "mode": mode,
            "manual_finish": manual_finish,
            **quality,
        }
        accepted_before = self._output_files()
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
                write_json(
                    self.root / "ghostpath.json",
                    {
                        "version": 1,
                        "config": self.config.model_dump(mode="json"),
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
        self._clean_review()
        self.frames = 0

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
        self.started = self.last = self.max_gap = self.max_skew = self.max_sensor_offset = 0.0
        self.max_arm_error = self.max_gripper_error = 0.0

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
