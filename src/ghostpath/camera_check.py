"""Measure simultaneous camera delivery without connecting to a robot."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cameras import Cameras, source_index
from .dataset import Datasets, Record, load_dataset_backend
from .models import CameraSpec, Configuration
from .motion import JOINTS


def _check_export(writer: Datasets, expected: int) -> dict:
    quality = writer.seal().result()
    if quality["frames"] != expected or quality["skipped_camera_samples"]:
        raise RuntimeError(
            f"Saved {quality['frames']}/{expected} rows. "
            "Lower dataset FPS or check camera delivery."
        )
    writer.save("camera-check", "teach").result()
    dataset_class, _ = load_dataset_backend()
    dataset = dataset_class(
        writer.repo_id, root=writer.root, video_backend="pyav", return_uint8=True
    )
    if dataset.num_frames != expected:
        raise RuntimeError(f"LeRobot loaded {dataset.num_frames}/{expected} rows.")
    recorded = writer.cameras.recorded
    for index in sorted({0, expected // 2, expected - 1}):
        row = dataset[index]
        for spec in recorded:
            if tuple(row[f"observation.images.{spec.name}"].shape) != (3, spec.height, spec.width):
                raise RuntimeError(f"{spec.name}: decoded video has unexpected dimensions.")
    telemetry = dataset.hf_dataset.with_format(None).select_columns(
        [
            "ghostpath.camera_timestamps",
            "ghostpath.camera_available_timestamps",
            "ghostpath.encoder_read",
        ]
    )
    cutoffs = np.asarray(telemetry["ghostpath.encoder_read"])[:, 1]
    available = np.asarray(telemetry["ghostpath.camera_available_timestamps"]).reshape(
        expected, len(recorded)
    )
    stamps = np.asarray(telemetry["ghostpath.camera_timestamps"]).reshape(expected, len(recorded))
    if np.any(available > cutoffs[:, None]) or np.any(np.diff(stamps, axis=0) <= 0):
        raise RuntimeError("Export includes late or reused camera frames.")
    video_bytes = sum(file.stat().st_size for file in writer.root.rglob("*.mp4"))
    return {"dataset_fps": writer.config.fps, "video_bytes": video_bytes, **quality}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Camera indexes or persistent device IDs")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--dataset-fps", type=int, help="Also verify a temporary LeRobot export")
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("Measure for at least two seconds.")
    try:
        specs = [
            CameraSpec(
                name=f"camera_{index}",
                source=source_index(source),
                width=args.width,
                height=args.height,
                capture_fps=args.fps,
            )
            for index, source in enumerate(args.sources)
        ]
        config = (
            Configuration(cameras=specs, fps=args.dataset_fps, task="Temporary camera check")
            if args.dataset_fps is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    if len({spec.source for spec in specs}) != len(specs):
        parser.error("Choose distinct sources.")

    cameras = Cameras(specs, simulation=False)
    frames: dict[str, dict[int, tuple[float, float, float]]] = {spec.name: {} for spec in specs}
    with ExitStack() as cleanup:
        writer = None
        if config:
            load_dataset_backend()
            path = cleanup.enter_context(TemporaryDirectory(prefix="ghostpath-camera-check-"))
            writer = Datasets(Path(path) / "camera-check", config, cameras)
            cleanup.callback(writer.close)
        cameras.open()
        cleanup.callback(cameras.close)
        wait_until = time.monotonic() + 12
        while not all(status["fps"] for status in cameras.status().values()):
            if time.monotonic() >= wait_until:
                print(json.dumps(cameras.status(), indent=2))
                raise SystemExit("Not all cameras started delivering frames.")
            time.sleep(0.05)

        if writer:
            writer.prepare().result()
        state = dict.fromkeys(JOINTS, 0.0)
        rows = 0
        expected = args.seconds * config.fps if config else 0
        period = 1 / config.fps if config else 0.0
        start = time.monotonic()
        finish = start + args.seconds
        while True:
            now = time.monotonic()
            if writer and rows < expected and now >= start + rows * period:
                writer.offer(Record(now, state, state, False))
                rows += 1
            for name, camera in cameras.captures.items():
                with camera.condition:
                    frames[name].update(
                        (frame.sequence, (frame.timestamp, frame.arrived, frame.available))
                        for frame in camera.history
                        if start <= frame.available < finish
                    )
            remaining = finish - time.monotonic()
            if remaining <= 0:
                break
            next_row = start + rows * period - time.monotonic() if rows < expected else remaining
            time.sleep(max(0, min(0.05, remaining, next_row)))
        statuses = cameras.status()
        result = _check_export(writer, expected) if writer else None

    for name, samples in frames.items():
        status = statuses[name]
        report = {"camera": name, **status, "delivered_frames": len(samples)}
        if len(samples) >= 2:
            stamps, arrivals, available = np.array(list(samples.values())).T
            gaps = np.diff(stamps)
            report.update(
                fps=round(len(samples) / args.seconds, 2),
                gap_p95_ms=round(float(np.percentile(gaps, 95) * 1000), 1),
                max_gap_ms=round(float(gaps.max() * 1000), 1),
                gaps_over_1_5_periods=int(np.sum(gaps > 1.5 / args.fps)),
                max_availability_gap_ms=round(float(np.diff(available).max() * 1000), 1),
                timestamp_to_arrival_p95_ms=round(
                    float(np.percentile(arrivals - stamps, 95) * 1000), 1
                ),
                arrival_to_available_p95_ms=round(
                    float(np.percentile(available - arrivals, 95) * 1000), 1
                ),
            )
        print(json.dumps(report))
    if result:
        print(json.dumps({"temporary_export": result}))
    if any(not status["ready"] for status in statuses.values()) or any(
        len(samples) < args.seconds * args.fps * 0.95 for samples in frames.values()
    ):
        raise SystemExit("At least one camera fell below 95% of the requested FPS.")


if __name__ == "__main__":
    main()
