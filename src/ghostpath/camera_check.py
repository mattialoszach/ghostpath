"""Measure simultaneous camera delivery without connecting to a robot."""

import argparse
import json
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cameras import (
    AVFOUNDATION_BUFFER,
    FRAME_HISTORY_SIZE,
    Cameras,
    CameraTiming,
    camera_candidates,
    latest_match,
    pairing_tolerance,
    source_index,
    timing_stats,
)
from .dataset import Datasets, Record, load_dataset_backend
from .models import DEFAULT_MAC_CAMERA_BUFFER_MS, CameraSpec, Configuration
from .motion import JOINTS

BUFFER_OPTIONS_MS = (105, 90, 75, 68, 60, 0)


def compare_pairing(
    frames: dict[str, list[CameraTiming]], ticks: list[tuple[float, float]],
    buffered: set[str], tolerance: float,
) -> list[dict]:
    """Replay (scheduled tick, encoder cutoff) pairs on a common delivery trace."""
    pending = {name: deque(stream) for name, stream in frames.items()}
    histories = {name: deque(maxlen=FRAME_HISTORY_SIZE) for name in frames}
    used = {buffer: dict.fromkeys(frames, -1) for buffer in BUFFER_OPTIONS_MS}
    ages = {buffer: {name: [] for name in frames} for buffer in BUFFER_OPTIONS_MS}
    skews = {buffer: [] for buffer in BUFFER_OPTIONS_MS}
    for scheduled, cutoff in ticks:
        for name, stream in pending.items():
            while stream and stream[0].available <= cutoff:
                histories[name].append(stream.popleft())
        for buffer in BUFFER_OPTIONS_MS:
            groups = [
                camera_candidates(
                    history, cutoff, used[buffer][name], scheduled,
                    buffer / 1000 if name in buffered else 0,
                )
                for name, history in histories.items()
            ]
            if match := latest_match(groups, tolerance):
                for name, frame in zip(frames, match, strict=True):
                    used[buffer][name] = frame.sequence
                    ages[buffer][name].append(cutoff - frame.timestamp)
                stamps = [frame.timestamp for frame in match]
                skews[buffer].append(max(stamps) - min(stamps))
    return [
        {
            "buffer_ms": buffer,
            "matched_rows": len(skews[buffer]),
            "missed_rows": len(ticks) - len(skews[buffer]),
            "image_age_ms": {
                name: timing_stats(values) for name, values in ages[buffer].items() if values
            },
            "camera_skew_ms": timing_stats(skews[buffer]) if skews[buffer] else None,
        }
        for buffer in BUFFER_OPTIONS_MS
    ]


def _write_trace(path, start, finish, statuses, frames, ticks, tolerance):
    with path.open("x") as stream:
        json.dump(
            {
                "clock": "seconds_relative_to_measurement_start",
                "measurement_seconds": finish - start,
                "pairing_tolerance_seconds": tolerance,
                "cameras": statuses,
                "frames": {
                    name: [
                        {
                            "sequence": frame.sequence,
                            "timestamp": frame.timestamp - start,
                            "arrival": frame.arrived - start,
                            "available": frame.available - start,
                        }
                        for frame in samples
                    ]
                    for name, samples in frames.items()
                },
                "ticks": [
                    {"scheduled": scheduled - start, "cutoff": cutoff - start}
                    for scheduled, cutoff in ticks
                ],
            },
            stream,
        )
        stream.write("\n")


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
            "ghostpath.camera_arrival_timestamps",
            "ghostpath.camera_available_timestamps",
            "ghostpath.encoder_read",
        ]
    )
    cutoffs = np.asarray(telemetry["ghostpath.encoder_read"])[:, 1, None]
    shape = (expected, len(recorded))
    stamps = np.asarray(telemetry["ghostpath.camera_timestamps"]).reshape(shape)
    arrivals = np.asarray(telemetry["ghostpath.camera_arrival_timestamps"]).reshape(shape)
    available = np.asarray(telemetry["ghostpath.camera_available_timestamps"]).reshape(shape)
    if (
        not np.isfinite([stamps, arrivals, available]).all()
        or np.any(stamps > arrivals) or np.any(arrivals > available)
        or np.any(available > cutoffs) or np.any(np.diff(stamps, axis=0) <= 0)
    ):
        raise RuntimeError("Export includes invalid, late or reused camera timestamps.")
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
    parser.add_argument("--compare", action="store_true", help="Compare causal pairing policies")
    parser.add_argument("--timing-trace", type=Path, help="Save timing metadata without images")
    parser.add_argument(
        "--buffer-ms", type=int, choices=BUFFER_OPTIONS_MS,
        help=f"Override the macOS buffer (default: {DEFAULT_MAC_CAMERA_BUFFER_MS} ms)",
    )
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("Measure for at least two seconds.")
    if args.dataset_fps is None and (args.compare or args.buffer_ms is not None):
        parser.error("--compare and --buffer-ms require --dataset-fps.")
    if args.timing_trace and args.timing_trace.exists():
        parser.error("Choose a new file for --timing-trace.")
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

    frames: dict[str, list[CameraTiming]] = {spec.name: [] for spec in specs}
    buffer = AVFOUNDATION_BUFFER if args.buffer_ms is None else args.buffer_ms / 1000
    cameras = Cameras(
        specs, simulation=False, on_frame=lambda name, frame: frames[name].append(frame),
        avfoundation_buffer=buffer,
    )
    ticks: list[tuple[float, float]] = []
    export_error = None
    result = None
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
                ticks.append(((ticks[0][1] if ticks else now) + rows * period, now))
                if export_error is None:
                    try:
                        writer.offer(Record(now, state, state, False))
                    except (RuntimeError, ValueError) as exc:
                        export_error = str(exc)
                rows += 1
            remaining = finish - time.monotonic()
            if remaining <= 0:
                break
            next_row = start + rows * period - time.monotonic() if rows < expected else remaining
            time.sleep(max(0, min(0.05, remaining, next_row)))
        statuses = cameras.status()
        if writer and export_error is None:
            try:
                result = _check_export(writer, expected)
            except (RuntimeError, ValueError) as exc:
                export_error = str(exc)

    tolerance = pairing_tolerance(specs)
    frames = {
        name: [frame for frame in stream if frame.available < finish]
        for name, stream in frames.items()
    }
    if args.timing_trace:
        _write_trace(args.timing_trace, start, finish, statuses, frames, ticks, tolerance)
    if args.compare:
        buffered = {
            name for name, status in statuses.items()
            if (status["mode"] or {}).get("timestamp_source") == "avfoundation_pts"
        }
        comparison = compare_pairing(frames, ticks, buffered, tolerance)
        print(json.dumps({"pairing_comparison": comparison}))
    delivered = {
        name: [frame for frame in stream if frame.available >= start]
        for name, stream in frames.items()
    }
    for name, samples in delivered.items():
        status = statuses[name]
        report = {"camera": name, **status, "delivered_frames": len(samples)}
        if len(samples) >= 2:
            stamps, arrivals, available = np.array([
                (frame.timestamp, frame.arrived, frame.available) for frame in samples
            ]).T
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
    if export_error is not None:
        raise SystemExit(export_error)
    if any(not status["ready"] for status in statuses.values()) or any(
        len(samples) < args.seconds * args.fps * 0.95 for samples in delivered.values()
    ):
        raise SystemExit("At least one camera fell below 95% of the requested FPS.")


if __name__ == "__main__":
    main()
