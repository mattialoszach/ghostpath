"""Measure simultaneous camera delivery without connecting to a robot."""

import argparse
import json
import time

import numpy as np

from .cameras import Cameras, source_index
from .models import CameraSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Camera indexes or persistent device IDs")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=20)
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
    except ValueError as exc:
        parser.error(str(exc))
    if len({spec.source for spec in specs}) != len(specs):
        parser.error("Choose distinct sources.")

    cameras = Cameras(specs, simulation=False)
    frames: dict[str, dict[int, tuple[float, float, float]]] = {spec.name: {} for spec in specs}
    cameras.open()
    try:
        wait_until = time.monotonic() + 12
        while not all(status["fps"] for status in cameras.status().values()):
            if time.monotonic() >= wait_until:
                print(json.dumps(cameras.status(), indent=2))
                raise SystemExit("Not all cameras started delivering frames.")
            time.sleep(0.05)

        start = time.monotonic()
        finish = start + args.seconds
        while True:
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
            time.sleep(min(0.05, remaining))
        statuses = cameras.status()
    finally:
        cameras.close()

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
    if any(not status["ready"] for status in statuses.values()) or any(
        len(samples) < args.seconds * args.fps * 0.95 for samples in frames.values()
    ):
        raise SystemExit("At least one camera fell below 95% of the requested FPS.")


if __name__ == "__main__":
    main()
