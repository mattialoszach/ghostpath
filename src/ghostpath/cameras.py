"""Independent capture with timestamped RGB frame histories and preview JPEGs."""

import io
import json
import math
import re
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Event, Thread
from typing import Never

import numpy as np
from PIL import Image

from .camera_worker import HEADER, MAX_MESSAGE, capture_frames
from .models import CameraSpec
from .motion import Pose
from .preview import render_camera

STALE_AFTER = 0.30
REOPEN_DELAY = 1.0
FRAME_DELIVERY_GRACE = 0.012


class CameraNotReady(ValueError):
    """A camera is unavailable while a take is still preparing."""


class CameraSyncError(ValueError):
    """Live streams yielded no valid, unused set for an individual sample."""


def source_index(value: int | str) -> int | str:
    if isinstance(value, int) or value.isdecimal():
        index = int(value)
        if 0 <= index <= 63:
            return index
    elif re.fullmatch(r"/dev/video[0-9]{1,3}", value):
        return value
    elif sys.platform == "darwin" and re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", value):
        return value
    raise ValueError("Use a camera index (0–63), macOS device ID, or Linux /dev/video device.")


def jpeg(image: Image.Image, small: bool = False) -> bytes:
    if small:
        image.thumbnail((480, 360))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=79)
    return buffer.getvalue()


@contextmanager
def _camera_process(spec: CameraSpec, device_id: str):
    receiver, sender = socket.socketpair()
    with receiver, sender:
        receiver.settimeout(0.10)
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ghostpath.camera_worker",
                str(sender.fileno()),
                spec.model_dump_json(),
                device_id,
            ],
            stdin=subprocess.DEVNULL,
            pass_fds=(sender.fileno(),),
        )
        sender.close()
        try:
            yield receiver
        finally:
            receiver.close()
            try:
                worker.wait(timeout=0.12)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=0.18)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    timestamp: float
    pixels: np.ndarray
    arrival_timestamp: float | None = None
    available_timestamp: float | None = None

    @property
    def arrived(self) -> float:
        return self.timestamp if self.arrival_timestamp is None else self.arrival_timestamp

    @property
    def available(self) -> float:
        return self.arrived if self.available_timestamp is None else self.available_timestamp


def _match_frames(
    groups: list[list[CameraFrame]], stamp: float, tolerance: float
) -> list[CameraFrame] | None:
    if not groups or not all(groups):
        return None

    # Find the tightest window containing the robot sample and a frame from each camera.
    # Picking each camera's nearest frame independently can straddle the sample by a frame period.
    starts = {frame.timestamp for frames in groups for frame in frames if frame.timestamp < stamp}
    starts.add(stamp)
    best = None
    best_score = (float("inf"),) * 4
    for start in sorted(starts):
        window = [[frame for frame in frames if frame.timestamp >= start] for frames in groups]
        if not all(window):
            continue
        end = max(stamp, *(min(frame.timestamp for frame in frames) for frames in window))
        if end > start + tolerance:
            continue
        match = [
            min(
                (frame for frame in frames if frame.timestamp <= end),
                key=lambda frame: (abs(frame.timestamp - stamp), frame.sequence),
            )
            for frames in window
        ]
        offsets = [abs(frame.timestamp - stamp) for frame in match]
        times = [frame.timestamp for frame in match]
        span = max(stamp, *times) - min(stamp, *times)
        score = (span, max(offsets), sum(offsets), max(times) - min(times))
        if score < best_score:
            best, best_score = match, score
    return best


def _latest_match(groups: list[list[CameraFrame]], tolerance: float) -> list[CameraFrame] | None:
    """Find the latest feasible frame on each camera to leave room for the current sample."""
    if not groups:
        return None
    indices = [len(frames) - 1 for frames in groups]
    while all(index >= 0 for index in indices):
        match = [frames[index] for frames, index in zip(groups, indices, strict=True)]
        latest = max(range(len(match)), key=lambda i: match[i].timestamp)
        if match[latest].timestamp <= min(frame.timestamp for frame in match) + tolerance:
            return match
        # This frame is too new even for another camera's newest candidate.
        indices[latest] -= 1
    return None


def _merge_candidates(
    first: list[list[CameraFrame]], second: list[list[CameraFrame]]
) -> list[list[CameraFrame]]:
    return [
        sorted(
            {frame.sequence: frame for frame in (*early, *late)}.values(),
            key=lambda frame: frame.sequence,
        )
        for early, late in zip(first, second, strict=True)
    ]


class Capture:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.condition = Condition()
        self.history: deque[CameraFrame] = deque(maxlen=12)
        self.cadence: deque[float] = deque(maxlen=1 + 3 * spec.capture_fps)
        self.closed = Event()
        self.preview: bytes | None = None
        self.last_preview = 0.0
        self.problem = ""
        self.actual: float | None = None
        self.mode: dict | None = None
        self.retrying = False
        self.sequence = 0
        self.thread = Thread(target=self._run, name=f"camera-{spec.name}", daemon=True)

    def _run(self):
        try:
            source = source_index(self.spec.source)
            if sys.platform == "darwin":
                self._run_macos()
                return

            import cv2

            cap = cv2.VideoCapture(source)
            try:
                for stamp, pixels in capture_frames(cap, cv2, self.spec, self.closed, self._mode):
                    self._publish(stamp, pixels, stamp)
            finally:
                cap.release()
        except Exception as exc:
            self._clear_stream(str(exc) or type(exc).__name__)

    def _run_macos(self):
        source = str(self.spec.source)
        device_id = "" if source.isdecimal() else source
        while not self.closed.is_set():
            try:
                with _camera_process(self.spec, device_id) as receiver:
                    _, _, identity = self._packet(receiver, b"I", time.monotonic() + 6.0)
                    resolved_id = identity.decode()
                    if device_id and resolved_id != device_id:
                        raise RuntimeError("The capture process opened a different camera.")
                    device_id = resolved_id
                    _, _, mode = self._packet(receiver, b"M", time.monotonic() + 8.0)
                    self._mode(json.loads(mode))
                    timeout = 6.0
                    while not self.closed.is_set():
                        stamp, arrival, raw = self._packet(
                            receiver, b"F", time.monotonic() + timeout
                        )
                        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(
                            self.spec.height, self.spec.width, 3
                        )
                        self._publish(stamp, pixels, arrival)
                        timeout = 1.0
            except (OSError, RuntimeError, ValueError) as exc:
                if self.closed.is_set():
                    break
                self._clear_stream(str(exc) or type(exc).__name__, retrying=True)
            self.closed.wait(REOPEN_DELAY)

    def _packet(
        self, receiver: socket.socket, expected: bytes, deadline: float
    ) -> tuple[float, float, bytearray]:
        kind, length, stamp, arrival = HEADER.unpack(
            self._receive(receiver, HEADER.size, deadline)
        )
        valid_size = (
            length == self.spec.height * self.spec.width * 3
            if kind == b"F"
            else 0 < length <= MAX_MESSAGE
        )
        if not valid_size or kind not in (expected, b"E"):
            raise RuntimeError("Could not read a valid camera frame.")
        if kind == b"F" and not (math.isfinite(stamp) and math.isfinite(arrival)):
            raise RuntimeError("The camera delivered an invalid timestamp.")
        payload = self._receive(receiver, length, deadline)
        if kind == b"E":
            raise RuntimeError(payload.decode(errors="replace"))
        return stamp, arrival, payload

    def _receive(self, receiver: socket.socket, length: int, deadline: float) -> bytearray:
        data = bytearray(length)
        view = memoryview(data)
        while view:
            if self.closed.is_set():
                raise InterruptedError("Camera capture is closing.")
            if time.monotonic() >= deadline:
                raise TimeoutError("The camera stopped delivering frames.")
            try:
                size = receiver.recv_into(view)
            except TimeoutError:
                continue
            if not size:
                raise RuntimeError("The camera capture process ended.")
            view = view[size:]
        return data

    def _mode(self, mode: dict):
        with self.condition:
            self.mode = mode

    def _publish(self, stamp: float, pixels: np.ndarray, arrival: float):
        with self.condition:
            if self.cadence and stamp <= self.cadence[-1]:
                return
            available = time.monotonic()
            self.history.append(CameraFrame(self.sequence, stamp, pixels, arrival, available))
            self.cadence.append(stamp)
            self.sequence += 1
            self.problem = ""
            self.retrying = False
            span = stamp - self.cadence[0]
            if span >= 1:
                self.actual = (len(self.cadence) - 1) / span
            self.condition.notify_all()
        if available - self.last_preview >= 0.14:
            preview = jpeg(Image.fromarray(pixels), small=True)
            with self.condition:
                self.preview = preview
                self.last_preview = available

    def _clear_stream(self, problem: str, *, retrying: bool = False):
        with self.condition:
            self.problem = problem
            self.retrying = retrying
            self.history.clear()
            self.cadence.clear()
            self.preview = None
            self.last_preview = 0.0
            self.actual = None
            self.mode = None
            self.condition.notify_all()

    def _fresh(self, now: float) -> bool:
        return bool(
            not self.closed.is_set()
            and self.history
            and now - self.history[-1].available < STALE_AFTER
            and not self.problem
        )

    def live_preview(self) -> bytes | None:
        with self.condition:
            return self.preview if self._fresh(time.monotonic()) else None

    def candidates(
        self, stamp: float, after: int, tolerance: float, *, through: float, deadline: float
    ) -> list[CameraFrame]:
        with self.condition:
            while not self.closed.is_set() and not self.problem:
                ready = (
                    self.history
                    and self.history[-1].sequence > after
                    and self.history[-1].timestamp >= through
                )
                remaining = deadline - time.monotonic()
                if ready or remaining <= 0:
                    break
                self.condition.wait(remaining)
            if self.closed.is_set() or self.problem:
                raise ValueError(f"{self.spec.name}: {self.problem or 'camera capture closed'}")
            return [
                f
                for f in self.history
                if f.sequence > after and stamp - tolerance <= f.timestamp <= stamp + tolerance
            ]

    def status(self) -> dict:
        with self.condition:
            ready = self._fresh(time.monotonic())
            return {
                "ready": ready,
                "retrying": self.retrying,
                "problem": self.problem
                or ("Camera frame is stale." if self.history and not ready else ""),
                "fps": round(self.actual, 1) if self.actual and ready else 0,
                "mode": self.mode,
            }


class Cameras:
    def __init__(self, specs: list[CameraSpec], simulation: bool):
        self.specs = specs
        self.recorded = [s for s in specs if s.record]
        self.simulation = simulation
        self.captures = {} if simulation else {s.name: Capture(s) for s in specs}
        self.indices = {s.name: i for i, s in enumerate(specs)}
        self.previews: dict[str, bytes] = {}
        self.demo_pose: Pose | None = None
        self.closed = Event()
        self.thread = Thread(target=self._simulate_preview, daemon=True)

    def open(self):
        if self.simulation:
            self.thread.start()
        else:
            for capture in self.captures.values():
                capture.thread.start()

    def _simulate_preview(self):
        while not self.closed.is_set():
            pose = self.demo_pose
            if pose is not None:
                self.previews = {
                    s.name: jpeg(render_camera(pose, 480, 360, self.indices[s.name]))
                    for s in self.specs
                }
            self.closed.wait(0.14)

    def preview(self, name: str) -> bytes | None:
        capture = self.captures.get(name)
        return capture.live_preview() if capture else self.previews.get(name)

    def status(self) -> dict:
        if self.simulation:
            return {
                s.name: {"ready": True, "problem": "", "fps": s.capture_fps} for s in self.specs
            }
        return {name: capture.status() for name, capture in self.captures.items()}

    def check_recording(self, fps: int):
        statuses = self.status()
        for spec in self.recorded:
            status = statuses[spec.name]
            if not status["ready"]:
                raise CameraNotReady(f"{spec.name}: {status['problem'] or 'waiting for camera'}")
            if not status["fps"]:
                raise CameraNotReady(f"{spec.name}: camera is warming up. Try again in a moment.")
            if status["fps"] < fps * 0.95:
                raise CameraNotReady(
                    f"{spec.name}: capturing {status['fps']:g} FPS for a {fps} FPS dataset. "
                    "Lower the dataset FPS."
                )

    def frames(
        self,
        timestamp: float,
        pose: Pose,
        period: float,
        after: dict[str, int],
        *,
        next_timestamp: float | None = None,
        reserved: list[CameraFrame] | None = None,
    ) -> tuple[list[CameraFrame], list[CameraFrame] | None]:
        if not self.recorded:
            return [], None
        if self.simulation:
            return [
                CameraFrame(
                    after.get(spec.name, -1) + 1,
                    timestamp,
                    np.asarray(
                        render_camera(pose, spec.width, spec.height, self.indices[spec.name])
                    ),
                )
                for spec in self.recorded
            ], None

        tolerance = period * 0.9

        def collect(stamp: float, through: float) -> list[list[CameraFrame]]:
            return [
                self.captures[spec.name].candidates(
                    stamp,
                    after.get(spec.name, -1),
                    tolerance,
                    through=through,
                    deadline=stamp + tolerance + FRAME_DELIVERY_GRACE,
                )
                for spec in self.recorded
            ]

        groups = collect(timestamp, timestamp + tolerance)
        if reserved:
            groups = _merge_candidates(groups, [[frame] for frame in reserved])
        match = _match_frames(groups, timestamp, tolerance)
        if match is None:
            self._raise_sync_error(groups, timestamp, tolerance)
        if next_timestamp is None:
            return match, None

        following = collect(next_timestamp, next_timestamp)

        def next_match(current: list[CameraFrame]) -> list[CameraFrame] | None:
            later = [
                [frame for frame in frames if frame.sequence > used.sequence]
                for frames, used in zip(following, current, strict=True)
            ]
            return _match_frames(later, next_timestamp, tolerance)

        if future := next_match(match):
            return match, future
        following = _merge_candidates(
            following, collect(next_timestamp, next_timestamp + tolerance)
        )
        if future := next_match(match):
            return match, future

        latest = _latest_match(following, tolerance)
        if latest is not None:
            earlier = [
                [frame for frame in frames if frame.sequence < upcoming.sequence]
                for frames, upcoming in zip(groups, latest, strict=True)
            ]
            alternative = _match_frames(earlier, timestamp, tolerance)
            if alternative is not None:
                return alternative, next_match(alternative)
        return match, None

    def _raise_sync_error(
        self, groups: list[list[CameraFrame]], timestamp: float, tolerance: float
    ) -> Never:
        for spec, frames in zip(self.recorded, groups, strict=True):
            if not frames:
                raise CameraSyncError(
                    f"{spec.name}: no unused camera frame within {tolerance * 1000:.0f} ms"
                    " of the robot sample"
                )
        offsets = ", ".join(
            f"{spec.name} {min((f.timestamp - timestamp for f in frames), key=abs) * 1000:+.1f} ms"
            for spec, frames in zip(self.recorded, groups, strict=True)
        )
        raise CameraSyncError(
            f"camera-to-camera skew is too high for the {tolerance * 1000:.0f} ms limit "
            f"(nearest individual frames relative to the robot sample: {offsets})"
        )

    def close(self):
        self.closed.set()
        for capture in self.captures.values():
            capture.closed.set()
            with capture.condition:
                capture.condition.notify_all()
        for thread in (self.thread, *(c.thread for c in self.captures.values())):
            if thread.is_alive():
                thread.join(timeout=0.7 if sys.platform == "darwin" else 0.25)
