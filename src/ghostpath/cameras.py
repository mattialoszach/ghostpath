"""Independent capture with timestamped RGB frame histories and preview JPEGs."""

import io
import re
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Event, Thread

import numpy as np
from PIL import Image

from .camera_worker import HEADER, MAX_MESSAGE, capture_frames
from .models import CameraSpec
from .motion import Pose
from .preview import render_camera

STALE_AFTER = 0.30
REOPEN_DELAY = 1.0
FRAME_DELIVERY_GRACE = 0.012


def source_index(value: int | str) -> int | str:
    if isinstance(value, int) or value.isdecimal():
        index = int(value)
        if 0 <= index <= 63:
            return index
    elif re.fullmatch(r"/dev/video[0-9]{1,3}", value):
        return value
    raise ValueError("Use a camera index (0–63) or a Linux /dev/video device.")


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


def _match_frames(
    groups: list[list[CameraFrame]], stamp: float, tolerance: float
) -> list[CameraFrame] | None:
    if not groups or not all(groups):
        return None

    def closest(frames: list[CameraFrame]) -> CameraFrame:
        return min(frames, key=lambda frame: (abs(frame.timestamp - stamp), frame.sequence))

    nearest = [closest(frames) for frames in groups]
    times = [frame.timestamp for frame in nearest]
    if max(times) <= min(times) + tolerance:
        return nearest

    # A feasible set lies in a window starting at one of the candidate times.
    # Within each window, the closest frame from each camera minimizes the offset.
    starts = sorted({frame.timestamp for frames in groups for frame in frames})
    best = None
    best_score = (float("inf"), float("inf"), float("inf"))
    for start in starts:
        window = [
            [frame for frame in frames if start <= frame.timestamp <= start + tolerance]
            for frames in groups
        ]
        if not all(window):
            continue
        match = [closest(frames) for frames in window]
        offsets = [abs(frame.timestamp - stamp) for frame in match]
        times = [frame.timestamp for frame in match]
        score = (max(offsets), sum(offsets), max(times) - min(times))
        if score < best_score:
            best, best_score = match, score
    return best


class Capture:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.condition = Condition()
        self.history: deque[CameraFrame] = deque(maxlen=12)
        self.closed = Event()
        self.preview: bytes | None = None
        self.last_preview = 0.0
        self.problem = ""
        self.actual: float | None = None
        self.retrying = False
        self.sequence = 0
        self.thread = Thread(target=self._run, name=f"camera-{spec.name}", daemon=True)

    def _run(self):
        try:
            source = source_index(self.spec.source)
            if sys.platform == "darwin" and isinstance(source, int):
                self._run_macos()
                return

            import cv2

            cap = cv2.VideoCapture(source)
            try:
                for stamp, pixels in capture_frames(cap, cv2, self.spec, self.closed):
                    self._publish(stamp, pixels)
            finally:
                cap.release()
        except Exception as exc:
            self._clear_stream(str(exc) or type(exc).__name__)

    def _run_macos(self):
        device_id = ""
        while not self.closed.is_set():
            try:
                with _camera_process(self.spec, device_id) as receiver:
                    _, identity = self._packet(receiver, b"I", time.monotonic() + 6.0)
                    resolved_id = identity.decode()
                    if device_id and resolved_id != device_id:
                        raise RuntimeError("The capture process opened a different camera.")
                    device_id = resolved_id
                    timeout = 6.0
                    while not self.closed.is_set():
                        stamp, raw = self._packet(receiver, b"F", time.monotonic() + timeout)
                        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(
                            self.spec.height, self.spec.width, 3
                        )
                        self._publish(stamp, pixels)
                        timeout = 1.0
            except (OSError, RuntimeError, ValueError) as exc:
                if self.closed.is_set():
                    break
                self._clear_stream(str(exc) or type(exc).__name__, retrying=True)
            self.closed.wait(REOPEN_DELAY)

    def _packet(
        self, receiver: socket.socket, expected: bytes, deadline: float
    ) -> tuple[float, bytearray]:
        kind, length, stamp = HEADER.unpack(self._receive(receiver, HEADER.size, deadline))
        valid_size = (
            length == self.spec.height * self.spec.width * 3
            if kind == b"F"
            else 0 < length <= MAX_MESSAGE
        )
        if not valid_size or kind not in (expected, b"E"):
            raise RuntimeError("Could not read a valid camera frame.")
        payload = self._receive(receiver, length, deadline)
        if kind == b"E":
            raise RuntimeError(payload.decode(errors="replace"))
        return stamp, payload

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

    def _publish(self, stamp: float, pixels: np.ndarray):
        preview = None
        if stamp - self.last_preview >= 0.14:
            preview = jpeg(Image.fromarray(pixels), small=True)
            self.last_preview = stamp
        with self.condition:
            self.history.append(CameraFrame(self.sequence, stamp, pixels))
            self.sequence += 1
            self.problem = ""
            self.retrying = False
            if preview is not None:
                self.preview = preview
            if len(self.history) >= 8:
                self.actual = (len(self.history) - 1) / (stamp - self.history[0].timestamp)
            self.condition.notify_all()

    def _clear_stream(self, problem: str, *, retrying: bool = False):
        with self.condition:
            self.problem = problem
            self.retrying = retrying
            self.history.clear()
            self.preview = None
            self.last_preview = 0.0
            self.actual = None
            self.condition.notify_all()

    def _fresh(self, now: float) -> bool:
        return bool(
            not self.closed.is_set()
            and self.history
            and now - self.history[-1].timestamp < STALE_AFTER
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
                raise ValueError(f"{spec.name}: {status['problem'] or 'waiting for camera'}")
            if not status["fps"]:
                raise ValueError(f"{spec.name}: camera is warming up. Try again in a moment.")
            if status["fps"] < fps * 0.95:
                raise ValueError(f"{spec.name}: capture is too slow. Lower the dataset FPS.")

    def frames(
        self, timestamp: float, pose: Pose, period: float, after: dict[str, int]
    ) -> list[CameraFrame]:
        if not self.recorded:
            return []
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
            ]

        tolerance = period * 0.9
        cutoff = timestamp + tolerance
        deadline = cutoff + FRAME_DELIVERY_GRACE

        def collect(through: float) -> list[list[CameraFrame]]:
            return [
                self.captures[spec.name].candidates(
                    timestamp,
                    after.get(spec.name, -1),
                    tolerance,
                    through=through,
                    deadline=deadline,
                )
                for spec in self.recorded
            ]

        groups = collect(timestamp)
        if match := _match_frames(groups, timestamp, tolerance):
            return match

        # A camera may still deliver a compatible frame before the sampling
        # window closes. Keep earlier candidates even if its history rolls over.
        groups = [
            list({frame.sequence: frame for frame in (*early, *late)}.values())
            for early, late in zip(groups, collect(cutoff), strict=True)
        ]
        for spec, frames in zip(self.recorded, groups, strict=True):
            if not frames:
                raise ValueError(
                    f"{spec.name}: no unused camera frame within {tolerance * 1000:.0f} ms"
                    " of the robot sample"
                )
        if match := _match_frames(groups, timestamp, tolerance):
            return match
        offsets = ", ".join(
            f"{spec.name} {min((f.timestamp - timestamp for f in frames), key=abs) * 1000:+.1f} ms"
            for spec, frames in zip(self.recorded, groups, strict=True)
        )
        raise ValueError(
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
