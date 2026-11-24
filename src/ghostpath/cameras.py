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
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import numpy as np
from PIL import Image

from .camera_worker import HEADER, MAX_MESSAGE, capture_frames
from .models import DEFAULT_MAC_CAMERA_BUFFER_MS, CameraSpec
from .motion import Pose
from .preview import render_camera

REOPEN_DELAY = 1.0
MAX_FRAME_AGE = 0.175
AVFOUNDATION_BUFFER = DEFAULT_MAC_CAMERA_BUFFER_MS / 1000
FRAME_HISTORY_SIZE = 12
PREVIEW_INTERVAL = 0.14


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
class CameraTiming:
    sequence: int
    timestamp: float
    arrival_timestamp: float | None = field(default=None, kw_only=True)
    available_timestamp: float | None = field(default=None, kw_only=True)

    @property
    def arrived(self) -> float:
        return self.timestamp if self.arrival_timestamp is None else self.arrival_timestamp

    @property
    def available(self) -> float:
        return self.arrived if self.available_timestamp is None else self.available_timestamp


@dataclass(frozen=True)
class CameraFrame(CameraTiming):
    pixels: np.ndarray


def camera_candidates[Frame: CameraTiming](
    history: Iterable[Frame], cutoff: float, after: int, scheduled: float, buffer: float
) -> list[Frame]:
    latest = min(cutoff, scheduled - buffer) if buffer else cutoff
    return [
        frame for frame in history
        if frame.sequence > after
        and cutoff - MAX_FRAME_AGE <= frame.timestamp <= latest
        and frame.available <= cutoff
    ]


def pairing_tolerance(specs: list[CameraSpec]) -> float:
    return 1.05 / min(spec.capture_fps for spec in specs)


def latest_match[Frame: CameraTiming](
    groups: list[list[Frame]], tolerance: float
) -> list[Frame] | None:
    """Find the newest set that satisfies the camera-to-camera skew limit."""
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


def timing_stats(values: Iterable[float]) -> dict[str, float]:
    percentiles = np.percentile(tuple(values), (50, 95, 99, 100))
    return dict(zip(
        ("p50", "p95", "p99", "max"),
        (round(float(value) * 1000, 1) for value in percentiles),
        strict=True,
    ))


class Capture:
    def __init__(
        self, spec: CameraSpec, on_frame: Callable[[str, CameraTiming], None] | None = None
    ):
        self.spec = spec
        self.on_frame = on_frame
        self.lock = Lock()
        self.history: deque[CameraFrame] = deque(maxlen=FRAME_HISTORY_SIZE)
        self.cadence: deque[float] = deque(maxlen=1 + 3 * spec.capture_fps)
        self.closed = Event()
        self.preview: bytes | None = None
        self.stream_generation = 0
        self.problem = ""
        self.actual: float | None = None
        self.mode: dict | None = None
        self.retrying = False
        self.sequence = 0
        self.thread = Thread(target=self._run, name=f"camera-{spec.name}", daemon=True)
        self.preview_thread = Thread(
            target=self._preview_loop, name=f"camera-preview-{spec.name}", daemon=True
        )

    def open(self):
        self.thread.start()
        self.preview_thread.start()

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
        with self.lock:
            self.mode = mode

    def _publish(self, stamp: float, pixels: np.ndarray, arrival: float):
        pixels.setflags(write=False)
        with self.lock:
            available = time.monotonic()
            if not (math.isfinite(stamp) and stamp <= arrival <= available):
                raise RuntimeError("The camera delivered timestamps out of order.")
            if self.cadence and stamp <= self.cadence[-1]:
                return
            self.history.append(CameraFrame(
                self.sequence, stamp, pixels,
                arrival_timestamp=arrival, available_timestamp=available,
            ))
            sequence = self.sequence
            self.cadence.append(stamp)
            self.sequence += 1
            self.problem = ""
            self.retrying = False
            span = stamp - self.cadence[0]
            if span >= 1:
                self.actual = (len(self.cadence) - 1) / span
        if self.on_frame is not None:
            self.on_frame(self.spec.name, CameraTiming(
                sequence, stamp, arrival_timestamp=arrival, available_timestamp=available
            ))

    def _preview_loop(self):
        last_sequence = -1
        while not self.closed.is_set():
            with self.lock:
                frame = self.history[-1] if self._fresh(time.monotonic()) else None
                generation = self.stream_generation
            if frame is not None and frame.sequence != last_sequence:
                try:
                    preview = jpeg(Image.fromarray(frame.pixels), small=True)
                except (OSError, ValueError):
                    preview = None
                with self.lock:
                    if generation == self.stream_generation:
                        self.preview = preview
                last_sequence = frame.sequence
            self.closed.wait(PREVIEW_INTERVAL)

    def _clear_stream(self, problem: str, *, retrying: bool = False):
        with self.lock:
            self.problem = problem
            self.retrying = retrying
            self.history.clear()
            self.cadence.clear()
            self.preview = None
            self.stream_generation += 1
            self.actual = None
            self.mode = None

    def _fresh(self, now: float) -> bool:
        return bool(
            not self.closed.is_set()
            and self.history
            and now - self.history[-1].timestamp <= MAX_FRAME_AGE
            and not self.problem
        )

    def live_preview(self) -> bytes | None:
        with self.lock:
            return self.preview if self._fresh(time.monotonic()) else None

    def candidates(
        self, cutoff: float, after: int, *,
        scheduled: float | None = None, buffer: float = AVFOUNDATION_BUFFER,
    ) -> list[CameraFrame]:
        with self.lock:
            if self.closed.is_set() or self.problem:
                raise ValueError(f"{self.spec.name}: {self.problem or 'camera capture closed'}")
            if not self.mode or self.mode.get("timestamp_source") != "avfoundation_pts":
                buffer = 0
            return camera_candidates(
                self.history, cutoff, after, cutoff if scheduled is None else scheduled, buffer
            )

    def status(self) -> dict:
        with self.lock:
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
    def __init__(
        self, specs: list[CameraSpec], simulation: bool, *,
        on_frame: Callable[[str, CameraTiming], None] | None = None,
        avfoundation_buffer: float = AVFOUNDATION_BUFFER,
    ):
        self.specs = specs
        self.recorded = [s for s in specs if s.record]
        self.simulation = simulation
        self.avfoundation_buffer = avfoundation_buffer
        self.tolerance = pairing_tolerance(self.recorded) if self.recorded else 0.0
        self.captures = {} if simulation else {s.name: Capture(s, on_frame) for s in specs}
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
                capture.open()

    def _simulate_preview(self):
        while not self.closed.is_set():
            pose = self.demo_pose
            if pose is not None:
                self.previews = {
                    s.name: jpeg(render_camera(pose, 480, 360, self.indices[s.name]))
                    for s in self.specs
                }
            self.closed.wait(PREVIEW_INTERVAL)

    def preview(self, name: str) -> bytes | None:
        capture = self.captures.get(name)
        return capture.live_preview() if capture else self.previews.get(name)

    def status(self) -> dict:
        if self.simulation:
            return {
                s.name: {"ready": True, "problem": "", "fps": s.capture_fps} for s in self.specs
            }
        return {name: capture.status() for name, capture in self.captures.items()}

    def check_recording(self, fps: int) -> dict:
        statuses = self.status()
        seen = set()
        for spec in self.recorded:
            status = statuses[spec.name]
            if not status["ready"]:
                raise CameraNotReady(f"{spec.name}: {status['problem'] or 'waiting for camera'}")
            if not status["fps"]:
                raise CameraNotReady(f"{spec.name}: camera is warming up. Try again in a moment.")
            if status["fps"] < fps:
                raise CameraNotReady(
                    f"{spec.name}: capturing {status['fps']:g} FPS for a {fps} FPS dataset. "
                    "Lower the dataset FPS."
                )
            device_id = (status.get("mode") or {}).get("device_id")
            if device_id and device_id in seen:
                raise CameraNotReady(f"{spec.name}: another view uses the same physical camera.")
            if device_id:
                seen.add(device_id)
        return {spec.name: statuses[spec.name] for spec in self.recorded}

    def frames(
        self,
        cutoff: float,
        pose: Pose,
        after: dict[str, int],
        *,
        scheduled: float | None = None,
    ) -> list[CameraFrame]:
        if not self.recorded:
            return []
        if self.simulation:
            return [
                CameraFrame(
                    after.get(spec.name, -1) + 1,
                    cutoff,
                    np.asarray(
                        render_camera(pose, spec.width, spec.height, self.indices[spec.name])
                    ),
                )
                for spec in self.recorded
            ]

        groups = []
        for spec in self.recorded:
            frames = self.captures[spec.name].candidates(
                cutoff, after.get(spec.name, -1),
                scheduled=scheduled, buffer=self.avfoundation_buffer,
            )
            if not frames:
                raise CameraSyncError(
                    f"{spec.name}: no fresh, unused camera frame at the observation cutoff"
                )
            groups.append(frames)
        if match := latest_match(groups, self.tolerance):
            return match
        raise CameraSyncError(
            f"Camera-to-camera skew exceeds {self.tolerance * 1000:.0f} ms "
            "at the observation cutoff."
        )

    def close(self):
        self.closed.set()
        for capture in self.captures.values():
            capture.closed.set()
        workers = [self.thread]
        for capture in self.captures.values():
            workers.extend((capture.thread, capture.preview_thread))
        for thread in workers:
            if thread.is_alive():
                thread.join(timeout=0.7 if sys.platform == "darwin" else 0.25)
