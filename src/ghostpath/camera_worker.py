"""OpenCV capture and the local frame transport for isolated macOS workers."""

import socket
import sys
import time
from contextlib import suppress
from struct import Struct

import numpy as np

from .models import CameraSpec

HEADER = Struct("!cId")
MAX_MESSAGE = 4096


def _mac_camera_ids() -> list[str]:
    from AVFoundation import AVCaptureDevice, AVMediaTypeMuxed, AVMediaTypeVideo
    from objc import autorelease_pool

    # Match OpenCV's AVFoundation index ordering.
    with autorelease_pool():
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        devices += AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed)
        return sorted(str(device.uniqueID()) for device in devices)


def _select_mac_camera(ids: list[str], index: int, selected_id: str) -> tuple[int, str]:
    if selected_id:
        if selected_id not in ids:
            raise RuntimeError("The selected camera is disconnected. Check its USB cable.")
        return ids.index(selected_id), selected_id
    if index >= len(ids):
        raise RuntimeError(f"Camera source {index} is missing. Check the camera index.")
    return index, ids[index]


def capture_frames(cap, cv2, spec: CameraSpec, closed=None):
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {spec.source}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    cap.set(cv2.CAP_PROP_FPS, spec.capture_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    failures = 0
    while closed is None or not closed.is_set():
        ok, bgr = cap.read()
        stamp = time.monotonic()
        if closed is not None and closed.is_set():
            break
        if not ok or bgr is None:
            failures += 1
            if failures >= 10:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.03)
            continue
        failures = 0
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise RuntimeError("The camera did not deliver 8-bit RGB-compatible images.")
        if bgr.shape[:2] != (spec.height, spec.width):
            bgr = cv2.resize(bgr, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
        yield stamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _send(sock: socket.socket, kind: bytes, payload: bytes | memoryview, timestamp: float = 0):
    sock.sendall(HEADER.pack(kind, len(payload), timestamp))
    sock.sendall(payload)


def _run(sock: socket.socket, spec: CameraSpec, selected_id: str):
    import cv2

    ids = _mac_camera_ids()
    index, device_id = _select_mac_camera(ids, int(spec.source), selected_id)
    _send(sock, b"I", device_id.encode())
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        # Discard the stream if the device list changed while OpenCV was opening.
        if _mac_camera_ids() != ids:
            raise RuntimeError("The camera list changed while opening the feed.")
        for stamp, pixels in capture_frames(cap, cv2, spec):
            _send(sock, b"F", memoryview(pixels).cast("B"), stamp)
    finally:
        cap.release()


def main():
    _, fd, config, device_id = sys.argv
    with socket.socket(fileno=int(fd)) as sock:
        try:
            _run(sock, CameraSpec.model_validate_json(config), device_id)
        except Exception as exc:
            with suppress(OSError):
                _send(sock, b"E", (str(exc) or type(exc).__name__).encode()[:MAX_MESSAGE])


if __name__ == "__main__":
    main()
