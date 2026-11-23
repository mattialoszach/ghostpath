"""OpenCV fallback and the local frame transport for isolated macOS workers."""

import json
import socket
import sys
import time
from contextlib import suppress
from struct import Struct

import numpy as np

from .models import CameraSpec

HEADER = Struct("!cIdd")
MAX_MESSAGE = 4096


def _mac_camera_ids() -> list[str]:
    from AVFoundation import AVCaptureDevice, AVMediaTypeMuxed, AVMediaTypeVideo
    from objc import autorelease_pool

    with autorelease_pool():
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        devices += AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed)
        return sorted({str(device.uniqueID()) for device in devices})


def _select_mac_camera(ids: list[str], index: int, selected_id: str) -> tuple[int, str]:
    if selected_id:
        if selected_id not in ids:
            raise RuntimeError("The selected camera is disconnected. Check its USB cable.")
        return ids.index(selected_id), selected_id
    if index >= len(ids):
        raise RuntimeError(f"Camera source {index} is missing. Check the camera index.")
    return index, ids[index]


def capture_frames(cap, cv2, spec: CameraSpec, closed, publish_mode):
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {spec.source}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    cap.set(cv2.CAP_PROP_FPS, spec.capture_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    publish_mode({
        "label": str(spec.source),
        "width": spec.width,
        "height": spec.height,
        "nominal_fps": fps if np.isfinite(fps) and fps > 0 else None,
        "timestamp_source": "host_read",
    })
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
            height, width = bgr.shape[:2]
            raise RuntimeError(
                f"Camera delivered {width} × {height}, not the requested resolution."
            )
        yield stamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _send(
    sock: socket.socket,
    kind: bytes,
    payload: bytes | memoryview,
    timestamp: float = 0,
    arrival: float = 0,
):
    sock.sendall(HEADER.pack(kind, len(payload), timestamp, arrival))
    sock.sendall(payload)


def _run(sock: socket.socket, spec: CameraSpec, selected_id: str):
    from .mac_capture import capture

    ids = _mac_camera_ids()
    source = str(spec.source)
    _, device_id = _select_mac_camera(ids, int(source) if source.isdecimal() else 0, selected_id)
    _send(sock, b"I", device_id.encode())

    def publish_mode(mode):
        _send(sock, b"M", json.dumps(mode).encode())

    for stamp, arrival, pixels in capture(device_id, spec, publish_mode):
        _send(sock, b"F", memoryview(pixels).cast("B"), stamp, arrival)


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
