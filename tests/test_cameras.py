import signal
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep

import numpy as np
import pytest

from ghostpath import cameras
from ghostpath.camera_worker import _select_mac_camera
from ghostpath.models import CameraSpec

WORKER = """
import socket
import sys
import time
from ghostpath.camera_worker import _send

with socket.socket(fileno=int(sys.argv[1])) as sock:
    _send(sock, b"I", b"camera-a")
    _send(sock, b"M", b'{"device_id":"camera-a","width":160,"height":120}')
    while True:
        stamp = time.monotonic()
        _send(sock, b"F", bytes(160 * 120 * 3), stamp, stamp)
        time.sleep(0.03)
"""


def wait_for(predicate):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("Expected camera state did not arrive")


def test_macos_source_stays_bound_to_physical_camera():
    selected = _select_mac_camera(["camera-a", "camera-b"], 0, "")[1]
    with pytest.raises(RuntimeError, match="disconnected"):
        _select_mac_camera(["camera-b"], 0, selected)
    assert _select_mac_camera(["a-new-camera", "camera-a", "camera-b"], 0, selected) == (
        1,
        "camera-a",
    )


@pytest.mark.parametrize(
    ("after", "error"),
    [
        ({}, r"skew exceeds 35 ms"),
        ({"workspace": 0}, r"workspace: no fresh, unused camera frame"),
    ],
)
def test_pairing_rejects_excess_skew_or_a_reused_frame(after, error):
    manager = cameras.Cameras(
        [CameraSpec(name="workspace", source=0), CameraSpec(name="wrist", source=1)],
        simulation=False,
    )
    stamp = monotonic() - 1
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for name, offset in (("workspace", -0.080), ("wrist", -0.010)):
        manager.captures[name].history.append(cameras.CameraFrame(0, stamp + offset, image))
    with pytest.raises(ValueError, match=error):
        manager.frames(stamp, {}, after)


def test_pairing_cannot_see_frames_that_were_not_yet_available():
    manager = cameras.Cameras([CameraSpec(name="workspace", source=0)], simulation=False)
    stamp = monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    history = manager.captures["workspace"].history
    history.append(cameras.CameraFrame(0, stamp - 0.06, image, stamp - 0.01, stamp - 0.005))
    history.append(cameras.CameraFrame(1, stamp - 0.03, image, stamp - 0.002, stamp + 0.005))
    history.append(cameras.CameraFrame(2, stamp + 0.01, image, stamp + 0.02, stamp + 0.03))
    assert [frame.sequence for frame in manager.frames(stamp, {}, {})] == [0]
    assert [frame.sequence for frame in manager.frames(stamp + 0.006, {}, {"workspace": 0})] == [1]


def test_avfoundation_pairs_at_a_stable_input_delay():
    manager = cameras.Cameras([CameraSpec(name="workspace", source=0)], simulation=False)
    stamp = monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = manager.captures["workspace"]
    capture.mode = {"timestamp_source": "avfoundation_pts"}
    capture.history.append(cameras.CameraFrame(0, stamp - 0.11, image, stamp - 0.03))
    capture.history.append(cameras.CameraFrame(1, stamp - 0.08, image, stamp - 0.01))
    hits = manager.frames(stamp + 0.025, {}, {}, scheduled=stamp)
    assert [frame.sequence for frame in hits] == [0]


@pytest.mark.skipif(sys.platform == "win32", reason="Requires inherited Unix sockets")
def test_restarts_blocked_camera_process_and_discards_stale_frames(monkeypatch):
    workers = []
    identities = []
    popen = subprocess.Popen

    def fake_worker(args, **kwargs):
        identities.append(args[-1])
        worker = popen([sys.executable, "-c", WORKER, args[3]], **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(cameras.subprocess, "Popen", fake_worker)
    monkeypatch.setattr(cameras, "REOPEN_DELAY", 0.25)
    capture = cameras.Capture(CameraSpec(name="workspace", source=0, width=160, height=120))
    try:
        capture.thread = Thread(target=capture._run_macos, daemon=True)
        capture.thread.start()
        wait_for(lambda: capture.live_preview() is not None)
        with capture.condition:
            first = capture.history[-1]

        workers[0].send_signal(signal.SIGSTOP)

        wait_for(lambda: capture.status()["retrying"])
        assert capture.live_preview() is None
        with pytest.raises(ValueError, match="stopped delivering frames"):
            capture.candidates(first.timestamp, -1)

        wait_for(lambda: len(workers) >= 2 and capture.live_preview() is not None)
        assert workers[0].poll() is not None
        with capture.condition:
            recovered = capture.history[-1]
        assert recovered.sequence > first.sequence
        assert recovered.timestamp > first.timestamp
        assert capture.status()["ready"]
        assert identities == ["", "camera-a"]
    finally:
        capture.closed.set()
        if capture.thread.is_alive():
            capture.thread.join(timeout=2)
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
        assert not capture.thread.is_alive()
