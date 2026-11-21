import signal
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep

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
    while True:
        _send(sock, b"F", bytes(160 * 120 * 3), time.monotonic())
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
            capture.nearest(first.timestamp, -1, 0.1)

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
