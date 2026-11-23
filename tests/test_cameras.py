import signal
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep

import numpy as np
import pytest

from ghostpath import cameras
from ghostpath.camera_check import compare_pairing
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
    history.append(cameras.CameraFrame(0, stamp - 0.06, image, available_timestamp=stamp - 0.005))
    history.append(cameras.CameraFrame(1, stamp - 0.03, image, available_timestamp=stamp + 0.005))
    history.append(cameras.CameraFrame(2, stamp + 0.01, image, available_timestamp=stamp + 0.03))
    assert [frame.sequence for frame in manager.frames(stamp, {}, {})] == [0]
    assert [frame.sequence for frame in manager.frames(stamp + 0.006, {}, {"workspace": 0})] == [1]


@pytest.mark.parametrize(("buffer", "expected"), [(0.105, 0), (0.075, 1)])
def test_avfoundation_pairs_at_a_stable_input_delay(buffer, expected):
    manager = cameras.Cameras(
        [CameraSpec(name="workspace", source=0)], simulation=False, avfoundation_buffer=buffer
    )
    stamp = monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = manager.captures["workspace"]
    capture.mode = {"timestamp_source": "avfoundation_pts"}
    capture.history.append(cameras.CameraFrame(
        0, stamp - 0.11, image, arrival_timestamp=stamp - 0.03
    ))
    capture.history.append(cameras.CameraFrame(
        1, stamp - 0.08, image, arrival_timestamp=stamp - 0.01
    ))
    hits = manager.frames(stamp + 0.025, {}, {}, scheduled=stamp)
    assert [frame.sequence for frame in hits] == [expected]


def test_trace_observes_each_publication_even_after_history_evicts_frames():
    published = []
    capture = cameras.Capture(
        CameraSpec(name="workspace", source=0, width=160, height=120),
        on_frame=lambda name, frame: published.append((name, frame)),
    )
    pixels = np.zeros((120, 160, 3), dtype=np.uint8)
    for sequence in range(20):
        capture._publish(sequence / 30, pixels, sequence / 30)
    assert [frame.sequence for name, frame in published if name == "workspace"] == list(range(20))
    assert capture.history[0].sequence > 0
    assert all(not hasattr(frame, "pixels") for _, frame in published)


def test_pairing_comparison_accounts_for_frames_that_arrive_late_and_cannot_be_reused():
    frames = {
        name: [
            cameras.CameraTiming(sequence, stamp + shift, available_timestamp=available + shift)
            for sequence, (stamp, available) in enumerate([
                (-0.130, -0.08), (-0.098, -0.046), (-0.065, -0.01),
                (-0.032, 0.059), (0.001, 0.080),
            ])
        ]
        for name, shift in [("workspace", 0), ("wrist", -0.002)]
    }
    comparison = {
        item["buffer_ms"]: item for item in compare_pairing(
            frames, [(0, 0), (0.05, 0.05), (0.10, 0.10)], set(frames), 0.035
        )
    }
    assert {buffer: result["missed_rows"] for buffer, result in comparison.items()} == {
        105: 0, 90: 0, 75: 0, 60: 1, 0: 1,
    }
    baseline_age = comparison[105]["image_age_ms"]["workspace"]["p50"]
    assert comparison[75]["image_age_ms"]["workspace"]["p50"] < baseline_age
    assert comparison[75]["camera_skew_ms"]["max"] == 2.0


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
