import json
from threading import Event
from time import monotonic

import numpy as np
import pytest

from ghostpath.camera_check import _check_export
from ghostpath.cameras import CameraFrame, Cameras
from ghostpath.dataset import Datasets, Record
from ghostpath.models import CameraSpec, Configuration
from ghostpath.motion import JOINTS
from ghostpath.robot import SimulatedFollower
from ghostpath.session import CONTROL


@pytest.fixture
def writer(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    config = Configuration(
        fps=15,
        cameras=[
            CameraSpec(name="front", source=0, width=160, height=120),
            CameraSpec(name="preview", source=1, width=160, height=120, record=False),
        ],
    )
    store = Datasets(tmp_path / "demo", config, Cameras(config.cameras, simulation=True))
    try:
        yield store
    finally:
        store.close()


def stage(writer, manual=False):
    writer.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    reference = None if manual else {**pose, "shoulder_pan": 2, "gripper": 67}
    for i in range(6):
        command = {**pose, "shoulder_pan": 5 + i}
        writer.offer(Record(100 + i / 15, pose, command, manual, reference))
    return writer.seal().result(timeout=30)


def test_official_v3_roundtrip_discard_and_failed_save(writer, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    quality = stage(writer)
    assert quality["review_frames"] == 2
    assert quality["capture_span_seconds"] == 0.333
    assert quality["max_arm_error_deg"] == 10
    assert quality["max_arm_path_error_deg"] == 2
    assert quality["arm_path_peak"] == {"joint": "shoulder_pan", "seconds": 0}
    assert quality["max_gripper_path_error_pct"] == 2
    assert writer.review("front", 0).startswith(b"\xff\xd8")
    assert writer.review("preview", 0) is None
    writer.discard().result(timeout=30)
    assert writer.accepted == 0
    stage(writer)
    start = SimulatedFollower(CONTROL).read()
    rest = {**start, "shoulder_pan": -12.0}
    writer.save("pass-1", "teach", start_pose=start, rest_pose=rest).result(timeout=30)

    stage(writer)
    committed = {file: file.read_bytes() for file in writer._output_files()}
    save_episode = writer.dataset.save_episode

    def fail_after_write(**kwargs):
        save_episode(**kwargs)
        raise OSError("Injected write failure")

    monkeypatch.setattr(writer.dataset, "save_episode", fail_after_write)
    with pytest.raises(OSError, match="Injected"):
        writer.save("bad-pass", "hybrid").result(timeout=30)
    assert writer.accepted == 1
    assert committed == {file: file.read_bytes() for file in writer._output_files()}
    assert stage(writer, manual=True)["max_arm_path_error_deg"] is None
    writer.save("pass-2", "hybrid").result(timeout=30)
    writer.finalize().result(timeout=30)

    data = LeRobotDataset(writer.repo_id, root=writer.root, video_backend="pyav", return_uint8=True)
    assert data.meta.info.codebase_version == "v3.0"
    assert data.features["action"]["names"][0] == "shoulder_pan.pos"
    assert data.num_frames == 12
    assert data.num_episodes == 2
    manifest = json.loads((writer.root / "ghostpath.json").read_text())
    policy = manifest["policy_io"]
    assert policy["fps"] == data.fps
    assert policy["cameras"] == [
        {"name": "front", "key": "observation.images.front", "width": 160, "height": 120,
         "capture_fps": 30, "color_order": "RGB"},
    ]
    assert policy["joint_order"] == list(JOINTS)
    assert policy["units"]["shoulder_pan"] == "degrees"
    assert policy["units"]["gripper"] == "percent_calibrated_travel"
    assert policy["state"]["key"] == "observation.state"
    assert policy["action"] == {"key": "action", "kind": "absolute_servo_position_goal"}
    assert manifest["observation_policy"]["max_camera_skew_seconds"] == pytest.approx(0.035)
    assert not manifest["observation_policy"]["reuse_frames"]
    assert manifest["episodes"][0]["poses"] == {"task_start": start, "rest": rest}
    assert manifest["episodes"][1]["poses"]["rest"] is None
    assert not {"port", "calibration_dir"} & manifest["config"].keys()
    assert "observation.images.preview" not in data.features
    first, last = data[0], data[11]
    assert first["observation.images.front"].shape == (3, 120, 160)
    assert first["observation.state"][0] == 0
    assert first["action"][0] == 5
    assert first["task"] == writer.config.task
    assert not first["ghostpath.manual"].item()
    assert last["episode_index"] == 1
    assert last["ghostpath.manual"].item()
    assert last["ghostpath.actual_timestamp"] == pytest.approx(5 / 15)


def test_rejects_out_of_order_rather_than_manufacturing_a_frame(writer):
    writer.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    writer.offer(Record(100, pose, pose, False))
    writer.offer(Record(99, pose, pose, False))
    with pytest.raises(ValueError, match="missed a recording deadline"):
        writer.seal().result(timeout=30)
    assert writer.accepted == 0


def test_rejects_an_invalid_encoder_read_interval(writer):
    writer.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    writer.offer(Record(100, pose, pose, False, encoder_read=(100.002, 100.001)))
    with pytest.raises(ValueError, match="invalid encoder read interval"):
        writer.seal().result(timeout=30)


def test_preflight_roundtrips_images_and_causal_timestamps(writer):
    stage(writer)
    result = _check_export(writer, expected=6)
    assert result["frames"] == 6
    assert result["video_bytes"] > 0
    assert result["dataset_fps"] == 15
    manifest = json.loads((writer.root / "ghostpath.json").read_text())
    assert manifest["observation_policy"]["avfoundation_buffer_seconds"] == (
        writer.cameras.avfoundation_buffer
    )


def test_rejects_video_and_telemetry_frame_count_mismatch(writer, monkeypatch):
    stage(writer)
    save_episode = writer.dataset.save_episode

    def save_with_mismatch(**kwargs):
        save_episode(**kwargs)
        writer.frames += 1

    monkeypatch.setattr(writer.dataset, "save_episode", save_with_mismatch)
    with pytest.raises(RuntimeError, match="encoder lost video frames"):
        writer.save("invalid", "teach").result(timeout=30)
    assert not writer._output_files()
    assert not (writer.root / "ghostpath.json").exists()


def test_rejects_a_truncated_telemetry_table(writer, monkeypatch):
    import pyarrow.parquet as pq

    stage(writer)
    finalize = writer.dataset.finalize

    def finalize_with_truncated_telemetry():
        finalize()
        if tables := sorted((writer.root / "data").rglob("*.parquet")):
            table = pq.read_table(tables[-1])
            pq.write_table(table.slice(0, table.num_rows - 1), tables[-1])

    monkeypatch.setattr(writer.dataset, "finalize", finalize_with_truncated_telemetry)
    with pytest.raises(RuntimeError, match="expected number of telemetry rows"):
        writer.save("invalid", "teach").result(timeout=30)
    assert not writer._output_files()
    assert not (writer.root / "ghostpath.json").exists()


@pytest.fixture
def physical_writer(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    config = Configuration(
        fps=20,
        cameras=[
            CameraSpec(name="workspace", source=0, width=160, height=120),
            CameraSpec(name="wrist", source=1, width=160, height=120),
        ],
    )
    store = Datasets(tmp_path / "paired", config, Cameras(config.cameras, simulation=False))
    try:
        yield store
    finally:
        store.close()


def load_camera_history(store, stamp, streams):
    for name, offsets in streams.items():
        capture = store.cameras.captures[name]
        capture.actual = 30.0
        for i, offset in enumerate(offsets):
            pixels = np.full((120, 160, 3), 20 * i, dtype=np.uint8)
            capture.history.append(CameraFrame(i, stamp + offset, pixels))


def test_writer_uses_original_camera_frames_even_when_history_is_evicted(physical_writer):
    store = physical_writer
    first = monotonic() - 0.20
    read_centers = [first + index / 20 for index in range(3)]
    for capture in store.cameras.captures.values():
        capture.actual = 30.0
        for index, stamp in enumerate(read_centers):
            capture.history.append(CameraFrame(
                index, stamp - 0.060, np.full((120, 160, 3), index * 100, dtype=np.uint8),
                arrival_timestamp=stamp - 0.010, available_timestamp=stamp - 0.008,
            ))
        capture.history.append(CameraFrame(
            len(read_centers), read_centers[-1] - 0.020, np.zeros((120, 160, 3), dtype=np.uint8),
            arrival_timestamp=read_centers[-1], available_timestamp=read_centers[-1] + 0.002,
        ))
    store.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    release = Event()
    blocker = store.executor.submit(release.wait, 5)
    try:
        for stamp in read_centers:
            store.offer(Record(stamp, pose, pose, False, encoder_read=(stamp - .001, stamp + .001)))
        for capture in store.cameras.captures.values():
            capture.history.clear()
    finally:
        release.set()
    assert blocker.result(timeout=5)
    quality = store.seal().result(timeout=30)
    assert quality["skipped_camera_samples"] == 0
    assert quality["frames"] == len(read_centers)
    assert quality["encoder_read_ms"]["p50"] == 2.0
    timing = quality["camera_timing"]["workspace"]
    assert timing["image_age_ms"]["max"] == 61.0
    assert timing["camera_callback_ms"]["p50"] == 50.0
    assert timing["app_delivery_ms"]["p50"] == 2.0
    assert timing["arrival_lead_ms"]["p50"] == 11.0
    assert timing["wait_ms"]["p50"] == 9.0


@pytest.mark.parametrize(
    ("streams", "expected", "skew"),
    [
        (
            {"workspace": [-0.024, 0.038, 0.080, 0.120], "wrist": [-0.041, 0.025, 0.072, 0.112]},
            [(-0.024, -0.041), (0.038, 0.025), (0.080, 0.072)],
            17.0,
        ),
        (
            {
                "workspace": [-0.017, 0.016, 0.049, 0.082, 0.115],
                "wrist": [-0.015, 0.013, 0.051, 0.084, 0.117],
            },
            [(-0.017, -0.015), (0.016, 0.013), (0.082, 0.084)],
            3.0,
        ),
    ],
    ids=["causal-skew", "avoid-mixing-camera-intervals"],
)
def test_joint_pairing_saves_episode_with_distinct_frames(physical_writer, streams, expected, skew):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    store = physical_writer
    stamp = monotonic() - 0.17
    load_camera_history(store, stamp, streams)
    store.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    for i in range(3):
        store.offer(Record(stamp + i / 20, pose, pose, False))
    quality = store.seal().result(timeout=30)
    assert quality["camera_skew_ms"] == pytest.approx(skew)
    assert quality["skipped_camera_samples"] == 0
    view = quality["camera_timing"]["workspace"]
    expected_age = max(index / 20 - pair[0] for index, pair in enumerate(expected)) * 1000
    assert view["image_age_ms"]["max"] == pytest.approx(expected_age)
    assert view["image_age_ms"] == view["wait_ms"]
    assert view["delivery_ms"]["max"] == 0
    source = streams["workspace"]
    intervals = source.index(expected[-1][0]) - source.index(expected[0][0])
    assert view["fps"] == round(intervals / (expected[-1][0] - expected[0][0]), 2)
    store.save("paired-cameras", "teach").result(timeout=30)
    store.finalize().result(timeout=30)

    dataset = module.LeRobotDataset(store.repo_id, root=store.root, video_backend="pyav")
    assert dataset.num_frames == 3
    for row, timestamps in zip(dataset, expected, strict=True):
        assert row["ghostpath.camera_timestamps"].tolist() == pytest.approx(timestamps)
        cutoff = row["ghostpath.encoder_read"][1]
        assert row["ghostpath.camera_available_timestamps"].max() <= cutoff


def test_isolated_wrist_gap_is_reported_and_keeps_actual_camera_and_robot_times(physical_writer):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    store = physical_writer
    stamp = monotonic() - 0.24
    load_camera_history(
        store, stamp, {"workspace": [0, 0.05, 0.10, 0.15, 0.20], "wrist": [0, 0.05, 0.15, 0.20]}
    )
    store.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    for i in range(5):
        store.offer(Record(stamp + i / 20, pose, pose, False))
    quality = store.seal().result(timeout=30)
    assert quality["frames"] == 4
    assert quality["skipped_camera_samples"] == 1
    assert quality["max_gap_ms"] == 100.0
    store.save("isolated-gap", "teach").result(timeout=30)
    store.finalize().result(timeout=30)
    manifest = json.loads((store.root / "ghostpath.json").read_text())
    assert manifest["episodes"][0]["skipped_camera_samples"] == 1
    assert manifest["episodes"][0]["discontinuities"] == [2]

    dataset = module.LeRobotDataset(store.repo_id, root=store.root, video_backend="pyav")
    assert dataset.num_frames == 4
    for row, actual in zip(dataset, [0, 0.05, 0.15, 0.20], strict=True):
        assert row["ghostpath.actual_timestamp"].item() == pytest.approx(actual)
        assert row["ghostpath.camera_timestamps"].tolist() == pytest.approx([actual, actual])
    assert [row["ghostpath.sample_index"].item() for row in dataset] == [0, 1, 3, 4]
    assert [row["ghostpath.segment_index"].item() for row in dataset] == [0, 0, 1, 1]


def test_repeated_wrist_gaps_reject_the_take(physical_writer):
    store = physical_writer
    stamp = monotonic() - 0.24
    load_camera_history(store, stamp, {"workspace": [0, 0.05, 0.10, 0.16], "wrist": [0, 0.16]})
    store.prepare().result(timeout=30)
    pose = SimulatedFollower(CONTROL).read()
    for i in range(3):
        store.offer(Record(stamp + i / 20, pose, pose, False))
    with pytest.raises(ValueError, match="Repeated camera gaps: wrist"):
        store.seal().result(timeout=30)
    assert store.accepted == 0
