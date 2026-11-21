from time import monotonic

import numpy as np
import pytest

from ghostpath.cameras import CameraFrame, Cameras
from ghostpath.dataset import Datasets, Record
from ghostpath.models import CameraSpec, Configuration
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
    for i in range(6):
        command = {**pose, "shoulder_pan": 5 + i}
        writer.offer(Record(100 + i / 15, pose, command, manual))
    return writer.seal().result(timeout=30)


def test_official_v3_roundtrip_discard_and_failed_save(writer, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert stage(writer)["review_frames"] == 2
    assert writer.review("front", 0).startswith(b"\xff\xd8")
    assert writer.review("preview", 0) is None
    writer.discard().result(timeout=30)
    assert writer.accepted == 0
    stage(writer)
    writer.save("pass-1", "teach").result(timeout=30)

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
    stage(writer, manual=True)
    writer.save("pass-2", "hybrid").result(timeout=30)
    writer.finalize().result(timeout=30)

    data = LeRobotDataset(writer.repo_id, root=writer.root, video_backend="pyav", return_uint8=True)
    assert data.meta.info.codebase_version == "v3.0"
    assert data.features["action"]["names"][0] == "shoulder_pan.pos"
    assert data.num_frames == 12
    assert data.num_episodes == 2
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


def test_joint_pairing_saves_an_episode_when_individual_nearest_frames_are_too_far_apart(tmp_path):
    module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    config = Configuration(
        fps=20,
        cameras=[
            CameraSpec(name="workspace", source=0, width=160, height=120),
            CameraSpec(name="wrist", source=1, width=160, height=120),
        ],
    )
    cameras = Cameras(config.cameras, simulation=False)
    store = Datasets(tmp_path / "paired", config, cameras)
    stamp = monotonic() - 0.17
    streams = {"workspace": [-0.024, 0.038, 0.080, 0.120], "wrist": [-0.041, 0.025, 0.072, 0.112]}
    try:
        for name, offsets in streams.items():
            capture = cameras.captures[name]
            capture.actual = 30.0
            for i, offset in enumerate(offsets):
                pixels = np.full((120, 160, 3), 20 * i, dtype=np.uint8)
                capture.history.append(CameraFrame(i, stamp + offset, pixels))
        store.prepare().result(timeout=30)
        pose = SimulatedFollower(CONTROL).read()
        for i in range(3):
            store.offer(Record(stamp + i / 20, pose, pose, False))
        assert store.seal().result(timeout=30)["camera_skew_ms"] == pytest.approx(13.0)
        store.save("paired-cameras", "teach").result(timeout=30)
        store.finalize().result(timeout=30)

        dataset = module.LeRobotDataset(store.repo_id, root=store.root, video_backend="pyav")
        assert dataset.num_frames == 3
        for row, expected in zip(
            dataset, [(0.038, 0.025), (0.080, 0.072), (0.120, 0.112)], strict=True
        ):
            assert row["ghostpath.camera_timestamps"].tolist() == pytest.approx(expected)
    finally:
        store.close()
