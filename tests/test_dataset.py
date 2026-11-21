import pytest

from ghostpath.cameras import Cameras
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
