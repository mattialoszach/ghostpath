import json
from concurrent.futures import Future
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ghostpath.api import create_app
from ghostpath.cameras import CameraNotReady, Cameras
from ghostpath.grip import CarryGripPath, GrippedPath
from ghostpath.models import Command, Configuration, Phase, ReplayProfile
from ghostpath.motion import JOINTS, Motion
from ghostpath.poses import write_json
from ghostpath.robot import SimulatedFollower
from ghostpath.session import CONTROL, FAST_REPLAY, Session, WorkflowError
from ghostpath.torque import TorqueManager
from ghostpath.trajectory import ReversePath, Sample, Teaching


def wait_for(ws, predicate):
    deadline = monotonic() + 8
    while monotonic() < deadline:
        ws.send_json({"action": "heartbeat"})
        event = ws.receive_json()
        assert event["type"] != "error", event
        if event["type"] == "state" and predicate(event["state"]):
            return event["state"]
    raise AssertionError("Expected workflow transition")


def test_default_collection_uses_two_vga_cameras_with_rate_headroom():
    config = Configuration()
    assert config.fps == 20
    assert config.mac_camera_buffer_ms == 75
    assert [camera.name for camera in config.cameras] == ["workspace", "wrist"]
    assert all(
        camera.record and (camera.width, camera.height, camera.capture_fps) == (640, 480, 30)
        for camera in config.cameras
    )


def test_connection_uses_the_selected_mac_camera_buffer(tmp_path):
    session = Session(tmp_path)
    config = Configuration(cameras=[], mac_camera_buffer_ms=68)
    try:
        session._command(Command(action="connect", config=config))
        assert session.cameras.avfoundation_buffer == pytest.approx(0.068)
        assert session._snapshot()["config"]["mac_camera_buffer_ms"] == 68
    finally:
        session._close_devices()


def test_recording_picks_the_closer_control_tick_with_matching_action(tmp_path):
    session = Session(tmp_path)
    session.config = Configuration(fps=20)
    records = []
    session.datasets = SimpleNamespace(error=None, offer=records.append)
    for i, stamp in enumerate((0.0, 0.017, 0.032, 0.043, 0.061, 0.077, 0.094, 0.112)):
        session.pose = {"gripper": float(i)}
        session._record(stamp, {"gripper": float(i + 1)}, reference={"gripper": float(i + 2)})
    assert [frame.timestamp for frame in records] == [0.0, 0.043, 0.094]
    assert [frame.state["gripper"] for frame in records] == [0, 3, 6]
    assert [frame.action["gripper"] for frame in records] == [1, 4, 7]
    assert [frame.reference["gripper"] for frame in records] == [2, 5, 8]


@pytest.mark.parametrize("powered_reset", [False, True])
def test_discard_episode_preserves_start_library_and_saved_count(tmp_path, powered_reset):
    session = Session(tmp_path)
    session.config = Configuration(cameras=[])
    session._command(Command(action="connect"))
    try:
        session._command(Command(action="capture_start"))
        session._command(Command(action="save_start", pose_label="Demo start"))
        selected = session.start_pose_id
        session._command(Command(action="teach"))
        pose = {**session.pose, "wrist_flex": 5.0}
        if powered_reset:
            session.robot.teach(pose)
            session._stop_teaching()
            # A compiled library entry is available after a completed teaching attempt.
            samples = [
                Sample(i / 60, {**session.start_pose, "wrist_flex": i / 8}) for i in range(41)
            ]
            session.teaching_origin = session.start_pose
            ident = session._save_raw(samples, complete=True)
            session._set_path(session._compile_path(samples, session.start_pose), ident)
            session.heartbeat()
            session._command(Command(action="auto_reset"))
            assert session.phase == Phase.RETURNING
        session.datasets = SimpleNamespace(accepted=2, quality=None)
        with pytest.raises(WorkflowError, match="Confirm"):
            session._command(Command(action="discard_episode"))
        session._command(Command(action="discard_episode", confirmed=True))
        assert session.phase == Phase.READY
        assert session.teacher is None and session.player is None and session.path is None
        assert session.start_pose_id == selected and session.datasets.accepted == 2
        assert session.power.state == ("on" if powered_reset else "off")
        if powered_reset:
            assert session._path_file(ident).exists()
            session._refresh_paths()
            assert {entry["id"] for entry in session.paths} == {ident}
            assert set(session.saved_starts) == {selected}
    finally:
        session._close_devices()


@pytest.mark.parametrize("during_prepare", [True, False])
def test_camera_not_ready_returns_to_reset_before_motion(tmp_path, during_prepare):
    session = Session(tmp_path)
    session.phase = Phase.PROCESSING
    session.cameras = Cameras(session.config.cameras, simulation=False)
    prepared = Future()
    if during_prepare:
        prepared.set_exception(CameraNotReady("wrist: waiting for camera"))
    else:
        prepared.set_result(None)
    session.operation = ("prepare", prepared)

    session._check_operation()

    assert session.phase == Phase.RESET
    assert session.operation is None
    assert session.player is None
    assert session.notice.startswith("Recording could not start.")


def test_tcp_workspace_disconnect_stops_torque_off_teaching(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        with client.websocket_connect(
            "ws://localhost/ws", headers={"origin": "http://localhost"}
        ) as ws:
            wait_for(ws, lambda s: s["phase"] == "setup")
            ws.send_json({"action": "connect"})
            wait_for(ws, lambda s: s["phase"] == "ready")
            ws.send_json({"action": "capture_start"})
            wait_for(ws, lambda s: s["at_start"])
            ws.send_json({"action": "teach"})
            state = wait_for(ws, lambda s: s["phase"] == "teaching" and s["elapsed"] > 0.1)
            assert state["torque"] == "off"
        with client.websocket_connect(
            "ws://localhost/ws", headers={"origin": "http://localhost"}
        ) as ws:
            state = wait_for(ws, lambda s: s["phase"] == "fault")
            assert state["torque"] == "off"
            assert not state["frames"]
        assert len(list((tmp_path / "paths").glob("*.json"))) == 1


def test_hardware_return_requires_confirmed_opt_in_and_power(tmp_path):
    session = Session(tmp_path)
    session.config = Configuration(device="so101", port="simulated-test-port", cameras=[])
    robot = SimulatedFollower(CONTROL)
    session.robot = robot
    session.power = TorqueManager(robot)
    session.start_pose = robot.read()
    robot.teach({**session.start_pose, "wrist_flex": 12.0})
    session.pose = robot.read()
    session.motion = Motion(session.pose, robot.limits, CONTROL)
    taught_start = {**session.start_pose, "wrist_flex": 1.5}
    session.path = SimpleNamespace(start=taught_start, end=session.pose)
    session.phase = Phase.RESET

    command = Command(action="auto_reset", confirmed=True, allow_auto_reset_once=True)
    with pytest.raises(WorkflowError, match="Turn on torque"):
        session._command(command)
    robot.arm()
    with pytest.raises(WorkflowError, match="Opt in"):
        session._command(Command(action="auto_reset", confirmed=True))
    with pytest.raises(WorkflowError, match="clear the entire return path"):
        session._command(Command(action="auto_reset", allow_auto_reset_once=True))
    assert session.phase == Phase.RESET
    assert robot.target == session.pose

    session.heartbeat()
    session._command(command)
    assert session.phase == Phase.RETURNING
    assert session.player.path.at(0) == session.pose
    assert session.player.path.at(session.player.path.duration) == taught_start
    assert session.player.settle_seconds >= 0.5
    assert not session.config.automatic_reset


def test_automatic_return_enables_replay_after_measured_settle(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws") as ws:
            ws.send_json({"action": "connect"})
            wait_for(ws, lambda s: s["phase"] == "ready")
            ws.send_json({"action": "capture_start"})
            wait_for(ws, lambda s: s["at_start"])
            ws.send_json({"action": "demo_path"})
            taught = wait_for(ws, lambda s: s["phase"] == "teach_paused")
            assert not taught["at_start"]
            ws.send_json({"action": "auto_reset"})
            wait_for(ws, lambda s: s["phase"] == "returning")
            settled = wait_for(ws, lambda s: s["phase"] == "reset")
            assert settled["at_start"]
            assert settled["frames"] == 0
            assert settled["path"]["id"] == taught["path"]["id"]
            assert settled["pose"] == pytest.approx(taught["reset_target"], abs=2.0)


def test_boomerang_can_start_at_distinct_rest_and_runs_before_recording(tmp_path):
    session = Session(tmp_path)
    session._command(Command(action="connect"))
    try:
        session.start_pose = session.robot.read()
        session.robot.teach({**session.start_pose, "shoulder_pan": -6.0})
        session._command(Command(action="capture_rest"))
        assert session.rest_pose["shoulder_pan"] == -6
        assert session._pose_file(rest=True).exists()
        assert session.start_pose["shoulder_pan"] == 0
        samples = [
            Sample(i / 60, {**session.start_pose, "wrist_flex": i / 10, "gripper": 65 - i / 5})
            for i in range(41)
        ]
        session._set_path(session._compile_path(samples, session.start_pose), "reverse-demo")
        session.robot.teach({**session.start_pose, "shoulder_pan": 12.0})
        session.heartbeat()
        with pytest.raises(WorkflowError, match="from the start pose or taught endpoint"):
            session._command(Command(action="reverse_reset"))
        session.robot.teach(samples[-1].pose)
        session.heartbeat()
        session._command(Command(action="hold_torque"))
        assert session.phase == Phase.RESET
        session._command(Command(action="park"))
        deadline = monotonic() + 8
        while session.phase == Phase.RETURNING and monotonic() < deadline:
            session.heartbeat()
            session._tick(monotonic())
            sleep(1 / CONTROL.fps)
        assert session._snapshot()["at_rest"]
        assert not session._snapshot()["at_start"]
        session._command(Command(action="reverse_reset", firm_carry_grip=True))
        assert session.reset_action == "reverse_approach"
        stages = set()
        deadline = monotonic() + 10
        while session.phase == Phase.RETURNING and monotonic() < deadline:
            stages.add(session.reset_action)
            if session.reset_action == "reverse":
                assert isinstance(session.player.path.path, CarryGripPath)
                assert session.notice.startswith("No carry qualified for extra grip.")
            session.heartbeat()
            session._tick(monotonic())
            sleep(1 / CONTROL.fps)
        assert stages == {"reverse_approach", "reverse"}
        assert session.phase == Phase.RESET
        assert session.scene_reversed
        assert session._snapshot()["firm_carry_grip"]
        assert session.datasets is None
        assert session._snapshot()["at_start"]
        assert session.pose == pytest.approx(samples[0].pose, abs=2)
        session._command(Command(action="release_torque"))
        assert session.power.state == "off"
        assert session.phase == Phase.RESET
    finally:
        session._close_devices()


def test_resuming_teaching_excludes_pause_and_preserves_prior_samples():
    teacher = Teaching()
    pose = dict.fromkeys(JOINTS, 0.0)
    teacher.resume(10.0, pose, transition=0.08)
    teacher.append(10.5, {**pose, "wrist_flex": 4.0})
    checkpoint = teacher.samples.copy()
    teacher.resume(1000.0, checkpoint[-1].pose, transition=0.08)
    teacher.append(1000.2, {**pose, "wrist_flex": 7.0})
    assert teacher.samples[:2] == checkpoint
    assert [sample.t for sample in teacher.samples] == pytest.approx([0, 0.5, 0.58, 0.78])


def test_continue_appends_to_the_same_path_at_the_selected_start(tmp_path):
    session = Session(tmp_path)
    session.config = Configuration(cameras=[])
    session._command(Command(action="connect"))
    try:
        rest = {**session.pose, "shoulder_pan": 8.0}
        session.robot.teach(rest)
        session._command(Command(action="capture_start"))
        session.robot.teach({**rest, "shoulder_pan": 9.0})
        session._command(Command(action="teach"))

        def guide(offset):
            for i in range(1, 19):
                sleep(0.018)
                session.heartbeat()
                session.robot.teach({**rest, "wrist_flex": offset + i * 0.3})
                session._tick(monotonic())
            session._command(Command(action="stop_teach"))

        guide(0)
        assert session.phase == Phase.TEACH_PAUSED
        assert session.power.state == "on"
        assert session._snapshot()["can_continue"]
        earlier = json.loads(session._path_file(session.path_id).read_text())
        assert earlier["start_pose"] == rest
        assert earlier["samples"][0]["pose"]["shoulder_pan"] == 9
        assert session.path.start == rest
        session.heartbeat()
        session._command(Command(action="continue_teach"))
        assert session.power.state == "off"
        guide(5.4)
        latest = json.loads(session._path_file(session.path_id).read_text())
        assert latest["samples"][: len(earlier["samples"])] == earlier["samples"]
        assert latest["samples"][-1]["pose"]["wrist_flex"] == pytest.approx(10.8)
        assert session.path.start == rest
        session._command(Command(action="reverse_reset"))
        assert session.reset_action == "reverse"
        assert isinstance(session.player.path.path, ReversePath)
        assert session.teacher is None
        assert session.datasets is None
    finally:
        session._close_devices()


def test_manual_practice_can_hold_or_park_without_saving_poses_or_data(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws") as ws:
            ws.send_json({"action": "connect", "config": {"cameras": []}})
            wait_for(ws, lambda state: state["phase"] == "ready")
            ws.send_json({"action": "capture_rest"})
            rest = wait_for(ws, lambda state: state["rest_target"])["rest_target"]
            rest_files = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
            ws.send_json({"action": "practice"})
            practice = wait_for(ws, lambda state: state["phase"] == "practice")
            assert practice["torque"] == "on" and not practice["recording"]
            ws.send_json({"action": "jog", "axes": {"shoulder_pan": 1}})
            moved = wait_for(ws, lambda state: state["pose"]["shoulder_pan"] > 0.15)
            ws.send_json({"action": "finish_practice"})
            held = wait_for(ws, lambda state: state["phase"] == "ready")
            assert held["pose"]["shoulder_pan"] >= moved["pose"]["shoulder_pan"]
            assert held["start_pose"] is None
            ws.send_json({"action": "practice"})
            wait_for(ws, lambda state: state["phase"] == "practice")
            ws.send_json({"action": "park"})
            wait_for(ws, lambda state: state["phase"] == "returning")
            parked = wait_for(ws, lambda state: state["phase"] == "ready" and state["at_rest"])
            assert parked["pose"]["shoulder_pan"] == pytest.approx(rest["shoulder_pan"], abs=0.15)
            assert parked["frames"] == 0
            ws.send_json({"action": "practice"})
            wait_for(ws, lambda state: state["phase"] == "practice")
            ws.send_json({"action": "stop"})
            stopped = wait_for(ws, lambda state: state["phase"] == "fault")
            assert stopped["fault"] == "Stopped by the operator."
    assert rest_files == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert rest_files
    assert not (tmp_path / "datasets").exists()


@pytest.mark.parametrize("release", ["normal", "blur", "stale"])
def test_manual_jog_eases_normal_release_but_holds_on_blur_or_stale_input(tmp_path, release):
    session = Session(tmp_path)
    session.config = Configuration(cameras=[])
    session._command(Command(action="connect"))
    try:
        session._command(Command(action="practice"))
        joint = "shoulder_pan"
        session._command(Command(action="jog", axes={joint: 1}))
        period = 1 / CONTROL.fps

        def tick(now):
            session.heartbeat()
            session._manual_tick(now, period, now, (now, now))
            session.pose = session.manual_goal.copy()

        for step in range(1, 10):
            tick(session.jog_at + step * period)
        before = session.manual_goal[joint]
        speed = session.motion.velocity[joint]
        if release == "stale":
            tick(session.jog_at + 0.26)
        else:
            session._command(Command(action="jog", immediate_stop=release == "blur"))
            started = session.jog_at
            tick(started + period)
            if release == "normal":
                assert session.manual_goal[joint] > before
                for step in range(2, 5):
                    tick(started + step * period)
                assert 0 < session.motion.velocity[joint] < speed
                for step in range(5, 26):
                    tick(started + step * period)
                assert session.manual_goal[joint] - before < 1.5
            else:
                assert session.manual_goal[joint] == before
        assert session.motion.stopped
        assert not session.jog_active and not session.axes
    finally:
        session._close_devices()


@pytest.mark.parametrize("handoff_kind", ["automatic", "fault_in_review", "takeover"])
def test_replay_records_initial_alignment_and_manual_handoff(
    tmp_path, monkeypatch, handoff_kind
):
    early = handoff_kind == "takeover"
    lose_arm_in_review = handoff_kind == "fault_in_review"
    dataset_module = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    rest = SimulatedFollower(CONTROL).read()
    save_id = "20261120T000000-abcdef12"
    write_json(
        tmp_path / "paths" / f"{save_id}.json",
        {
            "version": 1,
            "robot_id": "simulation",
            "calibration": "simulation",
            "start_pose": rest,
            "complete": True,
            "task": "move",
            "created": "2026-11-20T00:00:00Z",
            "samples": [
                {"t": i / 60, "pose": {**rest, "wrist_flex": 1.2 + i * 0.12}} for i in range(40)
            ],
        },
    )
    app = create_app(tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws") as ws:
            ws.send_json(
                {
                    "action": "connect",
                    "config": {"cameras": [], "fps": 20, "replay_profile": "fast"},
                }
            )
            wait_for(ws, lambda s: s["phase"] == "ready")
            ws.send_json({"action": "demo_move", "axes": {"shoulder_pan": -5}})
            parking = wait_for(ws, lambda s: s["pose"]["shoulder_pan"] == -5)["pose"]
            ws.send_json({"action": "capture_rest"})
            wait_for(ws, lambda s: s["rest_target"] == parking)
            ws.send_json({"action": "demo_move", "axes": {"shoulder_pan": 0}})
            ws.send_json({"action": "load_path", "path_id": save_id})
            ready = wait_for(ws, lambda s: s["phase"] == "reset" and s["at_start"])
            assert ready["path"]["teaching_duration"] == 0.65
            assert ready["path"]["replay_profile"] == "fast"
            ws.send_json({"action": "replay", "manual_finish": not early})
            aligning = wait_for(ws, lambda s: s["replay_stage"] == "aligning" and s["frames"] > 0)
            assert aligning["recording"]
            assert aligning["progress"] == 0
            assert app.state.session.robot.settings == FAST_REPLAY
            if early:
                wait_for(ws, lambda s: s["replay_stage"] == "path" and 0.15 < s["progress"] < 0.8)
                ws.send_json({"action": "take_over", "confirmed": True})
            handoff = wait_for(ws, lambda s: s["phase"] == "manual")
            assert handoff["recording"]
            if early:
                assert handoff["progress"] < 1
            else:
                assert handoff["progress"] == pytest.approx(1)
            assert app.state.session.robot.settings == CONTROL
            ws.send_json({"action": "jog", "axes": {"gripper": -1}})
            wait_for(ws, lambda s: s["phase"] == "manual" and s["pose"]["gripper"] < 64.9)
            ws.send_json({"action": "jog", "axes": {}})
            ws.send_json({"action": "finish_manual"})
            review = wait_for(ws, lambda s: s["phase"] == "review")
            quality = review["quality"]
            timing = quality["replay_timing"]
            if early:
                assert timing is None
            else:
                assert timing["alignment_seconds"] > 0
                assert timing["motion_seconds"] == pytest.approx(
                    ready["path"]["duration"], abs=0.15
                )
                assert timing["settle_seconds"] >= 0.55
            assert quality["max_arm_path_error_deg"] is not None
            if lose_arm_in_review:
                def unavailable():
                    raise OSError("Motor link disconnected")

                monkeypatch.setattr(app.state.session.robot, "read", unavailable)
                stopped = wait_for(ws, lambda s: s["fault"] != "")
                assert stopped["phase"] == "review"
                assert stopped["quality"] == review["quality"]
                with pytest.raises(WorkflowError, match="Robot control is stopped"):
                    app.state.session._command(Command(action="hold_torque"))

            ws.send_json({"action": "save"})
            next_phase = "fault" if lose_arm_in_review else "next"
            wait_for(ws, lambda s: s["phase"] == next_phase and s["accepted"] == 1)
            if lose_arm_in_review:
                ws.send_json({"action": "finish"})
                wait_for(ws, lambda s: s["phase"] == "finished")

    root = tmp_path / "datasets" / "block-placement"
    dataset = dataset_module.LeRobotDataset(
        "local/block-placement", root=root, video_backend="pyav"
    )
    manual = [frame["ghostpath.manual"].item() for frame in dataset]
    assert manual == sorted(manual) and not manual[0] and manual[-1]
    indices = [frame["ghostpath.sample_index"].item() for frame in dataset]
    assert indices == list(range(len(dataset)))
    assert dataset[0]["observation.state"][3].item() == pytest.approx(0, abs=0.05)
    manifest = json.loads((root / "ghostpath.json").read_text())
    entries = manifest["episodes"]
    assert manifest["policy_io"]["robot"]["calibration_hash"] == "simulation"
    assert manifest["policy_io"]["robot"]["position_limits"]["gripper"] == [0, 100]
    assert entries[0]["poses"] == {"task_start": ready["reset_target"], "rest": parking}
    assert entries[0]["poses"]["task_start"]["wrist_flex"] == pytest.approx(1.2)
    assert entries[0]["manual_finish"] is not early
    assert entries[0]["replay_timing"] == timing
    control = json.loads((root / entries[0]["control_trace"]).read_text())
    ticks = control["ticks"]
    assert control["nominal_hz"] == CONTROL.fps
    assert len(ticks) > dataset.num_frames
    if early:
        takeover = next(i for i, tick in enumerate(ticks) if tick["stage"] == "manual")
        assert ticks[takeover - 1]["stage"] == "path"
        assert all(tick["stage"] == "manual" for tick in ticks[takeover:])
        assert ticks[takeover]["action"] == ticks[takeover]["state"]
        assert entries[0]["manual_takeover_seconds"] == pytest.approx(
            ticks[takeover]["timestamp"], abs=0.001
        )
    else:
        assert "manual_takeover_seconds" not in entries[0]
        assert {tick["stage"] for tick in ticks} >= {"path", "settling", "manual"}
    assert all(
        tick["encoder_read"][0] <= tick["timestamp"] <= tick["encoder_read"][1] for tick in ticks
    )
    sent = [tick for tick in ticks if tick["command_send"] is not None]
    assert sent
    assert all(tick["encoder_read"][1] <= tick["command_send"][0] for tick in sent)
    assert any(tick["replay_rate"] is not None and tick["reference"] for tick in ticks)


@pytest.mark.parametrize("legacy_timing", [True, False])
def test_saved_grip_edits_follow_the_path_timing_used_to_make_them(tmp_path, legacy_timing):
    session = Session(tmp_path)
    original = ReplayProfile.STANDARD if legacy_timing else ReplayProfile.FAST
    session.config = Configuration(cameras=[], replay_profile=original)
    session._command(Command(action="connect"))
    try:
        start = session.pose
        samples = [
            Sample(i / 60, {**start, "elbow_flex": min(42, max(0, (i - 16) * 1.4))})
            for i in range(121)
        ]
        session.teaching_origin = start
        ident = session._save_raw(samples, complete=True)
        base = session._compile_path(samples, start, exact_start=True, legacy_timing=legacy_timing)
        knots = [(0, 0), (0.5, -4), (1, 0)]
        edited = GrippedPath(base, knots, base.settings)
        if legacy_timing:
            # Files from earlier versions only stored the fractional grip edit times.
            saved = json.loads(session._path_file(ident).read_text())
            write_json(session._path_file(ident), {**saved, "grip_edits": knots})
        else:
            session._set_path(base, ident)
            session.grip_assist = SimpleNamespace(result=lambda _: edited)
            session._finish_grip_assist()
            ident = session.path_id
        session.config.replay_profile = (
            ReplayProfile.FAST if legacy_timing else ReplayProfile.STANDARD
        )
        session._load_path(ident)
        assert session.path.base.legacy_timing is legacy_timing
        assert session._snapshot()["path"]["replay_profile"] == original
        assert session.path.duration == pytest.approx(edited.duration)
        for step in range(11):
            t = edited.duration * step / 10
            assert session.path.at(t) == pytest.approx(edited.at(t))
    finally:
        session._close_devices()


def test_current_and_saved_poses_are_selectable_without_a_rest_pose(tmp_path):
    session = Session(tmp_path)
    session.config = Configuration(cameras=[])
    session._command(Command(action="connect"))
    try:
        assert session._snapshot()["start_pose"] is None
        with pytest.raises(WorkflowError, match="Choose a start pose"):
            session._command(Command(action="teach"))

        session._command(Command(action="capture_start"))
        assert session.start_pose_id is None
        assert not session._start_library().exists()
        assert not session._pose_file(rest=True).exists()
        with pytest.raises(WorkflowError, match="Save a rest pose"):
            session._command(Command(action="park"))
        session._command(Command(action="save_start", pose_label="Original start"))
        first = session.start_pose_id
        expected = session.start_pose
        assert session._snapshot()["at_start"]
        assert session.power.state == "off"
        session.robot.teach({**session.pose, "shoulder_pan": 12.0})
        session._command(Command(action="capture_start"))
        assert session.start_pose_id is None
        assert set(session.saved_starts) == {first}
        session._command(Command(action="save_start", pose_label="Alternate start"))
        second = session.start_pose_id
        assert second != first

        session._command(Command(action="select_start", pose_id=first))
        assert session._reset_goal() == expected
        assert not session._snapshot()["at_start"]
        with pytest.raises(WorkflowError, match="Return to the selected start"):
            session._command(Command(action="teach"))
        session.heartbeat()
        session._command(Command(action="auto_reset"))
        deadline = monotonic() + 6
        while session.phase == Phase.RETURNING and monotonic() < deadline:
            session.heartbeat()
            session._tick(monotonic())
            sleep(1 / CONTROL.fps)
        assert session.phase == Phase.READY
        assert session._snapshot()["at_start"]
        session._command(Command(action="teach"))
        assert session.phase == Phase.TEACHING
        assert session.teaching_origin == expected
        session._command(Command(action="stop_teach"))
        session._command(Command(action="discard_teaching"))

        session._command(Command(action="disconnect"))
        session._command(Command(action="connect"))
        assert session._snapshot()["start_pose"] is None
        assert {p["id"]: p["label"] for p in session._snapshot()["start_poses"]} == {
            first: "Original start", second: "Alternate start",
        }
        session._command(Command(action="select_start", pose_id=second))
        assert session.start_pose["shoulder_pan"] == 12.0
        with pytest.raises(WorkflowError, match="Confirm"):
            session._command(Command(action="delete_start", pose_id=second))
        session._command(Command(action="delete_start", pose_id=second, confirmed=True))
        assert session.start_pose_id is None
        assert session.start_pose["shoulder_pan"] == 12.0
        assert not (session._start_library() / f"{second}.json").exists()
    finally:
        session._close_devices()


def test_previous_positions_remain_available_without_mixing_rest_into_start(tmp_path):
    session = Session(tmp_path)
    session.config = Configuration(cameras=[])
    session._command(Command(action="connect"))
    try:
        old_start = {**session.pose, "shoulder_pan": 12.0}
        old_rest = {**session.pose, "shoulder_pan": -8.0}
        write_json(session._pose_file(), {"version": 1, "position": old_start})
        write_json(session._pose_file(rest=True), {"version": 1, "position": old_rest})
        write_json(session._start_library() / "abcdef012345.json", {
            "version": 1, "position": old_start, "created": "2026-11-23T00:00:00+00:00",
        })
        session._command(Command(action="disconnect"))
        session._command(Command(action="connect"))
        assert session.start_pose is None
        assert session._park_goal() == old_rest
        assert set(session.saved_starts) == {"previous-start", "abcdef012345"}
        assert all(saved.legacy for saved in session.saved_starts.values())
        session._command(Command(action="select_start", pose_id="previous-start"))
        assert session._reset_goal() == old_start
        with pytest.raises(WorkflowError, match="Confirm changing"):
            session._command(Command(action="capture_rest"))
        assert session._park_goal() == old_rest
        session._command(Command(action="capture_rest", confirmed=True))
        assert session._park_goal() == session.robot.read()
        assert session._reset_goal() == old_start
        session._command(Command(action="clear_rest", confirmed=True))
        assert not session._pose_file(rest=True).exists()
        assert session._park_goal() is None
        assert session._reset_goal() == old_start
    finally:
        session._close_devices()
