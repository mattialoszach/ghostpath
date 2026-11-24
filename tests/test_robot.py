from time import monotonic
from types import SimpleNamespace

import pytest

from ghostpath.models import Command, JogSpeed
from ghostpath.motion import JOINTS, SafetyFault
from ghostpath.robot import Follower
from ghostpath.session import CONTROL, FAST_REPLAY, Session, WorkflowError


class Bus:
    is_connected = True

    def __init__(self):
        self.calls = []
        self.reg = {
            register: dict.fromkeys(JOINTS, value)
            for register, value in (
                ("Torque_Enable", 0),
                ("Operating_Mode", 0),
                ("Phase", 0),
                ("Present_Position", 2000),
                ("Goal_Speed", 0),
                ("Goal_Position", 0),
                ("Torque_Limit", 300),
                ("Acceleration", 0),
            )
        }

    def sync_read(self, register, names=None, **kwargs):
        return {j: self.reg[register][j] for j in names or JOINTS}

    def read(self, register, joint, **kwargs):
        return self.reg[register][joint]

    def write(self, register, joint, value, **kwargs):
        self.reg.setdefault(register, {})[joint] = value

    def sync_write(self, register, values, **kwargs):
        self.calls.append((register, values.copy()))
        self.reg[register].update(values)

    def enable_torque(self, names, **kwargs):
        self.calls.append(("enable", names))
        self.reg["Torque_Enable"].update(dict.fromkeys(names, 1))

    def disable_torque(self, **kwargs):
        self.calls.append(("disable",))
        self.reg["Torque_Enable"] = dict.fromkeys(JOINTS, 0)


@pytest.fixture
def rig():
    arm = Follower.__new__(Follower)
    arm._configured = False
    arm.settings = CONTROL
    arm.limits = {j: (0, 100) if j == "gripper" else (-100, 100) for j in JOINTS}
    calibration = {
        j: SimpleNamespace(range_min=500, range_max=3500, drive_mode=int(j == "gripper"))
        for j in JOINTS
    }
    arm.robot = SimpleNamespace(bus=Bus(), calibration=calibration)
    return arm, arm.robot.bus


def test_enable_follows_goal_seed_and_release_is_verified(rig):
    arm, bus = rig
    assert arm.arm()["gripper"] == 50
    assert bus.calls[0] == ("Goal_Position", dict.fromkeys(JOINTS, 2000))
    assert bus.calls[1] == ("enable", list(JOINTS))
    assert all(bus.reg["Goal_Speed"].values())
    assert all(bus.reg["Acceleration"].values())
    assert bus.reg["Torque_Limit"]["gripper"] == 300
    arm.release()
    assert arm.torque() == "off"


def test_failed_goals_and_operator_stop_prevent_torque_enable(rig, monkeypatch):
    arm, bus = rig
    write = bus.sync_write
    monkeypatch.setattr(bus, "sync_write", lambda *args, **kwargs: None)
    with pytest.raises(SafetyFault, match="Seed goal readback"):
        arm.arm()
    assert arm.torque() == "off"
    monkeypatch.setattr(bus, "sync_write", write)

    def stop():
        raise SafetyFault("operator stop")

    with pytest.raises(SafetyFault, match="operator stop"):
        arm.arm(stop)
    arm.hold()
    assert arm.torque() == "off"
    assert ("enable", list(JOINTS)) not in bus.calls


def test_stop_with_mixed_torque_never_energizes_unpowered_joints(rig):
    arm, bus = rig
    bus.reg["Torque_Enable"]["elbow_flex"] = 1
    arm.hold()
    assert arm.torque() == "mixed"
    assert bus.calls == [("Goal_Position", {"elbow_flex": 2000})]


def test_automatic_move_preserves_existing_hold_and_skips_repeated_packets(rig):
    arm, bus = rig
    arm.arm()
    bus.calls.clear()
    observed, latched = arm.motion_start()
    assert bus.calls == []
    assert observed == latched
    arm.send(latched)
    assert bus.calls == []
    requested = {**latched, "wrist_flex": latched["wrist_flex"] + 1}
    sent = arm.send(requested)
    arm.send(requested)
    arm.verify_goals()
    assert len(bus.calls) == 1
    assert list(bus.calls[0][1]) == ["wrist_flex"]
    assert sent["wrist_flex"] == pytest.approx(requested["wrist_flex"], abs=0.05)
    bus.reg["Goal_Position"]["wrist_flex"] = 2000
    with pytest.raises(SafetyFault, match="Wrist flex goal register changed"):
        arm.verify_goals()


def test_fast_profile_is_verified_and_restores_standard_registers(rig, monkeypatch):
    arm, bus = rig
    arm.arm()
    standard = {register: bus.reg[register].copy() for register in ("Goal_Speed", "Acceleration")}
    bus.calls.clear()
    arm.stream_profile(FAST_REPLAY)
    assert [register for register, _ in bus.calls] == ["Goal_Speed", "Acceleration"]
    for register, previous in standard.items():
        assert all(bus.reg[register][j] > previous[j] > 0 for j in JOINTS)
    assert arm.torque() == "on"

    arm.stream_profile(FAST_REPLAY)
    assert len(bus.calls) == 2
    arm.stream_profile(CONTROL)
    assert all(bus.reg[register] == values for register, values in standard.items())

    write = bus.sync_write
    monkeypatch.setattr(bus, "sync_write", lambda *args, **kwargs: None)
    with pytest.raises(SafetyFault, match="verify the servo Goal_Speed profile"):
        arm.stream_profile(FAST_REPLAY)
    monkeypatch.setattr(bus, "sync_write", write)
    arm.stream_profile(FAST_REPLAY)
    assert all(bus.reg["Goal_Speed"][j] > standard["Goal_Speed"][j] for j in JOINTS)


def test_manual_jog_speed_switch_checks_feedback_without_changing_goals(rig, tmp_path):
    arm, bus = rig
    arm.arm()
    observed, latched = arm.motion_start()
    session = Session(tmp_path)
    session.robot, session.pose = arm, observed
    session._start_manual(monotonic(), latched, practice=True)
    goals = bus.reg["Goal_Position"].copy()
    accelerations = bus.reg["Acceleration"].copy()
    bus.calls.clear()

    change_speed = Command(action="set_jog_speed", jog_speed=JogSpeed.FAST)
    bus.reg["Present_Position"]["elbow_flex"] -= 42
    with pytest.raises(WorkflowError, match="reach its hold position"):
        session._command(change_speed)
    assert not bus.calls
    assert session.jog_speed == JogSpeed.PRECISION

    bus.reg["Present_Position"].update(goals)
    session.pose = arm.read()
    session._command(change_speed)
    assert session.motion.target == latched
    assert bus.reg["Goal_Position"] == goals and arm.torque() == "on"
    assert bus.reg["Acceleration"] == accelerations
    assert [register for register, _ in bus.calls] == ["Goal_Speed", "Acceleration"]
    assert bus.reg["Goal_Speed"] == {
        joint: int((36 if joint == "gripper" else 37.5) * arm._steps_per_unit(joint))
        for joint in JOINTS
    }


def test_manual_jog_keeps_idle_goals_fixed_under_varying_load(rig, tmp_path):
    arm, bus = rig
    arm.arm()
    driven, loaded = "shoulder_pan", "elbow_flex"
    held = arm.last_raw_goal.copy()
    bus.reg["Present_Position"][loaded] = held[loaded] - 22
    observed, latched = arm.motion_start()
    session = Session(tmp_path)
    session.robot, session.pose = arm, observed
    session._start_manual(monotonic(), latched, practice=True)
    session._command(Command(action="jog", axes={driven: 1}))
    started, period = session.jog_at, 1 / CONTROL.fps
    bus.calls.clear()

    for tick in range(1, CONTROL.fps + 1):
        now = started + tick * period
        if tick % 5 == 0:
            session.jog_at = now
        bus.reg["Present_Position"][driven] = arm.last_raw_goal[driven]
        bus.reg["Present_Position"][loaded] = held[loaded] - 42 + (-1) ** tick * 8
        session.pose = arm.read()
        session.heartbeat()
        session._manual_tick(now, period, now, (now, now))

    goal_writes = [goals for register, goals in bus.calls if register == "Goal_Position"]
    assert goal_writes and all(set(goals) == {driven} for goals in goal_writes)
    driven_goals = [goals[driven] for goals in goal_writes]
    assert driven_goals == sorted(set(driven_goals))
    assert bus.reg["Goal_Position"][loaded] == held[loaded]

    for joint in JOINTS:
        nominal_rate = session.motion.settings.velocity(joint)
        assert bus.reg["Goal_Speed"][joint] == int(nominal_rate * arm._steps_per_unit(joint))
    assert session.motion.velocity[driven] == pytest.approx(12.5, abs=0.05)

    session._command(Command(action="jog", immediate_stop=True))
    now = started + (CONTROL.fps + 1) * period
    bus.calls.clear()
    session._manual_tick(now, period, now, (now, now))
    assert bus.calls == [("Goal_Position", arm._raw_goal(session.pose))]
    session._manual_tick(now + period, period, now + period, (now, now + period))
    assert len(bus.calls) == 1
