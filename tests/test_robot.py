from types import SimpleNamespace

import pytest

from ghostpath.motion import JOINTS, SafetyFault
from ghostpath.robot import Follower
from ghostpath.session import CONTROL


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
