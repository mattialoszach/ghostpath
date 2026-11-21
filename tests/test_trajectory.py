import math
from copy import deepcopy

import pytest

from ghostpath.grip import GRIP_LIMIT, REVERSE_PRELOAD, BoomerangPath, GripAssist
from ghostpath.motion import JOINTS, Motion, ParkTrajectory, SafetyFault
from ghostpath.robot import SimulatedFollower
from ghostpath.session import CONTROL
from ghostpath.trajectory import (
    JOINT_MARGIN,
    AlignedPath,
    NativePlayer,
    Player,
    ReplayPath,
    ReversePath,
    Sample,
    ServoPlayer,
)


def test_smoothing_preserves_gripper_turns_and_bounds_derivatives():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    motion = Motion(start, robot.limits, CONTROL)
    peaks = (65, 20, 55, 25, 65)
    samples = []
    for i in range(81):
        segment, point = min(i // 20, 3), i % 20 if i < 80 else 20
        fraction = (1 - math.cos(math.pi * point / 20)) / 2
        a, b = peaks[segment : segment + 2]
        pose = {**start, "shoulder_pan": i * 0.40, "gripper": a + (b - a) * fraction}
        samples.append(Sample(i / 60, pose))
    raw = deepcopy(samples)
    path = ReplayPath(samples, motion, CONTROL, speed=2)
    assert samples == raw
    assert path.start == samples[0].pose
    assert path.end == samples[-1].pose
    assert set(peaks).issubset({path.at(t)["gripper"] for t in path.times})
    assert path.keyframes < len(samples)

    dt = path.duration / 20_000
    before = path.at(0)
    previous = dict.fromkeys(JOINTS, 0.0)
    for i in range(1, 20_001):
        after = path.at(i * dt)
        for j in JOINTS:
            velocity = (after[j] - before[j]) / dt
            assert abs(velocity) <= CONTROL.velocity(j)
            assert abs(velocity - previous[j]) / dt <= CONTROL.accel(j)
            previous[j] = velocity
        assert 20 <= after["gripper"] <= 65
        before = after

    measured = {**start, "shoulder_pan": 1.0}
    aligned = AlignedPath(measured, path, CONTROL)
    assert aligned.at(0) == measured
    assert aligned.at(aligned.prefix) == start


def test_rejects_discontinuous_raw_measurements():
    robot = SimulatedFollower(CONTROL)
    home = robot.read()
    motion = Motion(home, robot.limits, CONTROL)
    samples = [Sample(i / 60, home) for i in range(31)]
    samples[20] = Sample(20 / 60, {**home, "wrist_roll": 95})
    with pytest.raises(ValueError, match="encoder jumped"):
        ReplayPath(samples, motion, CONTROL, speed=1)
    samples[20] = Sample(19 / 60, home)
    with pytest.raises(ValueError, match="out of order"):
        ReplayPath(samples, motion, CONTROL, speed=1)


def test_reverse_retraces_gripper_and_insets_calibration_endpoints():
    robot = SimulatedFollower(CONTROL)
    start = {**robot.read(), "elbow_flex": 100.0}
    motion = Motion(start, robot.limits, CONTROL)
    samples = [
        Sample(
            i / 60,
            {
                **start,
                "elbow_flex": 100 - i / 8,
                "gripper": 65 - 55 * math.sin(i * math.pi / 90) ** 4,
            },
        )
        for i in range(91)
    ]
    path = ReplayPath(samples, motion, CONTROL, speed=1, joint_margin=JOINT_MARGIN)
    reverse = ReversePath(path)
    assert samples[0].pose["elbow_flex"] == 100
    assert path.start["elbow_flex"] == 100 - JOINT_MARGIN
    assert reverse.start == path.end
    assert reverse.end == path.start
    for i in range(101):
        fraction = i / 100
        assert reverse.at(reverse.duration * fraction) == pytest.approx(
            path.at(path.duration * (1 - fraction))
        )


def test_grip_assist_eases_back_and_replays_edits_at_the_same_arm_poses():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    motion = Motion(start, robot.limits, CONTROL)
    samples = [
        Sample(i / 60, {**start, "shoulder_pan": i / 20, "gripper": 65 - i / 32})
        for i in range(241)
    ]
    base = ReplayPath(samples, motion, CONTROL, speed=1)
    reverse = ReversePath(base, time_scale=1.0)
    assist = GripAssist(base, reverse, prefix=0)
    dt = 1 / CONTROL.fps
    for i in range(1, round(base.duration / dt)):
        t = i * dt
        if 0.5 <= t <= 1.3:
            assist.input(-1, t)
        # Also exercise lost key-up. The heartbeat timeout must release the grip.
        assist.advance(t, dt, t)
        issued = assist.at(t)
        assist.capture(t, issued["gripper"])
        if t >= 2.2:
            assert abs(assist.offset) < 0.15
    assert assist.at(0) == reverse.start
    assert assist.at(assist.duration) == reverse.end
    edited = assist.result(CONTROL)
    assert edited is not None
    middle = 1 - 1.28 / base.duration
    assert edited.offset(middle) < -4
    assert all(abs(bias) <= GRIP_LIMIT for _, bias in edited.knots)
    assert edited.start == base.start and edited.end == base.end
    dt = edited.duration / 4000
    before, previous_velocity = edited.start["gripper"], 0.0
    for i in range(1, 4001):
        position = i / 4000
        target = edited.at(i * dt)
        original = base.at(position * base.duration)
        assert target["shoulder_pan"] == pytest.approx(original["shoulder_pan"])
        velocity = (target["gripper"] - before) / dt
        assert abs(velocity) <= CONTROL.gripper_speed
        assert abs(velocity - previous_velocity) / dt <= CONTROL.gripper_acceleration
        before, previous_velocity = target["gripper"], velocity


def test_boomerang_secures_the_pickup_and_keeps_the_taught_release_and_endpoints():
    robot = SimulatedFollower(CONTROL)
    frames = [
        (0, 1),
        (2, 1),
        (2, 30),
        (8, 30),
        (8, 14),
        (8, 14),
        (32, 14),
        (32, 14),
        (32, 31),
        (15, 31),
        (15, 1),
        (16, 1),
    ]
    start = {**robot.read(), "gripper": 1}
    samples = []
    for i, ((arm, grip), (next_arm, next_grip)) in enumerate(zip(frames, frames[1:], strict=False)):
        for k in range(50):
            f = k / 50
            pose = {
                **start,
                "elbow_flex": arm + f * (next_arm - arm),
                "gripper": grip + f * (next_grip - grip),
            }
            samples.append(Sample(i + f, pose))
    samples.append(Sample(len(frames) - 1, {**start, "elbow_flex": 16}))
    path = ReplayPath(samples, Motion(start, robot.limits, CONTROL), CONTROL, speed=1)
    reverse = BoomerangPath(path)
    assert len(reverse.grasps) == 1
    pick, release = reverse.grasps[0]
    assert reverse.duration > path.duration
    assert reverse.at(0) == path.end
    assert reverse.at(reverse.duration) == path.start
    assert reverse.preload(pick - 0.2) == 0
    assert reverse.preload(pick + 0.65) == pytest.approx(-REVERSE_PRELOAD)
    assert reverse.preload(release + 1) == 0
    assist = GripAssist(path, reverse, 0, reverse)
    dt = reverse.duration / 2000
    previous = reverse.start
    velocity = dict.fromkeys(JOINTS, 0.0)
    prior_fraction = 1.0
    for i in range(1, 2001):
        t = i * dt
        target = reverse.at(t)
        fraction = reverse.forward_fraction(t)
        assert 0 <= fraction <= prior_fraction
        prior_fraction = fraction
        original = path.at(fraction * path.duration)
        assert target["elbow_flex"] == pytest.approx(original["elbow_flex"])
        for j in JOINTS:
            speed = (target[j] - previous[j]) / dt
            assert abs(speed) <= CONTROL.velocity(j)
            assert abs(speed - velocity[j]) / dt <= CONTROL.accel(j)
            velocity[j] = speed
        assist.capture(t, target["gripper"])
        previous = target
    assert assist.result(CONTROL) is None


def test_player_slows_for_lag_and_keeps_commands_monotonic():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    path = ParkTrajectory(start, {**start, "elbow_flex": 32.0}, CONTROL)
    player = Player(path, Motion(start, robot.limits, CONTROL), 0, 2.0)
    observed = start.copy()
    command = start
    rates = []
    dt = 1 / CONTROL.fps
    for i in range(1, 1200):
        # A load delays the actual elbow, but the joint can still move.
        observed["elbow_flex"] += min(
            6 * dt, max(0, command["elbow_flex"] - observed["elbow_flex"] - 1.4)
        )
        earlier = command["elbow_flex"]
        command, done = player.step(i * dt, observed)
        assert command["elbow_flex"] >= earlier
        rates.append(player.rate)
        if done:
            break
    assert done
    assert min(rates) < 0.7
    assert observed["elbow_flex"] == pytest.approx(32, abs=2)


def test_player_aborts_a_stalled_joint_with_measured_goal_context():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    path = ParkTrajectory(start, {**start, "elbow_flex": 32.0}, CONTROL)
    player = Player(path, Motion(start, robot.limits, CONTROL), 0, 2.0)
    with pytest.raises(SafetyFault, match=r"Elbow flex (stalled|tracking).*measured .*goal"):
        for i in range(1, 600):
            player.step(i / CONTROL.fps, start)


def test_native_position_loop_trims_static_load_without_overshoot():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    goal = {**start, "shoulder_lift": -5.0}
    path = ParkTrajectory(start, goal, CONTROL)
    player = NativePlayer(path, Motion(start, robot.limits, CONTROL), 0, 2.0)
    observed = start.copy()
    issued = start
    for i in range(1, 541):
        # A proportional servo under gravity stops short of its register goal.
        equilibrium = min(0.0, issued["shoulder_lift"] + 4.1)
        observed["shoulder_lift"] += 0.11 * (equilibrium - observed["shoulder_lift"])
        issued, done = player.step(i / 60, observed)
        if done:
            break
    assert done
    assert issued["shoulder_lift"] < goal["shoulder_lift"]
    assert issued["shoulder_lift"] >= goal["shoulder_lift"] - 4.5
    assert observed["shoulder_lift"] == pytest.approx(goal["shoulder_lift"], abs=2.0)


def test_servo_player_continues_from_holding_goal_and_keeps_lookahead_ordered():
    robot = SimulatedFollower(CONTROL)
    start = robot.read()
    end = {**start, "shoulder_pan": 28.0}
    path = ParkTrajectory(start, end, CONTROL)
    held = {**start, "shoulder_lift": -3.0}
    player = ServoPlayer(path, Motion(start, robot.limits, CONTROL), 0, 2.0, held)
    observed = start.copy()
    previous = held
    assert player.step(1 / 60, observed)[0] == held
    for i in range(2, 600):
        observed["shoulder_pan"] += 0.2 * (previous["shoulder_pan"] - observed["shoulder_pan"])
        issued, done = player.step(i / 60, observed)
        assert issued["shoulder_pan"] >= previous["shoulder_pan"]
        assert issued["shoulder_pan"] <= end["shoulder_pan"]
        assert abs(issued["shoulder_pan"] - observed["shoulder_pan"]) < CONTROL.max_lead
        previous = issued
        if done:
            break
    assert done
