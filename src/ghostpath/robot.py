"""LeRobot 0.7.0 adapter with explicit, torque-preserving lifecycle."""

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .motion import JOINTS, Motion, ParkTrajectory, Pose, SafetyFault, Settings
from .poses import calibration_data, fingerprint


class Robot(Protocol):
    robot_id: str
    calibration_hash: str
    limits: dict[str, tuple[float, float]]

    def connect(self) -> None: ...
    def read(self) -> Pose: ...
    def torque(self) -> str: ...
    def release(self) -> None: ...
    def arm(self, check: Callable[[], None] | None = None) -> Pose: ...
    def motion_start(self) -> tuple[Pose, Pose]: ...
    def verify_goals(self) -> None: ...
    def send(self, pose: Pose, *, force: bool = False) -> Pose: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...


class Follower:
    def __init__(
        self,
        port: str,
        robot_id: str,
        settings: Settings,
        calibration_dir: Path | None = None,
    ):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=robot_id,
                calibration_dir=calibration_dir,
                use_degrees=True,
                disable_torque_on_disconnect=False,
            )
        )
        self.settings = settings
        self.robot_id = robot_id
        self._configured = False
        self._native_profile = False
        if set(self.robot.calibration) != set(JOINTS):
            raise ValueError(
                f"Missing calibration at {self.robot.calibration_fpath}; "
                "calibrate with LeRobot first"
            )
        self.calibration_hash = fingerprint(calibration_data(self.robot))
        self.limits = {}
        for j, cal in self.robot.calibration.items():
            if (
                not 0 <= cal.range_min < cal.range_max <= 4095
                or cal.id != self.robot.bus.motors[j].id
            ):
                raise ValueError(f"Invalid calibration for {j}")
            if cal.drive_mode not in (0, 1):
                raise ValueError(f"Unsupported drive mode for {j}")
            span = (cal.range_max - cal.range_min) * 360 / 4095
            self.limits[j] = (0.0, 100.0) if j == "gripper" else (-span / 2, span / 2)
            if self.limits[j][0] >= self.limits[j][1]:
                raise ValueError(f"Calibration range for {j} is too small")

    def connect(self):
        # No Robot.connect/configure: those toggle torque and use maximum acceleration.
        self.robot.bus.default_timeout = 50  # milliseconds per transport attempt
        try:
            self.robot.bus.connect()
            if not self.robot.bus.is_calibrated:
                raise ValueError(
                    "Motor calibration differs from the saved calibration; recalibrate first"
                )
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self) -> Pose:
        bus = self.robot.bus
        # Check raw feedback without silently clipping it. Allow 2 degrees (or 2%
        # gripper travel) of reading overshoot; command limits remain unchanged.
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=1)
        for j in JOINTS:
            cal = self.robot.calibration[j]
            tolerance = (
                (cal.range_max - cal.range_min) / 100 if j == "gripper" else 4095 / 360
            ) * 2
            if not cal.range_min - tolerance <= raw[j] <= cal.range_max + tolerance:
                raise SafetyFault(f"{j} encoder is outside its calibration range")
        pose = self._normalize(raw)
        Motion(pose, self.limits, self.settings).check_measurement(pose)
        return pose

    def _normalize(self, raw: dict) -> Pose:
        pose = {}
        for j in JOINTS:
            cal = self.robot.calibration[j]
            if j == "gripper":
                pose[j] = (raw[j] - cal.range_min) * 100 / (cal.range_max - cal.range_min)
                if cal.drive_mode:
                    pose[j] = 100 - pose[j]
            else:
                pose[j] = (raw[j] - (cal.range_min + cal.range_max) / 2) * 360 / 4095
        return pose

    def _raw_goal(self, pose: Pose) -> dict[str, int]:
        goals = {}
        for j in JOINTS:
            cal = self.robot.calibration[j]
            if j == "gripper":
                value = 100 - pose[j] if cal.drive_mode else pose[j]
                raw = value * (cal.range_max - cal.range_min) / 100 + cal.range_min
            else:
                raw = pose[j] * 4095 / 360 + (cal.range_min + cal.range_max) / 2
            goals[j] = max(cal.range_min, min(cal.range_max, round(raw)))
        return goals

    def torque(self) -> str:
        states = set(self.robot.bus.sync_read("Torque_Enable", normalize=False).values())
        return "on" if states == {1} else "off" if states == {0} else "mixed"

    def release(self):
        self.robot.bus.disable_torque(num_retry=1)
        if self.torque() != "off":
            raise SafetyFault("Could not verify that every motor released torque")

    def _configure_limits(self):
        if self._configured:
            return
        bus, s = self.robot.bus, self.settings
        torque = bus.sync_read("Torque_Enable", normalize=False)
        modes = bus.sync_read("Operating_Mode", normalize=False)
        phases = bus.sync_read("Phase", normalize=False)
        if any(torque[j] and modes[j] != 0 for j in JOINTS):
            raise SafetyFault(
                "An energized servo is not in position mode; configure it with LeRobot first"
            )
        if any(int(phases[j]) & 0x10 for j in JOINTS):
            raise SafetyFault("Servo Phase bit 4 is set; configure the follower with LeRobot first")
        for j in JOINTS:
            if not torque[j] and modes[j] != 0:
                bus.set_operating_mode("position", j)
            cal = self.robot.calibration[j]
            steps_per_unit = (cal.range_max - cal.range_min) / 100 if j == "gripper" else 4095 / 360
            # STS3215 Goal_Speed is encoder steps/s. Zero means unrestricted: never write zero.
            speed = max(1, math.floor(s.velocity(j) * steps_per_unit))
            accel = max(1, min(254, math.floor(s.accel(j) * steps_per_unit / 100)))
            bus.write("Goal_Speed", j, speed, normalize=False)
            bus.write("Acceleration", j, accel, normalize=False)
            bus.write("Goal_Time", j, 0, normalize=False)
            if j == "gripper":
                existing = bus.read("Torque_Limit", j, normalize=False)
                bus.write("Torque_Limit", j, min(500, existing), normalize=False)
            for register, expected in (
                ("Goal_Speed", speed),
                ("Acceleration", accel),
                ("Goal_Time", 0),
            ):
                if int(bus.read(register, j, normalize=False)) != expected:
                    raise SafetyFault(f"Could not verify {register} limit for {j}")
        if set(bus.sync_read("Operating_Mode", normalize=False).values()) != {0}:
            raise SafetyFault("Could not verify position mode on every motor")
        self._configured = True

    def arm(self, check: Callable[[], None] | None = None) -> Pose:
        bus = self.robot.bus
        self._configure_limits()
        torque = bus.sync_read("Torque_Enable", normalize=False)
        measured = self.read()
        pose = Motion(measured, self.limits, self.settings).target
        seeded = self._send(pose, force=True)
        goals = bus.sync_read("Goal_Position", normalize=False)
        if goals != self.last_raw_goal:
            raise SafetyFault("Seed goal readback failed; refusing to enable torque")
        fresh = self.read()
        if any(max(abs(fresh[j] - measured[j]), abs(fresh[j] - seeded[j])) > 0.6 for j in JOINTS):
            raise SafetyFault("Arm moved while preparing torque. Support it and recover.")
        off = [j for j in JOINTS if not torque[j]]
        if check:
            check()
        if off:
            bus.enable_torque(off, num_retry=1)
        if self.torque() != "on":
            raise SafetyFault("Could not verify torque on every motor")
        return fresh

    def motion_start(self) -> tuple[Pose, Pose]:
        """Continue from the goal the energized servos are already holding."""
        self._configure_limits()
        if self.torque() != "on":
            raise SafetyFault("All motors must be holding before starting automatic motion")
        measured = self.read()
        goals = self.robot.bus.sync_read("Goal_Position", normalize=False, num_retry=1)
        target = self._normalize(goals)
        motion = Motion(measured, self.limits, self.settings)
        motion.check_pose(target)
        motion.target = target
        motion.check_tracking(measured)
        self.last_raw_goal = goals
        return measured, target

    def _steps_per_unit(self, joint: str) -> float:
        cal = self.robot.calibration[joint]
        return (cal.range_max - cal.range_min) / 100 if joint == "gripper" else 4095 / 360

    def _set_profile(self, speeds: dict[str, float], accelerations: dict[str, float]):
        bus = self.robot.bus
        for register, values, divisor in (
            ("Goal_Speed", speeds, 1),
            ("Acceleration", accelerations, 100),
        ):
            raw = {
                j: max(1, math.floor(v * self._steps_per_unit(j) / divisor))
                for j, v in values.items()
            }
            bus.sync_write(register, raw, normalize=False)
            if bus.sync_read(register, list(raw), normalize=False, num_retry=1) != raw:
                raise SafetyFault(f"Could not verify the servo {register} profile")

    def native_profile(self, path: ParkTrajectory):
        speeds, accelerations = {}, {}
        duration = path.duration
        for j in JOINTS:
            distance = abs(path.end[j] - path.start[j])
            if distance < 0.08 and j in ("shoulder_lift", "elbow_flex"):
                speeds[j], accelerations[j] = 5.0, 25.0
            elif distance >= 0.08:
                speeds[j] = min(self.settings.velocity(j, parking=True), 2 * distance / duration)
                accelerations[j] = min(
                    self.settings.accel(j, parking=True),
                    4 * distance / duration**2,
                )
        if speeds:
            self._native_profile = True
            self._set_profile(speeds, accelerations)

    def stream_profile(self):
        if self._native_profile:
            self._set_profile(
                {j: self.settings.velocity(j) for j in JOINTS},
                {j: self.settings.accel(j) for j in JOINTS},
            )
            self._native_profile = False

    def verify_goals(self):
        goals = self.robot.bus.sync_read("Goal_Position", normalize=False, num_retry=1)
        if goals != self.last_raw_goal:
            joint = next(j for j in JOINTS if goals.get(j) != self.last_raw_goal.get(j))
            raise SafetyFault(f"{joint.replace('_', ' ').capitalize()} goal register changed")

    def send(self, pose: Pose, *, force: bool = False) -> Pose:
        if not self._configured:
            raise SafetyFault("Motor limits have not been configured")
        return self._send(pose, force=force)

    def _send(self, pose: Pose, *, force: bool = False) -> Pose:
        Motion(pose, self.limits, self.settings).check_pose(pose)
        raw_goal = self._raw_goal(pose)
        previous = getattr(self, "last_raw_goal", {})
        values = (
            raw_goal if force else {j: raw for j, raw in raw_goal.items() if previous.get(j) != raw}
        )
        if values:
            self.robot.bus.sync_write("Goal_Position", values, normalize=False)
            self.last_raw_goal = raw_goal
        return self._normalize(raw_goal)

    def hold(self):
        # A fault never energizes disabled motors. Seed only the motors known to be on.
        bus = self.robot.bus
        torque = bus.sync_read("Torque_Enable", normalize=False)
        if not any(torque.values()):
            return
        measured = Motion(self.read(), self.limits, self.settings).target
        goals = {j: tick for j, tick in self._raw_goal(measured).items() if torque[j]}
        bus.sync_write("Goal_Position", goals, normalize=False)
        if bus.sync_read("Goal_Position", list(goals), normalize=False) != goals:
            raise SafetyFault("Could not verify the hold on energized motors")
        self.last_raw_goal = goals

    def close(self):
        if self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)


class SimulatedFollower:
    """Lagged joint feedback for practicing the workflow without hardware."""

    robot_id = "simulation"
    calibration_hash = "simulation"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.limits = {j: ((0.0, 100.0) if j == "gripper" else (-100.0, 100.0)) for j in JOINTS}
        self.position = {j: (65.0 if j == "gripper" else 0.0) for j in JOINTS}
        self.target = dict(self.position)
        self.last = time.monotonic()
        self.armed = False

    def connect(self):
        self.last = time.monotonic()

    def read(self) -> Pose:
        now = time.monotonic()
        dt, self.last = now - self.last, now
        if self.armed:
            for j in JOINTS:
                delta = self.target[j] - self.position[j]
                cap = min(self.settings.velocity(j) * dt, abs(delta) * min(1, 24 * dt))
                self.position[j] += math.copysign(cap, delta)
        return dict(self.position)

    def torque(self) -> str:
        return "on" if self.armed else "off"

    def release(self):
        self.read()
        self.armed = False

    def arm(self, check: Callable[[], None] | None = None) -> Pose:
        if check:
            check()
        pose = self.read()
        self.target = dict(pose)
        self.armed = True
        return pose

    def motion_start(self) -> tuple[Pose, Pose]:
        if not self.armed:
            raise SafetyFault("Torque is off")
        return self.read(), self.target.copy()

    def verify_goals(self):
        pass

    def send(self, pose: Pose, *, force: bool = False) -> Pose:
        if not self.armed:
            raise SafetyFault("Torque is off")
        Motion(pose, self.limits, self.settings).check_pose(pose)
        self.target = dict(pose)
        return dict(pose)

    def teach(self, pose: Pose):
        if self.armed:
            raise SafetyFault("Manual simulation requires torque off")
        Motion(pose, self.limits, self.settings).check_pose(pose)
        self.position = dict(pose)

    def hold(self):
        self.target = self.read()

    def close(self):
        pass
