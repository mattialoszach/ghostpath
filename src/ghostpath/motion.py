"""Hardware-independent bounded motion. Angles are degrees, gripper is percent."""

import math
from dataclasses import dataclass

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
Pose = dict[str, float]


class SafetyFault(RuntimeError):
    """Movement must stop; automatic parking is inappropriate after a fault."""


def validate_pose(pose: Pose) -> Pose:
    if set(pose) != set(JOINTS):
        raise ValueError(f"Expected exactly these joints: {', '.join(JOINTS)}")
    if any(isinstance(v, bool) or not math.isfinite(v) for v in pose.values()):
        raise ValueError("All joint positions must be finite numbers")
    return {j: float(pose[j]) for j in JOINTS}


@dataclass(frozen=True)
class Settings:
    fps: int = 60
    speed: float = 90.0
    acceleration: float = 270.0
    base_speed_scale: float = 1.25
    gripper_speed: float = 90.0
    gripper_acceleration: float = 300.0
    response_time: float = 0.12
    park_speed: float = 30.0
    park_acceleration: float = 90.0
    max_lead: float = 15.0
    gripper_max_lead: float = 20.0
    watchdog: float = 0.25
    park_timeout: float = 90.0
    settle_seconds: float = 0.7
    tolerance: float = 1.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.fps, int) or not 10 <= self.fps <= 60:
            raise ValueError("fps must be an integer between 10 and 60")
        if self.speed > 120 or self.park_speed > self.speed or self.acceleration > 360:
            raise ValueError("Require park_speed <= speed <= 120 and acceleration <= 360")
        if self.park_speed > 60:
            raise ValueError("park_speed must not exceed 60")
        if not 0.08 <= self.response_time <= 0.5:
            raise ValueError("response_time must be between 0.08 and 0.5 seconds")
        if self.park_acceleration > self.acceleration:
            raise ValueError("park_acceleration must not exceed acceleration")
        if self.gripper_speed > 120 or self.gripper_acceleration > 480:
            raise ValueError("Gripper speed <= 120 and acceleration <= 480 required")
        if self.max_lead > 15 or self.gripper_max_lead > 25:
            raise ValueError("max_lead <= 15 and gripper_max_lead <= 25 required")
        if not 2 / self.fps < self.watchdog <= 0.5:
            raise ValueError("watchdog must exceed two frames and be at most 0.5 seconds")

    def velocity(self, joint: str, parking: bool = False) -> float:
        if joint == "gripper":
            return min(self.gripper_speed, self.park_speed) if parking else self.gripper_speed
        if joint == "shoulder_pan":
            speed = min(120.0, self.speed * self.base_speed_scale)
            return min(speed, self.park_speed) if parking else speed
        return self.park_speed if parking else self.speed

    def accel(self, joint: str, parking: bool = False) -> float:
        if joint == "gripper":
            return (
                min(self.gripper_acceleration, self.park_acceleration)
                if parking
                else self.gripper_acceleration
            )
        if joint == "shoulder_pan":
            acceleration = min(360.0, self.acceleration * self.base_speed_scale)
            return min(acceleration, self.park_acceleration) if parking else acceleration
        return self.park_acceleration if parking else self.acceleration

    def lead(self, joint: str) -> float:
        return self.gripper_max_lead if joint == "gripper" else self.max_lead

    @property
    def recording_fps(self) -> int:
        return min(self.fps, 30)


class Motion:
    def __init__(self, initial: Pose, limits: dict[str, tuple[float, float]], settings: Settings):
        self.settings = settings
        self.limits = limits
        self.target = validate_pose(initial)
        self.velocity = {j: 0.0 for j in JOINTS}
        self.acceleration = {j: 0.0 for j in JOINTS}
        self.check_measurement(self.target)
        self.target = self.clamp(self.target)

    def clamp(self, pose: Pose) -> Pose:
        return {j: max(self.limits[j][0], min(self.limits[j][1], pose[j])) for j in JOINTS}

    def check_measurement(self, measured: Pose):
        validate_pose(measured)
        for j in JOINTS:
            lo, hi = self.limits[j]
            # A little encoder overshoot is normal at an endpoint. Commands still stay
            # inside the calibrated range; only feedback gets this tolerance.
            if not lo - 2 <= measured[j] <= hi + 2:
                raise SafetyFault(f"{j}: feedback {measured[j]:.2f} outside calibrated range")

    def check_pose(self, pose: Pose):
        validate_pose(pose)
        for j in JOINTS:
            lo, hi = self.limits[j]
            if not lo <= pose[j] <= hi:
                raise SafetyFault(f"{j}: {pose[j]:.2f} outside safe range [{lo:.2f}, {hi:.2f}]")

    def check_tracking(self, measured: Pose):
        self.check_measurement(measured)
        for j in JOINTS:
            if abs(self.target[j] - measured[j]) > self.settings.lead(j):
                raise SafetyFault(f"{j} is not tracking its target; check for an obstruction")

    def hold(self, measured: Pose) -> Pose:
        """Cancel pending manual motion and latch a measured-position hold once."""
        self.check_measurement(measured)
        self.target = self.clamp(measured)
        self.velocity = {j: 0.0 for j in JOINTS}
        self.acceleration = {j: 0.0 for j in JOINTS}
        return dict(self.target)

    def _smooth_velocity(self, joint: str, desired: float, dt: float) -> tuple[float, float]:
        """Integrate a critically damped velocity response with bounded acceleration."""
        current, acceleration = self.velocity[joint], self.acceleration[joint]
        limit = self.settings.accel(joint)
        response = self.settings.response_time
        omega = 2 / response
        change = max(-limit * response, min(limit * response, current - desired))
        target = current - change
        decay = math.exp(-omega * dt)
        transient = (acceleration + omega * change) * dt
        velocity = target + (change + transient) * decay
        acceleration = (acceleration - omega * transient) * decay
        speed = self.settings.velocity(joint)
        velocity = max(-speed, min(speed, velocity))
        if desired == 0 and abs(velocity) < 0.02 and abs(acceleration) < 0.2:
            return 0.0, 0.0  # Below encoder resolution; finish braking without an endless tail.
        return velocity, acceleration

    def step(self, axes: Pose, dt: float, measured: Pose, precision: bool = False) -> Pose:
        if not math.isfinite(dt) or dt <= 0 or dt > self.settings.watchdog:
            raise SafetyFault("Control loop missed its deadline; release the controls and restart")
        self.check_measurement(measured)
        # Account for ordinary scheduling jitter without replaying a long stall.
        dt = min(dt, 2 / self.settings.fps)
        for j in JOINTS:
            axis = axes.get(j, 0.0)
            if not math.isfinite(axis) or abs(axis) > 1:
                raise SafetyFault("Invalid keyboard input")
            v, a = self.velocity[j], self.settings.accel(j)
            previous = self.target[j]
            desired = axis * self.settings.velocity(j) * (0.25 if precision else 1.0)
            lo, hi = self.limits[j]
            direction = math.copysign(1.0, desired)
            lead = self.settings.lead(j)
            distance = hi - previous if desired > 0 else previous - lo
            headroom = lead - direction * (previous - measured[j])
            distance, headroom = max(0.0, distance), max(0.0, headroom)
            # The tracking window governs following error, not mechanical stopping
            # distance. Applying the braking-distance cap to it silently caps speed.
            # Leave enough endpoint distance for the velocity easing to finish braking.
            desired = direction * min(
                abs(desired),
                headroom / self.settings.response_time,
                distance / (4 * self.settings.response_time),
                math.sqrt(2 * a * distance),
            )
            new_v, new_a = self._smooth_velocity(j, desired, dt)
            candidate = max(lo, min(hi, previous + (v + new_v) * 0.5 * dt))
            # Never pull a forward-moving reference backward to chase noisy feedback.
            # If feedback falls behind, hold the reference until the arm catches up.
            if candidate > previous:
                bounded = min(candidate, max(previous, measured[j] + lead))
            else:
                bounded = max(candidate, min(previous, measured[j] - lead))
            if bounded != candidate:
                candidate, new_v, new_a = bounded, 0.0, 0.0
            if candidate in (lo, hi):
                new_v, new_a = 0.0, 0.0
            self.target[j], self.velocity[j] = candidate, new_v
            self.acceleration[j] = new_a
        return dict(self.target)

    @property
    def stopped(self) -> bool:
        return all(abs(v) < 1e-8 for v in self.velocity.values())


class ParkTrajectory:
    """Synchronized quintic trajectory with zero endpoint velocity and acceleration."""

    def __init__(self, start: Pose, rest: Pose, settings: Settings):
        self.start, self.end = validate_pose(start), validate_pose(rest)
        self.duration = max(
            1.0,
            *(1.875 * abs(rest[j] - start[j]) / settings.velocity(j, True) for j in JOINTS),
            *(math.sqrt(5.774 * abs(rest[j] - start[j]) / settings.accel(j, True)) for j in JOINTS),
        )

    def at(self, elapsed: float) -> Pose:
        if elapsed >= self.duration:
            return self.end.copy()
        t = max(0.0, min(1.0, elapsed / self.duration))
        blend = max(0.0, min(1.0, 10 * t**3 - 15 * t**4 + 6 * t**5))
        return {j: self.start[j] + (self.end[j] - self.start[j]) * blend for j in JOINTS}
