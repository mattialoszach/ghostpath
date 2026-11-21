"""Bounded quintic paths and feedback-paced playback."""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Protocol

from .motion import JOINTS, Motion, ParkTrajectory, Pose, SafetyFault, Settings

MAX_TEACH_SECONDS = 180
MAX_REPLAY_SECONDS = 600
JOINT_MARGIN = 1.2
START_TOLERANCE = 2.0
GRIPPER_TOLERANCE = 2.5
LOAD_AXES = ("shoulder_lift", "elbow_flex")
TRIM_LIMIT = 4.5


class Path(Protocol):
    start: Pose
    end: Pose
    duration: float

    def at(self, elapsed: float) -> Pose: ...


def automatic_pose(pose: Pose, motion: Motion, guard: float) -> Pose:
    """Leave clearance inside manually calibrated arm endpoints."""
    bounded = motion.clamp(pose)
    for joint in JOINTS[:-1]:
        lo, hi = motion.limits[joint]
        inset = min(guard, (hi - lo) / 4)
        bounded[joint] = max(lo + inset, min(hi - inset, bounded[joint]))
    return bounded


def tolerance_for(joint: str, tolerance: float = START_TOLERANCE) -> float:
    return GRIPPER_TOLERANCE if joint == "gripper" else tolerance


@dataclass(frozen=True)
class Sample:
    t: float
    pose: Pose


class Teaching:
    def __init__(self):
        self.samples: list[Sample] = []
        self.segment_started = 0.0
        self.segment_offset = 0.0

    def resume(self, timestamp: float, pose: Pose, transition: float):
        self.segment_started = timestamp
        self.segment_offset = self.samples[-1].t + transition if self.samples else 0.0
        self.append(timestamp, pose)

    def append(self, timestamp: float, pose: Pose):
        timestamp = self.segment_offset + (timestamp - self.segment_started)
        if not self.samples or timestamp > self.samples[-1].t:
            self.samples.append(Sample(timestamp, pose.copy()))


def _anchors(samples: list[Sample]) -> list[int]:
    """Simplify against joint error and retain significant gripper turning points."""
    gripper = [s.pose["gripper"] for s in samples]
    required = {0, len(samples) - 1, gripper.index(min(gripper)), gripper.index(max(gripper))}
    pivot, direction = 0, 0
    for i, position in enumerate(gripper[1:], 1):
        change = position - gripper[pivot]
        if not direction or direction * change >= 0:
            if abs(change) >= 0.9:
                direction = 1 if change > 0 else -1
            if direction:
                pivot = i
        elif abs(change) >= 0.9:
            required.add(pivot)
            direction, pivot = -direction, i

    mandatory = sorted(required)
    pending = list(zip(mandatory, mandatory[1:], strict=False))
    while pending:
        start, end = pending.pop()
        a, b = samples[start], samples[end]
        peak, index = 1.0, start
        for i in range(start + 1, end):
            fraction = (samples[i].t - a.t) / (b.t - a.t)
            error = max(
                abs(samples[i].pose[j] - (a.pose[j] + fraction * (b.pose[j] - a.pose[j])))
                / (0.65 if j == "gripper" else 0.9)
                for j in JOINTS
            )
            if error > peak:
                peak, index = error, i
        if index != start:
            required.add(index)
            pending.extend(((start, index), (index, end)))
    return sorted(required)


def _tangents(values: list[float], spans: list[float]) -> list[float]:
    slopes = [(b - a) / h for a, b, h in zip(values[:-1], values[1:], spans, strict=True)]
    result = [0.0]
    for before, after in zip(slopes, slopes[1:], strict=False):
        if before * after <= 0:
            result.append(0.0)
        else:
            # A harmonic mean gives continuous velocity. Cap it so the Bézier
            # control points stay ordered and cannot overshoot either knot.
            mean = 2 * before * after / (before + after)
            limit = 1.20 * min(abs(before), abs(after))
            result.append(max(-limit, min(limit, mean)))
    return [*result, 0.0]


def _bernstein_peak(points: tuple[float, ...], depth: int = 2) -> float:
    """Bound a polynomial using subdivided Bézier control polygons."""
    if not depth:
        return max(map(abs, points))
    left, right = [points[0]], [points[-1]]
    while len(points) > 1:
        points = tuple((a + b) / 2 for a, b in zip(points, points[1:], strict=False))
        left.append(points[0])
        right.append(points[-1])
    return max(_bernstein_peak(tuple(left), depth - 1), _bernstein_peak(tuple(right), depth - 1))


class ReplayPath:
    def __init__(
        self,
        samples: list[Sample],
        motion: Motion,
        settings: Settings,
        speed: float,
        *,
        joint_margin: float = 0,
        start_pose: Pose | None = None,
    ):
        if len(samples) < 3 or samples[-1].t - samples[0].t < 0.25:
            raise ValueError("Teach for at least a quarter of a second.")
        for i, sample in enumerate(samples):
            motion.check_measurement(sample.pose)
            if not isfinite(sample.t):
                raise ValueError("Invalid teaching timestamp.")
            if i:
                before = samples[i - 1]
                dt = sample.t - before.t
                if not 0 < dt <= settings.watchdog:
                    raise ValueError("A teaching sample is missing or out of order. Teach again.")
                if any(abs(sample.pose[j] - before.pose[j]) > 500 * dt + 1.5 for j in JOINTS):
                    raise ValueError("An encoder jumped during teaching. Check the arm and retry.")
        anchors = [samples[i] for i in _anchors(samples)]
        poses = [automatic_pose(s.pose, motion, joint_margin) for s in anchors]
        if start_pose is not None:
            poses[0] = automatic_pose(start_pose, motion, joint_margin)
        self.start, self.end = poses[0], poses[-1]
        self.keyframes = len(anchors)
        if max(abs(p[j] - poses[0][j]) for p in poses for j in JOINTS) < 0.5:
            raise ValueError("No movement detected within the replay limits.")
        spans = [max(0.12, (b.t - a.t) / speed) for a, b in zip(anchors, anchors[1:], strict=False)]
        tracks = {j: [pose[j] for pose in poses] for j in JOINTS}
        for j, values in tracks.items():
            if max(values) - min(values) < (0.6 if j == "gripper" else 0.55):
                time_total = anchors[-1].t - anchors[0].t
                tracks[j] = [
                    values[0] + (values[-1] - values[0]) * (s.t - anchors[0].t) / time_total
                    for s in anchors
                ]
        # Only lengthen the intervals causing a limit violation. Shared knot
        # velocities are recomputed whenever a neighboring interval changes.
        for _ in range(10):
            curves, scales = self._fit(tracks, spans, settings)
            if max(scales) <= 1.00001:
                break
            spans = [
                h * (scale * 1.02 if scale > 1 else 1)
                for h, scale in zip(spans, scales, strict=True)
            ]
        else:
            curves, scales = self._fit(tracks, spans, settings)
            # A shared fallback scale preserves continuity without another fit.
            spans = [h * max(scales) for h in spans]
        self.curves = curves
        self.times = [0.0]
        for span in spans:
            self.times.append(self.times[-1] + span)
        self.duration = self.times[-1]
        if self.duration > MAX_REPLAY_SECONDS:
            raise ValueError("The bounded replay would exceed 10 minutes. Teach a shorter path.")

    @staticmethod
    def _fit(tracks: dict[str, list[float]], spans: list[float], settings: Settings):
        scales = [1.0] * len(spans)
        curves: dict[str, list[tuple[float, float, float, float, float]]] = {}
        for j, values in tracks.items():
            tangents = _tangents(values, spans)
            curves[j] = []
            for i, h in enumerate(spans):
                p0, p1 = values[i], values[i + 1]
                delta = p1 - p0
                u, v = tangents[i] * h, tangents[i + 1] * h
                # Quintic Hermite: shared first derivatives and zero second
                # derivative at every knot, including stationary reversals.
                a = 6 * delta - 3 * (u + v)
                b = -15 * delta + 8 * u + 7 * v
                c = 10 * delta - 6 * u - 4 * v
                curves[j].append((a, b, c, u, p0))
                middle = 5 * delta - 2 * (u + v)
                velocity = _bernstein_peak((u, u, middle, v, v)) / h
                acceleration = _bernstein_peak((0, 4 * (middle - u), 4 * (v - middle), 0)) / h**2
                # Reserve acceleration for feedback-driven changes of path speed.
                scales[i] = max(
                    scales[i],
                    velocity / (0.65 * settings.velocity(j)),
                    sqrt(acceleration / (0.48 * settings.accel(j))),
                )
        return curves, scales

    def at(self, elapsed: float) -> Pose:
        if elapsed <= 0:
            return self.start.copy()
        if elapsed >= self.duration:
            return self.end.copy()
        i = bisect_right(self.times, elapsed) - 1
        u = (elapsed - self.times[i]) / (self.times[i + 1] - self.times[i])
        pose = {}
        for j, curves in self.curves.items():
            a, b, c, tangent, start = curves[i]
            value = start + tangent * u + u**3 * (c + u * (b + u * a))
            end = curves[i + 1][-1] if i + 1 < len(curves) else self.end[j]
            pose[j] = min(max(start, end), max(min(start, end), value))
        return pose


class ReversePath:
    """Traverse the same poses, including the gripper schedule, in reverse order."""

    def __init__(self, forward: Path, time_scale: float = 1.2):
        self.forward = forward
        self.time_scale = time_scale
        self.start, self.end = forward.end, forward.start
        self.duration = forward.duration * time_scale

    def at(self, elapsed: float) -> Pose:
        return self.forward.at(
            self.forward.duration - max(0.0, min(self.duration, elapsed)) / self.time_scale
        )


class AlignedPath:
    """Ease from measured encoders to the first knot before traversal."""

    def __init__(self, measured: Pose, path: Path, settings: Settings):
        self.path = path
        self.approach = (
            ParkTrajectory(measured, path.start, settings)
            if any(abs(measured[j] - path.start[j]) > 0.09 for j in JOINTS)
            else None
        )
        self.prefix = self.approach.duration if self.approach else 0.0
        self.duration = self.prefix + path.duration
        self.start, self.end = measured, path.end

    def at(self, elapsed: float) -> Pose:
        if self.approach and elapsed < self.prefix:
            return self.approach.at(elapsed)
        return self.path.at(elapsed - self.prefix)


class Player:
    """Slow the entire path as joints lag, preserving their coordination."""

    stall_seconds = 1.4

    def __init__(
        self,
        path: Path,
        motion: Motion,
        now: float,
        tolerance: float,
        *,
        settle_seconds: float = 0.55,
    ):
        self.path, self.motion = path, motion
        self.started = self.last = now
        self.cursor = 0.0
        self.rate = 1.0
        self.tolerance = tolerance
        self.settle_seconds = settle_seconds
        self.ended: float | None = None
        self.settled: tuple[float, Pose] | None = None
        self.tracking_joint: str | None = None
        self.stall_check: tuple[str, float, float] | None = None

    def step(self, now: float, observed: Pose) -> tuple[Pose, bool]:
        settings = self.motion.settings
        dt, self.last = now - self.last, now
        if not 0 <= dt <= settings.watchdog:
            raise SafetyFault("The motion loop missed its deadline.")
        self.motion.check_measurement(observed)
        errors = self._lag(observed)
        worst = max(JOINTS, key=lambda j: errors[j] / settings.lead(j))
        ratio = errors[worst] / settings.lead(worst)
        if ratio > 1:
            raise SafetyFault(self._joint_fault(worst, observed, "tracking limit exceeded"))
        if now - self.started > 3 * self.path.duration + 10:
            raise SafetyFault(self._joint_fault(worst, observed, "motion timed out"))

        if self.ended is None:
            self._check_progress(now, observed, worst, ratio)
            self.tracking_joint = worst if ratio > 0.34 else None
            x = max(0.0, min(1.0, (ratio - 0.30) / 0.52))
            desired = 1 - x**2 * (3 - 2 * x)
            # No catch-up jumps after scheduling delays.
            step = min(dt, 1.3 / settings.fps)
            previous_rate = self.rate
            self.rate += max(-2.5 * step, min(1.4 * step, desired - self.rate))
            self.cursor = min(
                self.path.duration, self.cursor + step * (self.rate + previous_rate) / 2
            )
            self.motion.target = self.motion.clamp(self.path.at(self.cursor))
            if self.cursor >= self.path.duration:
                self.ended = now
                self.tracking_joint = None
            else:
                return self.motion.target, False

        target = self.motion.target
        if now - self.ended > 5:
            worst = max(
                JOINTS,
                key=lambda j: abs(observed[j] - target[j]) / tolerance_for(j, self.tolerance),
            )
            raise SafetyFault(
                self._joint_fault(worst, observed, "could not settle at the endpoint")
            )
        return target, self._stationary(now, observed, target)

    def _lag(self, observed: Pose) -> Pose:
        return {j: abs(self.motion.target[j] - observed[j]) for j in JOINTS}

    def _stationary(self, now: float, observed: Pose, target: Pose) -> bool:
        if any(abs(observed[j] - target[j]) > tolerance_for(j, self.tolerance) for j in JOINTS):
            self.settled = None
        elif self.settled is None or any(
            abs(observed[j] - self.settled[1][j]) > (0.7 if j == "gripper" else 0.45)
            for j in JOINTS
        ):
            self.settled = (now, observed.copy())
        return self.settled is not None and now - self.settled[0] >= self.settle_seconds

    def _check_progress(self, now: float, observed: Pose, joint: str, ratio: float):
        if ratio < 0.48:
            self.stall_check = None
            return
        position = observed[joint]
        if self.stall_check is None or self.stall_check[0] != joint:
            self.stall_check = (joint, now, position)
            return
        _, since, previous_position = self.stall_check
        goal = self.motion.target[joint]
        if abs(goal - previous_position) - abs(goal - position) >= 0.45:
            self.stall_check = (joint, now, position)
        elif now - since > self.stall_seconds and self.rate < 0.25:
            raise SafetyFault(
                self._joint_fault(joint, observed, "stalled while following the path")
            )
        elif self.rate >= 0.25:
            self.stall_check = (joint, now, position)

    def _joint_fault(self, joint: str, observed: Pose, reason: str) -> str:
        unit = "%" if joint == "gripper" else "°"
        name = joint.replace("_", " ").capitalize()
        return (
            f"{name} {reason} (measured {observed[joint]:.1f}{unit}, "
            f"goal {self.motion.target[joint]:.1f}{unit}). Check for load or obstruction."
        )


class LoadTrim:
    """Slow outer-loop correction for steady error in the load-bearing joints."""

    def __init__(self, initial: Pose, held: Pose):
        self.bias = {j: held[j] - initial[j] for j in LOAD_AXES}
        self.filtered = dict.fromkeys(LOAD_AXES, 0.0)

    def update(self, dt: float, reference: Pose, observed: Pose):
        for j in LOAD_AXES:
            error = reference[j] - observed[j]
            previous = self.filtered[j]
            self.filtered[j] += min(1.0, dt / 0.20) * (error - previous)
            residual = self.filtered[j]
            requested = (max(0, residual - 0.7) + min(0, residual + 0.7)) * 0.8
            adjustment = max(-1.25, min(1.25, requested)) * dt
            current = self.bias[j]
            bounded = max(-TRIM_LIMIT, min(TRIM_LIMIT, current + adjustment))
            step = 1.25 * dt
            self.bias[j] = current + max(-step, min(step, bounded - current))

    def goal(self, reference: Pose, motion: Motion) -> Pose:
        command = reference.copy()
        for j in LOAD_AXES:
            if not self.bias[j]:
                continue
            lo, hi = motion.limits[j]
            # At a soft endpoint hold the taught position. An integrator must
            # never increase the force against a calibrated end stop.
            lo = min(reference[j], lo + JOINT_MARGIN)
            hi = max(reference[j], hi - JOINT_MARGIN)
            value = min(hi, max(lo, reference[j] + self.bias[j]))
            command[j] = value
        return command

    @staticmethod
    def limit(joint: str, settings: Settings) -> float:
        return settings.lead(joint) + (TRIM_LIMIT if joint in LOAD_AXES else 0)


class ServoPlayer(Player):
    """Stream bounded lookahead goals while the servos interpolate between packets."""

    stall_seconds = 3.3

    def __init__(
        self,
        path: Path,
        motion: Motion,
        now: float,
        tolerance: float,
        held: Pose,
        *,
        adjust_grip: bool = False,
    ):
        super().__init__(path, motion, now, tolerance)
        self.goal_cursor = 0.0
        self.issued = held
        self.adjust_grip = adjust_grip
        self.trim = LoadTrim(path.start, held)
        self.next_write = now + 0.030
        self.last_write = now

    def _lag(self, observed: Pose) -> Pose:
        future = self.path.at(self.cursor + 0.055)
        lag = {}
        for j in JOINTS:
            target = self.motion.target[j]
            direction = future[j] - target
            error = target - observed[j]
            lag[j] = (
                abs(error)
                if abs(direction) < 0.035
                else max(0.0, error if direction > 0 else -error)
            )
        return lag

    def step(self, now: float, observed: Pose) -> tuple[Pose, bool]:
        settings = self.motion.settings
        self.motion.check_measurement(observed)
        for joint in JOINTS:
            if abs(self.issued[joint] - observed[joint]) > self.trim.limit(joint, settings):
                unit = "%" if joint == "gripper" else "°"
                name = joint.replace("_", " ").capitalize()
                raise SafetyFault(
                    f"{name} tracking limit exceeded (measured {observed[joint]:.1f}{unit}, "
                    f"sent {self.issued[joint]:.1f}{unit}). Check for load or obstruction."
                )
        dt = min(max(0.0, now - self.last), 1.3 / settings.fps)
        target, done = super().step(now, observed)
        self.trim.update(dt, target, observed)
        if now >= self.next_write or done:
            advance = min(0.065, (now - self.last_write) * 1.30)
            self.next_write = now + 0.030
            self.last_write = now
            for horizon in (0.14, 0.07, 0.0):
                cursor = min(self.path.duration, self.cursor + horizon)
                if self.ended is None:
                    cursor = min(cursor, self.goal_cursor + advance)
                if self.ended is None and cursor <= self.goal_cursor:
                    self._refresh_grip(observed)
                    break
                candidate = self.trim.goal(self.path.at(cursor), self.motion)
                if all(
                    abs(candidate[j] - observed[j]) <= self.trim.limit(j, settings) - 0.25
                    for j in JOINTS
                ):
                    self.goal_cursor, self.issued = cursor, self.motion.clamp(candidate)
                    break
            else:
                self._refresh_grip(observed)
        return self.issued, done

    def _refresh_grip(self, observed: Pose):
        if self.adjust_grip:
            candidate = self.motion.clamp(self.path.at(self.goal_cursor))["gripper"]
            if abs(candidate - observed["gripper"]) <= self.motion.settings.gripper_max_lead - 0.25:
                self.issued = {**self.issued, "gripper": candidate}


class NativePlayer(Player):
    """Supervise a single point-to-point firmware profile without streaming setpoints."""

    def __init__(self, path: ParkTrajectory, motion: Motion, now: float, tolerance: float):
        super().__init__(path, motion, now, tolerance)
        self.origin = {
            j: (min(motion.target[j], path.start[j]), max(motion.target[j], path.start[j]))
            for j in JOINTS
        }
        self.motion.target = path.end
        self.progress_by_joint: dict[str, tuple[float, float]] = {}
        self.trim = LoadTrim(path.end, path.end)
        self.issued = path.end
        self.next_trim = now

    def step(self, now: float, observed: Pose) -> tuple[Pose, bool]:
        dt, self.last = now - self.last, now
        if not 0 <= dt <= self.motion.settings.watchdog:
            raise SafetyFault("The motion loop missed its deadline.")
        self.motion.check_measurement(observed)
        target = self.path.end
        elapsed = now - self.started
        self.cursor = min(self.path.duration, elapsed)
        stall_window = max(1.8, min(4.5, self.path.duration / 4))
        for j in JOINTS:
            tip = observed[j]
            lo, hi = self.origin[j]
            if not min(lo, target[j]) - 2.6 <= tip <= max(hi, target[j]) + 2.6:
                raise SafetyFault(self._joint_fault(j, observed, "left the return corridor"))
            error = abs(tip - target[j])
            if error <= tolerance_for(j, self.tolerance):
                self.progress_by_joint.pop(j, None)
                continue
            since, previous = self.progress_by_joint.setdefault(j, (now, error))
            if previous - error >= 0.30:
                self.progress_by_joint[j] = (now, error)
            elif now - since > stall_window:
                can_trim = (
                    j in LOAD_AXES
                    and error < self.motion.settings.lead(j)
                    and abs(self.trim.bias[j]) < TRIM_LIMIT - 0.15
                    and now - since < 4.8
                )
                if not can_trim:
                    raise SafetyFault(
                        self._joint_fault(j, observed, "stalled during the servo move")
                    )
        if elapsed > 2 * self.path.duration + 5:
            worst = max(JOINTS, key=lambda j: abs(observed[j] - target[j]))
            raise SafetyFault(
                self._joint_fault(worst, observed, "could not settle at the endpoint")
            )
        if elapsed >= min(0.6, self.path.duration * 0.5) and all(
            abs(observed[j] - target[j]) < self.motion.settings.lead(j) for j in LOAD_AXES
        ):
            self.trim.update(min(dt, 1.3 / self.motion.settings.fps), target, observed)
            if now >= self.next_trim:
                self.next_trim = now + 0.080
                self.issued = self.trim.goal(target, self.motion)
        return self.issued, self._stationary(now, observed, target)
