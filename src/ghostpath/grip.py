"""Bounded gripper edits recorded against the forward path clock."""

from bisect import bisect_right
from math import ceil, isfinite, sqrt

from .motion import JOINTS, Pose, Settings
from .trajectory import Path, ReplayPath, ReversePath, _bernstein_peak

GRIP_LIMIT = 6.0
INPUT_TIMEOUT = 0.23
REVERSE_PRELOAD = 2.5
STABLE_GRIP_WINDOW = 0.32
PRELOAD_RAMP = 0.55
Knots = list[tuple[float, float]]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _linear(points: Knots, position: float) -> float:
    if not points or position < points[0][0] or position > points[-1][0]:
        return 0.0
    if position == points[-1][0]:
        return points[-1][1]
    index = bisect_right(points, position, key=lambda point: point[0]) - 1
    a, b = points[index : index + 2]
    return a[1] + (b[1] - a[1]) * (position - a[0]) / (b[0] - a[0])


def _simplify(points: Knots) -> Knots:
    keep = {0, len(points) - 1}
    intervals = [(0, len(points) - 1)]
    while intervals:
        first, last = intervals.pop()
        a, b = points[first], points[last]
        slope = (b[1] - a[1]) / (b[0] - a[0])
        peak, index = 0.15, first
        for i in range(first + 1, last):
            x, y = points[i]
            error = abs(y - a[1] - slope * (x - a[0]))
            if error > peak:
                peak, index = error, i
        if index != first:
            keep.add(index)
            intervals.extend(((first, index), (index, last)))
    return [points[i] for i in sorted(keep)]


def _ramp(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value**3 * (10 + value * (-15 + 6 * value))


class GrippedPath:
    """Preserve arm geometry and retime only if the edited grip needs more time."""

    def __init__(self, base: ReplayPath, knots: Knots, settings: Settings):
        self.base, self.knots = base, knots
        self.start, self.end = base.start, base.end
        self.keyframes = base.keyframes
        if not 2 <= len(knots) <= 7000 or knots[0] != (0, 0) or knots[-1] != (1, 0):
            raise ValueError("Invalid gripper edit endpoints.")
        for i, (fraction, offset) in enumerate(knots):
            if not isfinite(fraction) or not isfinite(offset) or abs(offset) > GRIP_LIMIT:
                raise ValueError("Gripper edit exceeds its bounds.")
            if i and fraction <= knots[i - 1][0]:
                raise ValueError("Gripper edit times must be in order.")
        # A centered cubic B-spline filters capture noise without adding lag.
        # Zero outer control points keep both exact poses, even on a late key-up.
        segments = max(4, ceil(base.duration / 0.18))
        self.span = base.duration / segments
        values = [0.0, 0.0, 0.0]
        values.extend(_linear(knots, i / segments) for i in range(2, segments - 1))
        values.extend([0.0, 0.0, 0.0])
        self.curves: list[tuple[float, float, float, float]] = []
        base_times = base.times
        base_bounds = []
        for i, (a, b, c, u, start) in enumerate(base.curves["gripper"]):
            end = (
                base.curves["gripper"][i + 1][-1]
                if i + 1 < base.keyframes - 1
                else base.end["gripper"]
            )
            base_bounds.append(
                _quintic_bounds(
                    end - start, u, 5 * a + 4 * b + 3 * c + u, base_times[i + 1] - base_times[i]
                )
            )
        scale = 1.0
        for i in range(segments):
            left, start, end, right = values[i : i + 4]
            a = (-left + 3 * start - 3 * end + right) / 6
            b = (left - 2 * start + end) / 2
            c = (end - left) / 2
            d = (left + 4 * start + end) / 6
            self.curves.append((a, b, c, d))
            if a or b or c:
                velocity = _bernstein_peak((c, c + b, c + 2 * b + 3 * a)) / self.span
                acceleration = max(abs(2 * b), abs(6 * a + 2 * b)) / self.span**2
                first = bisect_right(base_times, i * self.span) - 1
                last = bisect_right(base_times, (i + 1) * self.span)
                overlap = base_bounds[first:last]
                scale = max(
                    scale,
                    (velocity + max(bound[0] for bound in overlap))
                    / (0.65 * settings.gripper_speed),
                    sqrt(
                        (acceleration + max(bound[1] for bound in overlap))
                        / (0.48 * settings.gripper_acceleration)
                    ),
                )
        self.scale = scale
        self.duration = base.duration * scale
        if self.duration > 600:
            raise ValueError("The gripper edits exceed the maximum replay duration.")

    def offset(self, fraction: float) -> float:
        if fraction <= 0 or fraction >= 1:
            return 0.0
        position = fraction * len(self.curves)
        i = min(int(position), len(self.curves) - 1)
        t = position - i
        a, b, u, start = self.curves[i]
        return _clamp(start + t * (u + t * (b + t * a)), -GRIP_LIMIT, GRIP_LIMIT)

    def at(self, elapsed: float) -> Pose:
        pose = self.base.at(elapsed / self.scale)
        pose["gripper"] = _clamp(pose["gripper"] + self.offset(elapsed / self.duration), 0, 100)
        return pose


def _quintic_bounds(delta: float, u: float, v: float, span: float) -> tuple[float, float]:
    middle = 5 * delta - 2 * (u + v)
    return (
        _bernstein_peak((u, u, middle, v, v)) / span,
        _bernstein_peak((0, 4 * (middle - u), 4 * (v - middle), 0)) / span**2,
    )


def _reverse_grasps(path: Path) -> list[tuple[float, float]]:
    """Find a close, a stable carrying interval, and a release in reverse order."""
    count = ceil(path.duration / 0.08)
    step = path.duration / count
    poses = [path.at(i * step) for i in range(count + 1)]
    grip = [pose["gripper"] for pose in poses]
    look = ceil(2.5 / step)
    settle = ceil(STABLE_GRIP_WINDOW / step)
    grasps = []
    i = settle
    while i < count - settle:
        level = grip[i]
        if (
            not 4 <= level <= 90
            or max(grip[max(0, i - look) : i]) < level + 5
            or max(abs(g - level) for g in grip[i : i + settle + 1]) > 1.2
            or any(
                max(poses[k][j] for k in range(i - settle, i + settle + 1))
                - min(poses[k][j] for k in range(i - settle, i + settle + 1))
                > 2.6
                for j in JOINTS[:-1]
            )
        ):
            i += 1
            continue
        end = i + settle + 1
        while end <= count and abs(grip[end] - level) <= 1.5:
            end += 1
        if end <= count and grip[end] < level:
            # A jaw still closing can look like a plateau for a few samples.
            i += 1
            continue
        if (
            end <= count
            and max(grip[end : end + look]) >= level + 5
            and any(
                abs(poses[k][j] - poses[i][j]) >= 4
                for k in range(i + settle, end)
                for j in JOINTS[:-1]
            )
        ):
            grasps.append((i * step, end * step))
            i = end + settle
        else:
            i = end
    return grasps


class CarryGripPath:
    """Close the jaws slightly more during inferred carries on the reverse path."""

    def __init__(self, forward: ReplayPath | GrippedPath):
        self.forward = forward
        self.reverse = ReversePath(forward, time_scale=1.0)
        self.start, self.end = self.reverse.start, self.reverse.end
        self.duration = self.reverse.duration
        self.grasps = [
            (pick, release)
            for pick, release in _reverse_grasps(self.reverse)
            if release - pick >= 2 * PRELOAD_RAMP and not self._grasp_has_edits(pick, release)
        ]
        self.pickups = [pick for pick, _ in self.grasps]

    def _grasp_has_edits(self, pick: float, release: float) -> bool:
        if not isinstance(self.forward, GrippedPath):
            return False
        steps = ceil((release - pick) / 0.08)
        return any(
            abs(self.forward.offset(1 - (pick + (release - pick) * i / steps) / self.duration))
            >= 0.25
            for i in range(steps + 1)
        )

    def preload(self, elapsed: float) -> float:
        i = bisect_right(self.pickups, elapsed) - 1
        if i < 0:
            return 0.0
        pick, release = self.grasps[i]
        return -REVERSE_PRELOAD * _ramp(min(elapsed - pick, release - elapsed) / PRELOAD_RAMP)

    def at(self, elapsed: float) -> Pose:
        pose = self.reverse.at(elapsed)
        pose["gripper"] = max(0.0, pose["gripper"] + self.preload(elapsed))
        return pose


class GripAssist:
    """Hold-to-adjust grip. Only changes actually issued during Boomerang are kept."""

    def __init__(
        self,
        forward: ReplayPath | GrippedPath,
        reverse: Path,
        prefix: float,
        carry_grip: CarryGripPath | None = None,
    ):
        self.forward = forward
        self.base = forward.base if isinstance(forward, GrippedPath) else forward
        self.reverse, self.prefix = reverse, prefix
        self.carry_grip = carry_grip
        self.start, self.end, self.duration = reverse.start, reverse.end, reverse.duration
        self.direction = 0
        self.last_input = 0.0
        self.bias = self.velocity = 0.0
        self.fraction = 1.0
        self.offset = 0.0
        self.trace: Knots = [(0.0, 0.0)]
        self.changed = False

    def input(self, direction: int, now: float):
        self.direction, self.last_input = direction, now

    def advance(self, now: float, dt: float, cursor: float):
        if cursor < self.prefix or cursor >= self.duration or now - self.last_input > INPUT_TIMEOUT:
            self.direction = 0
        target = self.direction * GRIP_LIMIT
        dt = _clamp(dt, 0, 0.025)
        # Ease velocity toward the requested offset, then brake near its destination.
        gap = target - self.bias
        desired = _clamp(8 * gap, -20, 20)
        self.velocity += _clamp(desired - self.velocity, -95 * dt, 95 * dt)
        next_bias = self.bias + self.velocity * dt
        self.bias = _clamp(next_bias, -GRIP_LIMIT, GRIP_LIMIT)
        if self.bias != next_bias:
            self.velocity = 0.0
        if abs(gap) < 0.025 and abs(self.velocity) < 0.25:
            self.bias, self.velocity = target, 0.0

    def _previous(self, fraction: float) -> float:
        return self.forward.offset(fraction) if isinstance(self.forward, GrippedPath) else 0.0

    def _fraction(self, elapsed: float) -> float:
        return _clamp(1 - (elapsed - self.prefix) / self.forward.duration, 0.0, 1.0)

    def _delta(self, elapsed: float) -> float:
        travel = elapsed - self.prefix
        if travel <= 0 or elapsed >= self.duration:
            return 0.0
        fraction = self._fraction(elapsed)
        fade = _ramp(travel / 0.24) * _ramp((self.duration - elapsed) / 0.48)
        previous = self._previous(fraction)
        if self.carry_grip:
            previous += self.carry_grip.preload(travel)
        offset = _clamp(previous + self.bias * fade, -GRIP_LIMIT, GRIP_LIMIT) - previous
        return offset

    def at(self, elapsed: float) -> Pose:
        pose = self.reverse.at(elapsed)
        pose["gripper"] = _clamp(pose["gripper"] + self._delta(elapsed), 0, 100)
        return pose

    def capture(self, elapsed: float, issued: float):
        if elapsed < self.prefix:
            return
        self.fraction = self._fraction(elapsed)
        progress = 1 - self.fraction
        reference = self.reverse.at(elapsed)["gripper"]
        self.offset = _clamp(issued - reference, -GRIP_LIMIT, GRIP_LIMIT)
        if (progress - self.trace[-1][0]) * self.base.duration >= 0.10 and progress < 1.0:
            saved = self.offset
            if self.carry_grip:
                # Save the issued grip when the operator adjusts it. Fade away
                # the reverse-only preload as the manual adjustment ends.
                preload = self.carry_grip.preload(elapsed - self.prefix)
                saved += preload * _ramp(abs(self.offset))
            self.trace.append((progress, _clamp(saved, -GRIP_LIMIT, GRIP_LIMIT)))
            self.changed = self.changed or abs(saved) >= 0.15

    def preview(self, fraction: float) -> float:
        position = 1 - fraction
        last, value = self.trace[-1]
        if last < position <= 1 - self.fraction:
            return value
        return _linear(self.trace, position)

    def result(self, settings: Settings) -> GrippedPath | None:
        if not self.changed:
            return None
        points = [(0.0, 0.0)] + [(1 - t, bias) for t, bias in reversed(self.trace)]
        axes = {x for x, _ in points}
        previous = self.forward.knots if isinstance(self.forward, GrippedPath) else []
        axes.update(x for x, _ in previous)
        merged = [
            (x, _clamp(_linear(previous, x) + _linear(points, x), -GRIP_LIMIT, GRIP_LIMIT))
            for x in sorted(axes)
        ]
        return GrippedPath(self.base, _simplify(merged), settings)
