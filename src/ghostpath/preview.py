"""Procedural simulator camera views. Every render is derived from joint feedback."""

import math
from functools import lru_cache

from PIL import Image, ImageDraw

from .motion import Pose


@lru_cache(maxsize=8)
def _backdrop(width: int, height: int, wrist: bool) -> Image.Image:
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    horizon = int(height * (0.30 if wrist else 0.21))
    for y in range(height):
        t = y / height
        palette = (26, 27, 30) if y < horizon else (45, 40, 36)
        light = 19 * math.sin(t * math.pi * 0.83)
        draw.line((0, y, width, y), tuple(int(c + light) for c in palette))
    for u in range(-6, 9):
        x = width * (0.5 + u * 0.10)
        draw.line((width * 0.48 + u * width * 0.031, horizon, x, height), fill="#53504a")
    for i in range(1, 9):
        y = horizon + (height - horizon) * (i / 8) ** 1.7
        draw.line((0, y, width, y), fill="#53504a")
    return image


def render_camera(pose: Pose, width: int, height: int, side: int = 0) -> Image.Image:
    wrist = side % 2 == 1
    backdrop = _backdrop(width, height, wrist).copy()
    d = ImageDraw.Draw(backdrop)
    s = min(width / 640, height / 480)

    def point(x, y):
        return x * width, y * height

    def polygon(points, fill, outline=None):
        d.polygon([point(*p) for p in points], fill=fill, outline=outline)

    def ellipse(x, y, w, h, fill, outline=None):
        d.ellipse((*point(x - w, y - h), *point(x + w, y + h)), fill=fill, outline=outline)

    def box(x, y, w, h, fill, top, right):
        skew = h * 0.34 * height / width
        polygon([(x, y), (x + skew, y - h * 0.32), (x + w + skew, y - h * 0.32), (x + w, y)], top)
        polygon(
            [
                (x + w, y),
                (x + w + skew, y - h * 0.32),
                (x + w + skew, y + h * 0.68),
                (x + w, y + h),
            ],
            right,
        )
        polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill)

    if wrist:
        polygon([(0.08, 0.72), (0.62, 0.66), (0.95, 0.81), (0.29, 0.94)], "#373432", "#67615a")
        polygon([(0.13, 0.72), (0.62, 0.69), (0.88, 0.81), (0.29, 0.91)], "#252529", "#4b4641")
        bx = 0.49 + pose["shoulder_pan"] * 0.003
        by = 0.64 - pose["shoulder_lift"] * 0.0013
        ellipse(bx + 0.04, by + 0.11, 0.095, 0.032, "#242129")
        box(bx, by, 0.14, 0.14, "#9c4824", "#e69c5e", "#6f3221")
        x = 0.59 - pose["shoulder_pan"] * 0.002
        gap = 0.035 + pose["gripper"] * 0.00065
        polygon(
            [(x - 0.13, 0), (x + 0.13, 0), (x + 0.097, 0.24), (x - 0.088, 0.29)],
            "#333742",
            "#6a6d73",
        )
        polygon(
            [(x - 0.088, 0.27), (x + 0.097, 0.23), (x + 0.068, 0.38), (x - 0.065, 0.42)],
            "#17181c",
            "#686669",
        )
        for sign in (-1, 1):
            q = x + sign * gap
            polygon(
                [
                    (x + sign * 0.056, 0.35),
                    (q + sign * 0.026, 0.43),
                    (q + sign * 0.031, 0.56),
                    (q, 0.60),
                    (q - sign * 0.019, 0.51),
                    (x + sign * 0.021, 0.39),
                ],
                "#44454d",
                "#777676",
            )
    else:
        polygon([(0.55, 0.70), (0.76, 0.67), (0.89, 0.76), (0.64, 0.84)], "#44413e", "#6b6259")
        polygon([(0.58, 0.711), (0.76, 0.688), (0.85, 0.753), (0.65, 0.815)], "#29292b", "#5b514a")
        box(0.67, 0.65, 0.069, 0.09, "#9b4322", "#e7995b", "#6c301e")
        ellipse(0.37, 0.802, 0.158, 0.045, "#302c29")
        box(0.24, 0.724, 0.219, 0.055, "#272831", "#53565e", "#1f2028")
        box(0.29, 0.642, 0.136, 0.09, "#353845", "#666b72", "#23242f")
        shoulder = (0.357, 0.634)
        swing = pose["shoulder_pan"] * 0.0026
        elbow = (0.37 + swing, 0.31 - pose["shoulder_lift"] * 0.0018)
        hand = (0.61 + swing + pose["elbow_flex"] * 0.0018, 0.40 + pose["wrist_flex"] * 0.002)
        for a, b, thick in ((shoulder, elbow, 45), (elbow, hand, 38)):
            ax, ay = point(*a)
            bx, by = point(*b)
            d.line(
                (ax + 5 * s, ay + 5 * s, bx + 5 * s, by + 5 * s),
                fill="#1d1e24",
                width=max(2, int(thick * s)),
            )
            d.line((ax, ay, bx, by), fill="#7a7c82", width=max(2, int(thick * s)))
            d.line(
                (ax + 3 * s, ay + 3 * s, bx + 3 * s, by + 3 * s),
                fill="#484b54",
                width=max(2, int((thick - 8) * s)),
            )
            d.line(
                (ax - 6 * s, ay - 6 * s, bx - 6 * s, by - 6 * s),
                fill="#62656c",
                width=max(2, int(4 * s)),
            )
        for x, y, r in ((*shoulder, 0.040), (*elbow, 0.039), (*hand, 0.029)):
            ellipse(x, y, r, r * 1.18, "#272932", "#88888a")
            ellipse(x, y, r * 0.52, r * 0.63, "#4c4d55", "#696a72")
            ellipse(x, y, r * 0.12, r * 0.15, "#999697")
        x, y = hand
        gap = 0.011 + 0.00027 * pose["gripper"]
        polygon(
            [
                (x + 0.014, y - 0.015),
                (x + 0.067, y + 0.039),
                (x + 0.046, y + 0.072),
                (x + 0.006, y + 0.016),
            ],
            "#252630",
            "#6b6b72",
        )
        for sign in (-1, 1):
            dx = sign * gap
            polygon(
                [
                    (x + 0.052 + dx, y + 0.047),
                    (x + 0.079 + dx, y + 0.085),
                    (x + 0.077 + dx, y + 0.110),
                    (x + 0.064 + dx, y + 0.102),
                    (x + 0.064 + dx, y + 0.083),
                    (x + 0.041 + dx, y + 0.061),
                ],
                "#44444c",
                "#74737a",
            )
    return backdrop
