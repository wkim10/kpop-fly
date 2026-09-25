"""The fly's body: brain channels -> springy pose -> 2D geometry. No drawing here, just points.

The fly stands upright on its hind legs facing you, cartoon style, so front legs read as
arms. That is also the mapping Phase 2 will use for human choreography (arms -> front legs,
legs -> hind legs).

What moves and when comes from the brain. How far each body part moves per unit of DN
activity is set by the artistic constants in `targets()`. The springs are underdamped,
so every burst overshoots and settles like a dancer hitting a beat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class Spring:
    stiffness: float = 180.0
    damping: float = 14.0
    x: float = 0.0
    v: float = 0.0

    def update(self, target: float, dt: float) -> float:
        n = max(1, math.ceil(dt / 0.004))                         # substep for stability at low fps
        h = dt / n
        for _ in range(n):
            self.v += (self.stiffness * (target - self.x) - self.damping * self.v) * h
            self.x += self.v * h
        return self.x


POSE_KEYS = ("squat", "nod", "lean", "arm_l", "arm_r", "mid", "wings", "antenna")


def targets(levels: dict[str, float], drive: float) -> dict[str, float]:
    """Artistic mapping from brain channels (0..1) to pose targets."""
    bias = float(np.clip(levels["left"] - levels["right"], -1, 1))
    front = levels["front"]
    return {
        "squat": levels["hind"],                                 # slow responders: bounce on the knees
        "nod": levels["all"],                                    # everyone: head bob
        "lean": bias,                                            # left vs right brain
        "arm_l": min(1.0, front * (1 + 0.6 * bias)),             # fast responders: arms
        "arm_r": min(1.0, front * (1 - 0.6 * bias)),
        "mid": levels["mid"],                                    # middle responders: mid legs
        "wings": max(0.0, levels["all"] - 0.35) / 0.65,          # only the big bursts flare wings
        "antenna": min(1.0, drive),                              # the sensory input itself
    }


@dataclass
class Dancer:
    springs: dict[str, Spring] = field(default_factory=lambda: {k: Spring() for k in POSE_KEYS})

    def __post_init__(self):
        self.springs["antenna"] = Spring(stiffness=900, damping=30)   # antennae twitch, they don't sway
        self.springs["lean"] = Spring(stiffness=60, damping=9)        # the body sways slowly

    def update(self, levels: dict[str, float], drive: float, dt: float) -> dict[str, float]:
        return {k: self.springs[k].update(v, dt) for k, v in targets(levels, drive).items()}


@dataclass
class Skeleton:
    thorax: list[Point]
    abdomen: list[Point]
    stripes: list[list[Point]]
    wings: list[list[Point]]
    legs: list[list[Point]]          # front L/R, mid L/R, hind L/R
    head: tuple[Point, float]        # center, radius
    eyes: list[list[Point]]
    antennae: list[list[Point]]
    ground: float


def _ellipse(cx, cy, rx, ry, angle=0.0, n=28) -> list[Point]:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = math.cos(angle), math.sin(angle)
    return list(zip(cx + x * c - y * s, cy + x * s + y * c))


def _polar(p: Point, length: float, deg: float) -> Point:
    a = math.radians(deg)
    return p[0] + length * math.cos(a), p[1] + length * math.sin(a)


def _knee(hip: Point, foot: Point, a: float, b: float, outward: float) -> Point:
    """Two-bone IK: the knee joining hip and foot, bent away from the body's midline."""
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    d = min(max(math.hypot(dx, dy), abs(a - b) + 1e-3), a + b - 1e-3)
    base = math.atan2(dy, dx)
    off = math.acos((a * a + d * d - b * b) / (2 * a * d))
    knees = [(hip[0] + a * math.cos(base + sgn * off), hip[1] + a * math.sin(base + sgn * off)) for sgn in (1, -1)]
    return max(knees, key=lambda k: k[0] * outward)


def skeleton(pose: dict[str, float], cx: float, ground: float, scale: float = 1.0) -> Skeleton:
    """Geometry in screen pixels. Built in a body frame (y up, feet at 0) then leaned and mapped."""
    squat, nod, lean = pose["squat"], pose["nod"], pose["lean"]
    hip_y = 125 - 35 * max(0.0, squat)
    thorax_c = (0.0, hip_y + 48)
    parts: dict[str, list] = {"legs": [], "wings": [], "eyes": [], "antennae": [], "stripes": []}

    wing_open = 15 + 55 * max(0.0, pose["wings"])
    for side in (-1, 1):
        root = (side * 18, hip_y + 78)
        tilt = math.radians(180 + wing_open if side < 0 else -wing_open)
        tip = _polar(root, 58, math.degrees(tilt))
        mid = ((root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2)
        parts["wings"].append(_ellipse(*mid, 62, 20, tilt))

    parts["abdomen"] = _ellipse(0, hip_y - 8, 30, 46)
    for k in range(3):
        y = hip_y - 30 + 17 * k
        w = 30 * math.sqrt(max(0.0, 1 - ((y - (hip_y - 8)) / 46) ** 2))
        parts["stripes"].append([(-w, y), (w, y)])
    parts["thorax"] = _ellipse(*thorax_c, 38, 46)

    for side, arm in ((-1, pose["arm_l"]), (1, pose["arm_r"])):          # front legs as arms
        shoulder = (side * 32, hip_y + 72)
        ang = -60 + 170 * max(-0.1, arm)                                     # hanging down -> raised overhead
        deg = 180 - ang if side < 0 else ang
        elbow = _polar(shoulder, 50, deg)
        hand = _polar(elbow, 48, deg + side * 35 * (1 - arm) - side * 25)
        parts["legs"].append([shoulder, elbow, hand, _polar(hand, 14, deg + side * 10)])
    for side in (-1, 1):                                                     # mid legs, out to the sides
        root = (side * 36, hip_y + 44)
        deg = -35 + 55 * pose["mid"]
        deg = 180 - deg if side < 0 else deg
        knee = _polar(root, 52, deg)
        foot = _polar(knee, 50, deg - side * 40)
        parts["legs"].append([root, knee, foot, _polar(foot, 12, -90)])
    for side in (-1, 1):                                                     # hind legs, standing
        hip, foot = (side * 22, hip_y + 12), (side * 55, 0.0)
        knee = _knee(hip, foot, 72, 82, side)
        parts["legs"].append([hip, knee, foot, (foot[0] + side * 16, 0.0)])

    head_c = (6 * lean, hip_y + 48 + 46 + 28 - 12 * nod)
    head_r = 30.0
    for side in (-1, 1):
        parts["eyes"].append(_ellipse(head_c[0] + side * 17, head_c[1] + 3, 15, 19, side * 0.35))
        base = (head_c[0] + side * 7, head_c[1] + 22)
        deg = 90 - side * (18 + 35 * max(0.0, pose["antenna"]))
        seg = _polar(base, 20, deg)
        parts["antennae"].append([base, seg, _polar(seg, 22, deg - side * 25)])

    theta = math.radians(-9 * lean)                                          # lean about the feet
    c, s = math.cos(theta), math.sin(theta)

    def to_screen(p: Point) -> Point:
        x, y = p[0] * c - p[1] * s, p[0] * s + p[1] * c
        return cx + x * scale, ground - y * scale

    def m(pts):
        return [to_screen(p) for p in pts]

    return Skeleton(
        thorax=m(parts["thorax"]), abdomen=m(parts["abdomen"]), stripes=[m(p) for p in parts["stripes"]],
        wings=[m(p) for p in parts["wings"]], legs=[m(p) for p in parts["legs"]],
        head=(to_screen(head_c), head_r * scale), eyes=[m(p) for p in parts["eyes"]],
        antennae=[m(p) for p in parts["antennae"]], ground=ground,
    )
