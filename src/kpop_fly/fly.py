"""The fly's body: pose targets -> springy pose -> 2D geometry. No drawing here, just points.

The fly stands upright on its hind legs facing you, cartoon style, so front legs read as
arms and hind legs as legs. Targets come from two places:

  brain (`targets()`)       scalar channels: squat, nod, lean, arm raise, mid legs, wings, antennae
  choreography (retarget)   explicit limbs: front- and hind-leg segment directions,
                            torso tilt, head shift and roll (see `retarget.py`)

Explicit limb keys, when present, override the brain's scalar version of that body part.
Sides are screen sides: -1 is screen left, +1 is screen right.

Brain springs are underdamped, so every burst overshoots and settles like a dancer hitting a
beat; choreography springs are stiff and critically damped, so they follow the dancer closely.
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


def limb_key(part: str, side: int, component: str = "") -> str:
    """Pose key for an explicit limb, e.g. limb_key("front", -1, "ux") -> "front-1_ux"."""
    return f"{part}{side:+d}" + (f"_{component}" if component else "")


SPRING_PROFILES = {
    "antenna": (900, 30),        # antennae twitch, they don't sway
    "lean": (60, 9),             # the body sways slowly
}
BRAIN_SPRING = (180, 14)         # underdamped: bursts overshoot and bounce
TRACK_SPRING = (1600, 80)        # critically damped, ~25 ms: follows the choreography, smooths the joins


@dataclass
class Dancer:
    springs: dict[str, Spring] = field(default_factory=dict)

    def _spring(self, key: str, first: float) -> Spring:
        if key not in self.springs:
            k, c = SPRING_PROFILES.get(key, BRAIN_SPRING if key in POSE_KEYS else TRACK_SPRING)
            self.springs[key] = Spring(stiffness=k, damping=c, x=first if key not in POSE_KEYS else 0.0)
        return self.springs[key]

    def follow(self, goals: dict[str, float], dt: float, stiffness: float | None = None) -> dict[str, float]:
        """Move every spring toward its goal. Keys seen for the first time start at their goal
        (brain keys start at rest), so switching a choreography on doesn't fling the limbs.
        `stiffness` retunes the choreography springs (critically damped) for this step."""
        pose = {}
        for k, v in goals.items():
            spring = self._spring(k, v)
            if stiffness is not None and k not in POSE_KEYS:
                spring.stiffness, spring.damping = stiffness, 2 * math.sqrt(stiffness)
            elif k not in POSE_KEYS:
                spring.stiffness, spring.damping = TRACK_SPRING
            pose[k] = spring.update(v, dt)
        return pose

    def update(self, levels: dict[str, float], drive: float, dt: float) -> dict[str, float]:
        return self.follow(targets(levels, drive), dt)


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


def _rotate(p: Point, pivot: Point, rad: float) -> Point:
    c, s = math.cos(rad), math.sin(rad)
    x, y = p[0] - pivot[0], p[1] - pivot[1]
    return pivot[0] + x * c - y * s, pivot[1] + x * s + y * c


def _unit(x: float, y: float) -> Point:
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-6 else (0.0, -1.0)


HIND_HIP = (22, 12)              # hind-leg hip joint, relative to (0, hip_y)
FEMUR, TIBIA = 72, 82
UPPER_ARM, FOREARM, HAND = 50, 48, 14


def skeleton(pose: dict[str, float], cx: float, ground: float, scale: float = 1.0) -> Skeleton:
    """Geometry in screen pixels. Built in a body frame (y up, floor at 0) then leaned and mapped."""
    g = pose.get
    nod, lean = g("nod", 0.0), g("lean", 0.0)
    torso = -math.radians(g("torso", 0.0))          # + degrees = upper body leans toward screen right

    # hind legs first: with choreography the legs decide how high the hips are
    knees: dict[int, Point] = {}
    if limb_key("hind", -1, "ux") in pose:
        rel = {}
        for side in (-1, 1):
            u = _unit(pose[limb_key("hind", side, "ux")], pose[limb_key("hind", side, "uy")])
            f = _unit(pose[limb_key("hind", side, "lx")], pose[limb_key("hind", side, "ly")])
            knee = (FEMUR * u[0], FEMUR * u[1])
            rel[side] = (knee, (knee[0] + TIBIA * f[0], knee[1] + TIBIA * f[1]))
        hip_y = min(max(-(HIND_HIP[1] + min(foot[1] for _, foot in rel.values())), 40.0), 160.0)  # lowest foot on the floor
        base = {s: (s * HIND_HIP[0], hip_y + HIND_HIP[1]) for s in (-1, 1)}
        knees = {s: (base[s][0] + k[0], base[s][1] + k[1]) for s, (k, _) in rel.items()}
        feet = {s: (base[s][0] + f[0], max(0.0, base[s][1] + f[1])) for s, (_, f) in rel.items()}
    else:
        hip_y = 125 - 35 * max(0.0, g("squat", 0.0))
        feet = {s: (s * 55.0, 0.0) for s in (-1, 1)}
    pivot = (0.0, hip_y)

    def up(p: Point) -> Point:                      # upper body: tilts with the torso about the hips
        return _rotate(p, pivot, torso)

    parts: dict[str, list] = {"legs": [], "wings": [], "eyes": [], "antennae": [], "stripes": []}

    wing_open = 15 + 55 * max(0.0, g("wings", 0.0))
    for side in (-1, 1):
        root = (side * 18, hip_y + 78)
        tilt = math.radians(180 + wing_open if side < 0 else -wing_open)
        tip = _polar(root, 58, math.degrees(tilt))
        mid = ((root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2)
        parts["wings"].append([up(p) for p in _ellipse(*mid, 62, 20, tilt)])

    parts["abdomen"] = _ellipse(0, hip_y - 8, 30, 46)
    for k in range(3):
        y = hip_y - 30 + 17 * k
        w = 30 * math.sqrt(max(0.0, 1 - ((y - (hip_y - 8)) / 46) ** 2))
        parts["stripes"].append([(-w, y), (w, y)])
    parts["thorax"] = [up(p) for p in _ellipse(0.0, hip_y + 48, 38, 46)]

    for side in (-1, 1):                                                     # front legs as arms
        shoulder = (side * 32, hip_y + 72)
        if limb_key("front", side, "ux") in pose:
            u = _unit(pose[limb_key("front", side, "ux")], pose[limb_key("front", side, "uy")])
            f = _unit(pose[limb_key("front", side, "lx")], pose[limb_key("front", side, "ly")])
            s0 = up(shoulder)
            elbow = (s0[0] + UPPER_ARM * u[0], s0[1] + UPPER_ARM * u[1])
            hand = (elbow[0] + FOREARM * f[0], elbow[1] + FOREARM * f[1])
            parts["legs"].append([s0, elbow, hand, (hand[0] + HAND * f[0], hand[1] + HAND * f[1])])
        else:
            arm = g("arm_l" if side < 0 else "arm_r", 0.0)
            ang = -60 + 170 * max(-0.1, arm)                                 # hanging down -> raised overhead
            deg = 180 - ang if side < 0 else ang
            elbow = _polar(shoulder, UPPER_ARM, deg)
            hand = _polar(elbow, FOREARM, deg + side * 35 * (1 - arm) - side * 25)
            parts["legs"].append([up(p) for p in (shoulder, elbow, hand, _polar(hand, HAND, deg + side * 10))])
    for side in (-1, 1):                                                     # mid legs, out to the sides
        root = (side * 36, hip_y + 44)
        deg = -35 + 55 * g(limb_key("mid", side), g("mid", 0.0))
        deg = 180 - deg if side < 0 else deg
        knee = _polar(root, 52, deg)
        foot = _polar(knee, 50, deg - side * 40)
        parts["legs"].append([up(p) for p in (root, knee, foot, _polar(foot, 12, -90))])
    for side in (-1, 1):                                                     # hind legs
        hip, foot = (side * HIND_HIP[0], hip_y + HIND_HIP[1]), feet[side]
        knee = knees.get(side) or _knee(hip, foot, FEMUR, TIBIA, side)
        toe = (foot[0] + side * 16, foot[1]) if foot[1] <= 1e-6 else _polar(foot, 16, -90 + side * 60)
        parts["legs"].append([hip, knee, foot, toe])

    # head: rides on the torso, but its roll is absolute (it comes from the dancer's ear line)
    head_c = up((6 * lean + g("head_dx", 0.0), hip_y + 48 + 46 + 28 - 12 * nod))
    roll = math.radians(g("head_roll", 0.0))
    head_r = 30.0

    def on_head(p: Point) -> Point:
        return _rotate((head_c[0] + p[0], head_c[1] + p[1]), head_c, roll)

    for side in (-1, 1):
        parts["eyes"].append([on_head(p) for p in _ellipse(side * 17, 3, 15, 19, side * 0.35)])
        base = (side * 7, 22)
        deg = 90 - side * (18 + 35 * max(0.0, g("antenna", 0.0)))
        seg = _polar(base, 20, deg)
        parts["antennae"].append([on_head(p) for p in (base, seg, _polar(seg, 22, deg - side * 25))])

    theta = math.radians(-9 * lean)                                          # the brain's sway, about the feet
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
