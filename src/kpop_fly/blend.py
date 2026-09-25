"""Who moves the fly: the dance, the brain, or both.

  dance   the choreography alone: arms, legs, torso and head copy the consensus dancer;
          the brain keeps running (and shows in the panel) but drives nothing
  brain   Phase 1: the brain's descending neurons drive a generic beat dance
  blend   the dance decides WHAT the moves are; the brain decides WHEN they land and HOW HARD:
            snap       on a DN burst the fly snaps into the dancer's pose; between bursts it
                       follows loosely (spring stiffness tracks the "all" channel)
            arm hits   fast responders push arm moves past the dancer's (up to HIT x) on a
                       burst and relax them (QUIET x) between bursts
            knees      slow responders scale the leg moves the same way and add a knee bounce,
                       so each beat ripples from the arms down to the legs in wiring order
            head       all responders scale torso and head moves and add a nod
            wings, mid legs, antennae   straight from the brain, as in Phase 1

The moves are scaled around a neutral standing pose, so "70%" means 70% of the way from
standing to the dancer's pose. All constants below are artistic; the timing isn't.

On dense pop music the descending neurons never fall quiet: their level swings over roughly
0.25-0.65 instead of the metronome's 0.1-0.8. `Blender` therefore rescales each channel to its
own range over the last few seconds (automatic contrast, like the onset detector's), so the
brain's relative response to the song uses the whole QUIET..HIT range.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .brain import CHANNELS
from .choreo import KP
from .fly import limb_key, targets
from .retarget import REST, retarget

MODES = ("blend", "dance", "brain")
QUIET, HIT = 0.7, 1.3          # move amplitude between bursts, and at a full burst
MAX_SWING = 175.0              # degrees from neutral a scaled limb segment may reach
BOUNCE_DEG = 22.0              # knee flex at a full slow-responder burst
NOD_KICK = 0.5                 # extra nod at a full burst
SNAP = (250.0, 1600.0)         # choreography spring stiffness: quiet brain -> full burst
BRAIN_PARTS = ("mid", "wings", "antenna")

_REST = retarget(np.array([REST[k] for k in KP], float), np.array([0.0, REST["nose"][1] - REST["left_shoulder"][1]]))


def _amplify(goals: dict, out: dict, part: str, factor: float) -> None:
    """Scale a limb's segment directions away from (or toward) the neutral pose."""
    for side in (-1, 1):
        for seg in ("u", "l"):
            kx, ky = limb_key(part, side, f"{seg}x"), limb_key(part, side, f"{seg}y")
            rest = math.atan2(_REST[ky], _REST[kx])
            d = math.remainder(math.atan2(goals[ky], goals[kx]) - rest, math.tau)
            a = rest + math.radians(float(np.clip(math.degrees(d) * factor, -MAX_SWING, MAX_SWING)))
            out[kx], out[ky] = math.cos(a), math.sin(a)


def _bounce(out: dict, amount: float) -> None:
    """Flex both knees outward: thighs swing out, shins back in, and the hips drop."""
    theta = math.radians(BOUNCE_DEG * amount)
    for side in (-1, 1):
        for seg, sign in (("u", 1), ("l", -1)):
            kx, ky = limb_key("hind", side, f"{seg}x"), limb_key("hind", side, f"{seg}y")
            a = math.atan2(out[ky], out[kx]) + side * sign * theta
            out[kx], out[ky] = math.cos(a), math.sin(a)


def compose(mode: str, levels: dict[str, float], drive: float,
            choreo: dict[str, float] | None) -> tuple[dict[str, float], float | None]:
    """Pose goals for this frame, and the stiffness for the choreography springs (None = default)."""
    brain = targets(levels, drive)
    if mode == "brain" or choreo is None:
        return brain, None
    if mode == "dance":
        return {**choreo, "mid": 0.0, "wings": 0.0, "antenna": 0.0}, None
    if mode != "blend":
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    level = lambda ch: float(np.clip(levels[ch], 0.0, 1.0))
    factor = lambda ch: QUIET + (HIT - QUIET) * level(ch)
    out = dict(choreo)
    _amplify(choreo, out, "front", factor("front"))
    _amplify(choreo, out, "hind", factor("hind"))
    _bounce(out, level("hind"))
    f = factor("all")
    out["torso"] = choreo["torso"] * f
    out["head_dx"] = choreo["head_dx"] * f
    out["head_roll"] = choreo["head_roll"] * f
    out["nod"] = choreo["nod"] * f + NOD_KICK * level("all")
    out |= {k: brain[k] for k in BRAIN_PARTS}
    return out, SNAP[0] + (SNAP[1] - SNAP[0]) * level("all")


class Blender:
    """compose() plus automatic contrast for blended mode. Call once per display frame."""

    def __init__(self, window_s: float = 4.0, floor: float = 0.15):
        self.window_s = window_s
        self.floor = floor                                  # minimum range, so silence isn't amplified into moves
        self._hist: deque[tuple[float, np.ndarray]] = deque()
        self._t = 0.0

    def contrast(self, levels: dict[str, float], dt: float) -> dict[str, float]:
        x = np.array([levels[c] for c in CHANNELS])
        self._t += dt
        self._hist.append((self._t, x))
        while self._hist[0][0] < self._t - self.window_s:
            self._hist.popleft()
        h = np.array([v for _, v in self._hist])
        lo, hi = np.percentile(h, 10, axis=0), np.percentile(h, 95, axis=0)
        norm = np.clip((x - lo) / np.maximum(hi - lo, self.floor), 0.0, 1.0)
        return dict(zip(CHANNELS, norm.tolist()))

    def __call__(self, mode: str, levels: dict[str, float], drive: float, choreo: dict[str, float] | None,
                 dt: float) -> tuple[dict[str, float], float | None]:
        contrasted = self.contrast(levels, dt)              # keep the window current in every mode
        return compose(mode, contrasted if mode == "blend" and choreo else levels, drive, choreo)
