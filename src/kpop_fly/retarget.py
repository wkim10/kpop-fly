"""Human consensus dancer -> fly pose targets.

  arms   -> front legs   upper-arm and forearm directions carry over as-is, so elbows bend
  legs   -> hind legs    thigh and shin directions carry over the same way, so knees bend (or
                         stay straight) exactly as the dancer's do; the lowest foot stays on the floor
  torso  -> lean         the shoulders-over-hips angle tilts the fly's upper body
  head   -> nod          nose drop below the dancer's usual head carriage nods; sideways shift
                         moves the head; the ear line rolls it

Everything is in screen space: the dancers face the camera, so their right arm is on screen
left, and so is the fly limb it drives. You see the fly do exactly what you'd see in the video.
Limbs are assigned by which side of the screen they're on, so a dancer turning their back
doesn't cross the fly's arms.

Mid legs, wings and antennae have no human counterpart; the brain keeps driving those.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .choreo import CHOREO_DIR, KP, Choreo
from .fly import limb_key

HEAD_SHIFT = 40.0         # fly units per torso length of sideways head movement
NOD_RANGE = 0.2           # torso lengths of nose drop for a full nod
MAX_TORSO_DEG = 40.0
MAX_ROLL_DEG = 35.0
SMOOTH = np.array([1, 4, 6, 4, 1], float) / 16           # binomial, ~7 Hz cutoff at 30 fps: kills jitter, keeps hits

# a neutral standing pose (torso units, y up), for keypoints a song never shows
REST = {"nose": (0, 1.37), "left_eye": (0.05, 1.42), "right_eye": (-0.05, 1.42), "left_ear": (0.1, 1.4),
        "right_ear": (-0.1, 1.4), "left_shoulder": (0.29, 1), "right_shoulder": (-0.29, 1),
        "left_elbow": (0.35, 0.45), "right_elbow": (-0.35, 0.45), "left_wrist": (0.37, -0.05),
        "right_wrist": (-0.37, -0.05), "left_hip": (0.15, 0), "right_hip": (-0.15, 0), "left_knee": (0.16, -0.8),
        "right_knee": (-0.16, -0.8), "left_ankle": (0.17, -1.6), "right_ankle": (-0.17, -1.6)}


def fill_gaps(t: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Linearly interpolate unseen keypoints over time; hold the nearest value at the ends."""
    out = pose.astype(float).copy()
    for k, name in enumerate(REST):
        for c in range(2):
            v = out[:, k, c]
            ok = np.isfinite(v)
            out[:, k, c] = np.interp(t, t[ok], v[ok]) if ok.any() else REST[name][c]
    return out


def smooth(pose: np.ndarray) -> np.ndarray:
    pad = len(SMOOTH) // 2
    padded = np.concatenate([pose[:1].repeat(pad, 0), pose, pose[-1:].repeat(pad, 0)])
    return sum(w * padded[i:i + len(pose)] for i, w in enumerate(SMOOTH))


def _unit(v: np.ndarray) -> tuple[float, float]:
    n = float(np.hypot(*v))
    return (float(v[0]) / n, float(v[1]) / n) if n > 1e-6 else (0.0, -1.0)


def _screen_sides(h: np.ndarray, joint: str) -> dict[int, str]:
    """Which of the dancer's sides is on screen left (-1) and right (+1), judged by this joint pair."""
    facing_camera = h[KP[f"left_{joint}"], 0] >= h[KP[f"right_{joint}"], 0]
    return {-1: "right", 1: "left"} if facing_camera else {-1: "left", 1: "right"}


def retarget(h: np.ndarray, neutral_head: np.ndarray) -> dict[str, float]:
    """One (17,2) body-relative human pose -> fly pose targets."""
    p = lambda name: h[KP[name]]
    out: dict[str, float] = {}

    for part, (root, joint, end), pair in (("front", ("shoulder", "elbow", "wrist"), "shoulder"),
                                           ("hind", ("hip", "knee", "ankle"), "hip")):
        for side, who in _screen_sides(h, pair).items():
            (ux, uy) = _unit(p(f"{who}_{joint}") - p(f"{who}_{root}"))
            (lx, ly) = _unit(p(f"{who}_{end}") - p(f"{who}_{joint}"))
            out |= {limb_key(part, side, "ux"): ux, limb_key(part, side, "uy"): uy,
                    limb_key(part, side, "lx"): lx, limb_key(part, side, "ly"): ly}

    shoulders = (p("left_shoulder") + p("right_shoulder")) / 2
    hips = (p("left_hip") + p("right_hip")) / 2
    v = shoulders - hips
    out["torso"] = float(np.clip(math.degrees(math.atan2(v[0], v[1])), -MAX_TORSO_DEG, MAX_TORSO_DEG))

    d = (p("nose") - shoulders) - neutral_head
    out["nod"] = float(np.clip(-d[1] / NOD_RANGE, -0.6, 1.5))
    out["head_dx"] = float(np.clip(d[0] * HEAD_SHIFT, -20, 20))
    ears = p("left_ear") - p("right_ear")
    if ears[0] < 0:                                   # back to camera: measure the line the other way
        ears = -ears
    out["head_roll"] = (float(np.clip(math.degrees(math.atan2(ears[1], ears[0])), -MAX_ROLL_DEG, MAX_ROLL_DEG))
                        if np.hypot(*ears) > 0.1 else 0.0)
    return out


class ChoreoTrack:
    """A song's consensus dancer, gap-filled and smoothed, sampled at any time t (seconds)."""

    def __init__(self, choreo: Choreo, path: Path | None = None):
        self.choreo = choreo
        self.path = path
        self.t = choreo.t.astype(float)
        self.pose = smooth(fill_gaps(self.t, choreo.pose))
        shoulders = (self.pose[:, KP["left_shoulder"]] + self.pose[:, KP["right_shoulder"]]) / 2
        self.neutral_head = np.median(self.pose[:, KP["nose"]] - shoulders, axis=0)
        self.n_dancers = choreo.n_dancers

    @classmethod
    def load(cls, name: str | Path) -> ChoreoTrack:
        """A slug from choreo/ ("fancy") or a path to an .npz."""
        path = Path(name)
        if path.suffix != ".npz":
            path = CHOREO_DIR / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"no choreography at {path}; run `kpop-fly extract {name}` first")
        return cls(Choreo.load(path), path)

    @property
    def title(self) -> str:
        m = self.choreo.meta
        return f"{m.get('artist', '')} - {m.get('title', m['slug'])}".strip(" -")

    @property
    def audio(self) -> Path:
        """The song saved next to the choreography (its name is recorded in the file's meta)."""
        return self.path.parent / self.choreo.meta["audio"]

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def _index(self, t: float) -> tuple[int, float]:
        t = min(max(t, self.t[0]), self.t[-1])
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        return i, (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)

    def human_at(self, t: float) -> np.ndarray:
        i, a = self._index(t)
        return (1 - a) * self.pose[i] + a * self.pose[i + 1]

    def dancers_at(self, t: float) -> int:
        return int(self.n_dancers[self._index(t)[0]])

    def targets_at(self, t: float) -> dict[str, float]:
        return retarget(self.human_at(t), self.neutral_head)
