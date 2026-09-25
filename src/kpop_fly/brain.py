"""The fly's brain: the MaleCNS v1.0 connectome (via flybrain) hearing sound through its antennae.

Sound drives the Johnston's organ (JO) neurons, the antennal mechanosensors flies hear with.
We read out descending neurons (DNs), the brain's commands to the body. Which DNs respond,
how fast and how strongly is not scripted: `calibrate()` measures it from the wiring by
playing the brain a 120 BPM pulse train and comparing against silence, seed for seed.

The responding DNs are grouped into body channels:
  front / mid / hind  the fastest, middle and slowest third of responders by latency, so each
                      beat ripples front to back down the body in the order the wiring sets
  left / right        by which side of the brain the DN sits on
  all                 every responder
Each channel is normalized to 0..1 between its silent baseline and its calibrated peak.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import flybrain
import numpy as np
from flybrain import FlyBrain

PERIOD = 25          # calibration beat period in 20 ms steps: 0.5 s = 120 BPM
PULSE = 3            # steps each calibration beat drives the antennae
CAL_VERSION = 1
CHANNELS = ("all", "front", "mid", "hind", "left", "right")
TIERS = ("front", "mid", "hind")
HEARING = ("all", "sound")


def hearing_neurons(brain: FlyBrain, mode: str = "all") -> np.ndarray:
    """JO neuron indices. "sound" keeps only JO-A/JO-B, the subtypes tuned to sound vibration;
    "all" also includes the wind/gravity-sensing subtypes (a loud beat deflects the antenna too)."""
    ct = np.asarray(brain.cell_type).astype(str)
    if mode == "sound":
        mask = np.char.startswith(ct, "JO-A") | np.char.startswith(ct, "JO-B")
    elif mode == "all":
        mask = np.char.startswith(ct, "JO")
    else:
        raise ValueError(f"hearing mode must be one of {HEARING}, got {mode!r}")
    return np.flatnonzero(mask)


@dataclass
class Calibration:
    dn: np.ndarray            # (k,) neuron indices of the DNs that respond to sound
    cell_type: np.ndarray     # (k,) str
    side: np.ndarray          # (k,) str: L / R / M ...
    latency: np.ndarray       # (k,) steps from pulse onset to peak response
    effect: np.ndarray        # (k,) extra spike probability at that peak
    tier: np.ndarray          # (k,) 0 front, 1 mid, 2 hind
    members: np.ndarray       # (len(CHANNELS), k) bool
    base: np.ndarray          # (len(CHANNELS),) expected spikes per step in silence
    peak: np.ndarray          # (len(CHANNELS),) spikes per step at the calibrated beat peak
    meta: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items() if k != "meta"}
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: Path) -> Calibration:
        with np.load(path) as z:
            fields = {k: z[k] for k in z.files if k != "meta"}
            return cls(meta=json.loads(str(z["meta"])), **fields)


def calibrate(brain: FlyBrain, jo: np.ndarray, amp: float = 1.0, seeds=(1, 2, 3), cycles: int = 12,
              progress: Callable[[str], None] | None = None) -> Calibration:
    sc = np.asarray(brain.superclass).astype(str)
    dn = np.flatnonzero(sc == "descending_neuron")
    slot = np.full(brain.n, -1)
    slot[dn] = np.arange(len(dn))

    def psth(a: float, seed: int) -> np.ndarray:
        """Spike probability of every DN at each phase of the beat cycle."""
        brain.reset(seed=seed)
        out = np.zeros((PERIOD, len(dn)))
        for s in range(PERIOD * cycles):
            phase = s % PERIOD
            fired = brain.step(inject=[(jo, a)] if a and phase < PULSE else ())
            if s >= PERIOD:                          # skip the first cycle's settling transient
                hit = slot[fired]
                out[phase, hit[hit >= 0]] += 1
        return out / (cycles - 1)

    quiet, driven = [], []
    for seed in seeds:
        if progress:
            progress(f"calibrating: seed {seed} ({len(quiet) + 1}/{len(seeds)})")
        quiet.append(psth(0.0, seed))
        driven.append(psth(amp, seed))
    quiet, driven = np.stack(quiet), np.stack(driven)

    diff = driven - quiet                            # paired: same seed, same noise to start
    evoked = diff.mean(0)
    sem = diff.std(0, ddof=1) / np.sqrt(len(seeds)) + 1.0 / (len(seeds) * (cycles - 1))
    cols = np.arange(len(dn))
    latency = evoked.argmax(0)
    effect = evoked[latency, cols]
    ok = (effect >= 0.1) & (effect > 3 * sem[latency, cols])
    if not ok.any():
        raise RuntimeError("no descending neuron responded to the antennae; try --hearing all or a higher --gain")

    latency, effect = latency[ok], effect[ok]
    order = np.lexsort((-effect, latency))           # fastest first, strongest first among ties
    tier = np.zeros(ok.sum(), int)
    for t, chunk in enumerate(np.array_split(order, len(TIERS))):
        tier[chunk] = t

    side = np.asarray(brain.side).astype(str)[dn[ok]]
    members = np.stack([
        np.ones(ok.sum(), bool),
        *(tier == t for t in range(len(TIERS))),
        side == "L",
        side == "R",
    ])
    quiet_rate = quiet.mean((0, 1))[ok]
    driven_psth = driven.mean(0)[:, ok]
    base = np.array([quiet_rate[m].sum() for m in members])
    peak = np.array([driven_psth[:, m].sum(1).max() if m.any() else 0.0 for m in members])

    return Calibration(
        dn=dn[ok], cell_type=np.asarray(brain.cell_type).astype(str)[dn[ok]], side=side,
        latency=latency, effect=effect, tier=tier, members=members, base=base, peak=peak,
        meta={"n_jo": int(len(jo)), "amp": amp, "seeds": list(seeds), "cycles": cycles,
              "n_dn_total": int(len(dn)), "flybrain": flybrain.__version__},
    )


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "kpop-fly"


@dataclass
class BrainFrame:
    step: int
    drive: float              # voltage injected into the JO neurons this step
    total_spikes: int         # neurons that fired anywhere in the CNS this step
    dn_fired: np.ndarray      # (k,) bool: which responsive DNs fired
    levels: np.ndarray        # (len(CHANNELS),) 0..1 channel activity (instant attack, slow release)
    step_ms: float            # wall time of the connectome step

    def level(self, channel: str) -> float:
        return float(self.levels[CHANNELS.index(channel)])


class ListeningBrain:
    """The whole CNS, stepped 20 ms at a time, with sound strength as antennal input."""

    def __init__(self, hearing: str = "all", gain: float = 1.0, seed: int = 64, recalibrate: bool = False,
                 log: Callable[[str], None] = print, release_s: float = 0.12, ring: float = 0.55):
        t0 = time.perf_counter()
        self.brain = FlyBrain(device="cpu", sensory_input=False, seed=seed)
        self.dt = self.brain.dt
        self.seed = seed
        self.gain = gain
        self.ring = ring
        self.jo = hearing_neurons(self.brain, hearing)
        log(f"connectome loaded: {self.brain.n:,} neurons in {time.perf_counter() - t0:.1f} s; "
            f"{len(self.jo)} JO neurons hearing ({hearing})")

        key = hashlib.sha1(json.dumps([CAL_VERSION, flybrain.__version__, hearing, PERIOD, PULSE]).encode())
        path = cache_dir() / f"calibration-{key.hexdigest()[:12]}.npz"
        if path.exists() and not recalibrate:
            self.cal = Calibration.load(path)
        else:
            t0 = time.perf_counter()
            self.cal = calibrate(self.brain, self.jo, progress=log)
            self.cal.save(path)
            log(f"calibrated in {time.perf_counter() - t0:.1f} s -> {path}")
        log(f"{len(self.cal.dn)} of {self.cal.meta['n_dn_total']} descending neurons respond to sound")

        self.brain.step()                                   # compile the numba kernels before real time starts
        self.brain.reset(seed=seed)
        self._slot = np.full(self.brain.n, -1)
        self._slot[self.cal.dn] = np.arange(len(self.cal.dn))
        self._members = self.cal.members.astype(float)
        span = self.cal.peak - self.cal.base
        self._span = np.where(span > 0, span, np.inf)
        self._release = 1.0 - np.exp(-self.dt / release_s)
        self._levels = np.zeros(len(CHANNELS))
        self._env = 0.0
        self._hold = 0
        self._step = 0

    @property
    def n_responsive(self) -> int:
        return len(self.cal.dn)

    def step(self, strength: float) -> BrainFrame:
        """Advance the brain 20 ms with `strength` (0..1) of sound hitting the antennae."""
        # A hit drives the antenna for PULSE steps, the same stimulus calibration measured, then rings down.
        if strength >= self._env:
            self._env, self._hold = strength, PULSE
        elif self._hold > 1:
            self._hold -= 1
        else:
            self._env *= self.ring
        drive = self.gain * self._env
        t0 = time.perf_counter()
        fired = self.brain.step(inject=[(self.jo, drive)] if drive > 0.02 else ())
        step_ms = (time.perf_counter() - t0) * 1000

        hit = self._slot[fired]
        dn_fired = np.zeros(len(self.cal.dn), bool)
        dn_fired[hit[hit >= 0]] = True
        raw = np.clip((self._members @ dn_fired - self.cal.base) / self._span, 0.0, 1.0)
        # Envelope follower: jump up with a burst, fall back over release_s.
        self._levels = np.where(raw > self._levels, raw, self._levels + self._release * (raw - self._levels))
        self._step += 1
        return BrainFrame(self._step, drive, len(fired), dn_fired, self._levels.copy(), step_ms)
