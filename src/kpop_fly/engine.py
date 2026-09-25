"""Audio -> onsets -> brain, with no clock of its own.

`Pipeline` is fed audio from any thread and stepped 20 ms at a time. `BrainThread` steps it
in real time for live runs; `analyze` steps it as fast as possible over a whole recording.
`Timeline` keeps the last few seconds for the display.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .beat import OnsetDetector
from .brain import CHANNELS, BrainFrame, ListeningBrain


class Timeline:
    """Ring buffers of the last `length` brain steps, written by the brain thread, read by the display."""

    def __init__(self, n_dn: int, length: int = 250):
        self.length = length
        self.lock = threading.Lock()
        self.strength = np.zeros(length)
        self.drive = np.zeros(length)
        self.beat = np.zeros(length, bool)
        self.levels = np.zeros((length, len(CHANNELS)))
        self.raster = np.zeros((n_dn, length), bool)
        self.latest: BrainFrame | None = None
        self.step_ms = deque(maxlen=50)
        self._i = 0

    def push(self, frame: BrainFrame, strength: float, beat: bool) -> None:
        with self.lock:
            i = self._i % self.length
            self.strength[i], self.drive[i], self.beat[i] = strength, frame.drive, beat
            self.levels[i] = frame.levels
            self.raster[:, i] = frame.dn_fired
            self.latest = frame
            self.step_ms.append(frame.step_ms)
            self._i += 1

    def snapshot(self) -> dict:
        """Oldest-to-newest copies of every buffer."""
        with self.lock:
            shift = -(self._i % self.length)
            return {
                "strength": np.roll(self.strength, shift), "drive": np.roll(self.drive, shift),
                "beat": np.roll(self.beat, shift), "levels": np.roll(self.levels, shift, axis=0),
                "raster": np.roll(self.raster, shift, axis=1), "latest": self.latest,
                "step_ms": float(np.mean(self.step_ms)) if self.step_ms else 0.0,
            }


class Pipeline:
    def __init__(self, brain: ListeningBrain, sr: int):
        self.brain = brain
        self.detector = OnsetDetector(sr)
        self.timeline = Timeline(brain.n_responsive)
        self.beats = 0
        self._pending: deque[tuple[float, float, bool]] = deque()
        self._lock = threading.Lock()

    def feed(self, mono: np.ndarray) -> None:
        """Called from the audio thread with each block of sound."""
        events = self.detector.process(mono)
        with self._lock:
            self._pending.extend(events)

    def step(self) -> tuple[BrainFrame, float, bool]:
        """One 20 ms brain step driven by the loudest onset heard since the last step."""
        with self._lock:
            events = list(self._pending)
            self._pending.clear()
        strength = max((s for _, s, _ in events), default=0.0)
        beat = any(b for _, _, b in events)
        self.beats += beat
        frame = self.brain.step(strength)
        self.timeline.push(frame, strength, beat)
        return frame, strength, beat


class BrainThread(threading.Thread):
    """Steps the pipeline every `dt` of wall-clock time."""

    def __init__(self, pipeline: Pipeline):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.stopped = threading.Event()
        self.late_steps = 0

    def run(self) -> None:
        dt = self.pipeline.brain.dt
        due = time.perf_counter()
        while not self.stopped.is_set():
            self.pipeline.step()
            due += dt
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.2:                        # fell badly behind: resync instead of bursting
                self.late_steps += 1
                due = time.perf_counter()

    def stop(self) -> None:
        self.stopped.set()


@dataclass
class Report:
    seconds: float
    wall_s: float
    beats: int
    bpm: float | None
    n_dn: int
    beat_lock: float           # mean DN activity just after beats / mean activity elsewhere
    level_trace: np.ndarray    # (steps,) level of "all"
    beat_trace: np.ndarray     # (steps,) bool
    top_types: list[tuple[str, int]]


def analyze(pipeline: Pipeline, mono: np.ndarray, sr: int) -> Report:
    """Run a whole recording through the pipeline as fast as the brain allows."""
    hop = int(round(pipeline.brain.dt * sr))
    levels, beats = [], []
    t0 = time.perf_counter()
    for i in range(0, len(mono) - hop + 1, hop):
        pipeline.feed(mono[i:i + hop])
        frame, _, beat = pipeline.step()
        levels.append(frame.level("all"))
        beats.append(beat)
    wall = time.perf_counter() - t0
    levels, beats = np.array(levels), np.array(beats, bool)

    after = np.zeros_like(beats)
    for lag in range(1, 6):                          # 20..100 ms after each detected beat
        after[lag:] |= beats[:-lag]
    lock = float(levels[after].mean() / max(levels[~after].mean(), 1e-6)) if after.any() else float("nan")
    types, counts = np.unique(pipeline.brain.cal.cell_type, return_counts=True)
    top = sorted(zip(types.tolist(), counts.tolist()), key=lambda x: -x[1])[:6]
    return Report(len(levels) * pipeline.brain.dt, wall, int(beats.sum()), pipeline.detector.bpm(),
                  pipeline.brain.n_responsive, lock, levels, beats, top)
