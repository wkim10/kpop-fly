"""Onset detection: raw audio in, a 0..1 "how hard did the sound just hit" signal out.

Spectral flux (the summed increase in log-magnitude between successive FFT frames) is
normalized against a rolling window of its own history, so loud and quiet songs both use
the full 0..1 range. That strength is what the fly's antennae feel; beats (upward threshold
crossings) are only used for display and the BPM readout.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class OnsetDetector:
    def __init__(self, sr: int, frame: int = 1024, hop: int = 512, history_s: float = 4.0,
                 gate_db: float = -55.0, beat_threshold: float = 0.5, refractory_s: float = 0.25):
        self.sr, self.frame, self.hop = sr, frame, hop
        self.window = np.hanning(frame).astype(np.float32)
        self.gate = 10 ** (gate_db / 20)
        self.beat_threshold = beat_threshold
        self.refractory_s = refractory_s
        self.t = 0.0                                   # seconds of audio consumed, at hop resolution
        self._buf = np.zeros(0, np.float32)
        self._prev_mag: np.ndarray | None = None
        self._flux = deque(maxlen=max(16, int(history_s * sr / hop)))
        self._prev_strength = 0.0
        self._last_beat = -np.inf
        self._beats = deque(maxlen=17)

    def process(self, block: np.ndarray) -> list[tuple[float, float, bool]]:
        """Feed mono samples. Returns (time_s, strength, is_beat) for every hop completed."""
        self._buf = np.concatenate([self._buf, np.asarray(block, np.float32).ravel()])
        out = []
        while len(self._buf) >= self.frame:
            frame = self._buf[:self.frame]
            self._buf = self._buf[self.hop:]
            self.t += self.hop / self.sr
            out.append(self._analyze(frame))
        return out

    def _analyze(self, frame: np.ndarray) -> tuple[float, float, bool]:
        mag = np.log1p(100.0 * np.abs(np.fft.rfft(frame * self.window)))
        flux = 0.0 if self._prev_mag is None else float(np.maximum(mag - self._prev_mag, 0.0).mean())
        self._prev_mag = mag
        self._flux.append(flux)

        strength = 0.0
        if np.sqrt(np.mean(frame ** 2)) >= self.gate and len(self._flux) >= 8:
            hist = np.fromiter(self._flux, float)
            med, top = np.median(hist), np.percentile(hist, 98)
            strength = float(np.clip((flux - med) / max(top - med, 1e-3), 0.0, 1.0))

        is_beat = (strength >= self.beat_threshold > self._prev_strength
                   and self.t - self._last_beat >= self.refractory_s)
        if is_beat:
            self._last_beat = self.t
            self._beats.append(self.t)
        self._prev_strength = strength
        return self.t, strength, is_beat

    def bpm(self) -> float | None:
        """Tempo from the median interval of recent beats, folded into 70..180 BPM."""
        if len(self._beats) < 5:
            return None
        gaps = np.diff(np.fromiter(self._beats, float))
        gaps = gaps[(gaps > 0.25) & (gaps < 2.0)]
        if len(gaps) < 4:
            return None
        bpm = 60.0 / float(np.median(gaps))
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        return bpm
