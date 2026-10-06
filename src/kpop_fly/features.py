"""Chroma: how much of each of the 12 pitch classes is sounding, 10 times a second.

Onset envelopes pin timing to 20 ms, but through a speaker and a room a 10 s clip's rhythm fits
almost as well one beat or one bar off: pop rhythm repeats. Harmony changes bar to bar and
survives speaker coloration and echo, so live audio is located in two stages: chroma finds the
bar, the onset envelope then finds the exact frame (see listen.py).

Through simulated laptop-speaker-to-room-to-mic audio (band-limited, echo, noise), onsets alone
put 27 of 55 clips at the wrong position; chroma first got all 55 right. Matching clips scored
0.55-0.85, non-matching ones at most 0.22.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .brain import cache_dir

CHROMA_RATE = 10                   # frames per second
N_FFT = 8192
LOW_HZ, HIGH_HZ = 100.0, 4000.0    # where the notes are; below is kick rumble, above is hiss


def chroma(x: np.ndarray, sr: int, rate: int = CHROMA_RATE) -> np.ndarray:
    """(frames, 12) pitch-class energies, log-compressed, unit length per frame."""
    x = np.asarray(x, np.float32)
    hop = int(sr / rate)
    n = 1 + max(0, (len(x) - N_FFT) // hop)
    window = np.hanning(N_FFT).astype(np.float32)
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    band = (f >= LOW_HZ) & (f <= HIGH_HZ)
    pitch_class = (np.round(12 * np.log2(f[band] / 440.0)) % 12).astype(int)
    fold = np.zeros((band.sum(), 12), np.float32)
    fold[np.arange(band.sum()), pitch_class] = 1.0
    out = np.zeros((n, 12), np.float32)
    for i in range(0, n, 256):                                       # STFT in batches of frames
        idx = np.arange(i, min(n, i + 256))[:, None] * hop + np.arange(N_FFT)[None, :]
        frames = x[np.minimum(idx, len(x) - 1)] * window
        out[i:i + len(idx)] = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1))[:, band]) @ fold
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def chroma_ncc(window: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of a (W,12) chroma window against every position of a (N,12)
    reference (each pitch class's mean removed), over all 12 classes at once."""
    W, N = len(window), len(ref)
    if N < W:
        return np.zeros(0)
    w = window - window.mean(0)
    norm = np.sqrt((w ** 2).sum())
    if norm < 1e-9:
        return np.zeros(N - W + 1)
    n = 1 << int(np.ceil(np.log2(N + W)))
    corr = np.fft.irfft((np.fft.rfft(ref, n, axis=0) * np.fft.rfft(w[::-1], n, axis=0)).sum(1), n)[W - 1:N]
    c1 = np.vstack([np.zeros(12), np.cumsum(ref, 0)])
    c2 = np.vstack([np.zeros(12), np.cumsum(ref ** 2, 0)])
    s, s2 = c1[W:] - c1[:-W], c2[W:] - c2[:-W]
    return corr / (norm * np.sqrt(np.maximum((s2 - s * s / W).sum(1), 1e-9)))


def song_chroma(audio: Path) -> np.ndarray:
    """A saved song's chroma, cached in ~/.cache/kpop-fly/chroma (keyed by the file's size and time)."""
    from .audio import load_audio
    st = audio.stat()
    key = hashlib.sha1(f"{audio.resolve()}:{st.st_size}:{st.st_mtime_ns}:{CHROMA_RATE}:{N_FFT}".encode()).hexdigest()[:12]
    path = cache_dir() / "chroma" / f"{audio.stem}-{key}.npy"
    if path.exists():
        return np.load(path)
    samples, sr = load_audio(audio)
    c = chroma(samples.mean(axis=1), sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, c)
    return c
