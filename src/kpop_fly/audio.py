"""Audio sources. Each one calls `on_block(mono_float32)` from its own thread as sound arrives.

Playback sources (file, metronome) analyze exactly the block they hand to the speakers, so
what the fly hears stays in sync with what you hear. `mute=True` paces playback with a timer
instead of a sound device, which is handy for headless runs.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

BLOCK = 512
OnBlock = Callable[[np.ndarray], None]


def click_track(bpm: float, sr: int = 44100, bars: int = 1) -> np.ndarray:
    """A 4/4 loop: kick + click on every beat, accented on the one. Shape (frames, 1)."""
    beat = int(round(sr * 60 / bpm))
    out = np.zeros(beat * 4 * bars, np.float32)
    t = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 25)
    click = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 90)
    for i in range(4 * bars):
        hit = (0.9 if i % 4 == 0 else 0.6) * kick + 0.3 * click
        out[i * beat:i * beat + len(t)] += hit[:len(out) - i * beat]
    return out[:, None]


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode wav/flac/ogg/mp3 with libsndfile. Returns (frames, channels) float32 and sample rate."""
    import soundfile as sf
    try:
        samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        raise SystemExit(f"can't decode {path}: {e}\n(convert it first, e.g. `ffmpeg -i in.m4a out.wav`)")
    return samples, sr


class PlaybackSource:
    def __init__(self, samples: np.ndarray, sr: int, name: str, loop: bool = False):
        self.samples, self.sr, self.name, self.loop = samples, sr, name, loop
        self.finished = threading.Event()
        self._pos = 0
        self._stream = None
        self._thread: threading.Thread | None = None

    @classmethod
    def file(cls, path: str | Path) -> PlaybackSource:
        samples, sr = load_audio(path)
        return cls(samples, sr, Path(path).name)

    @classmethod
    def metronome(cls, bpm: float, sr: int = 44100) -> PlaybackSource:
        return cls(click_track(bpm, sr), sr, f"metronome {bpm:g} BPM", loop=True)

    def _next(self, frames: int) -> np.ndarray:
        if self.loop:
            idx = (self._pos + np.arange(frames)) % len(self.samples)
            chunk = self.samples[idx]
        else:
            chunk = self.samples[self._pos:self._pos + frames]
        self._pos += frames
        return chunk

    def start(self, on_block: OnBlock, mute: bool = False) -> None:
        if mute:
            self._thread = threading.Thread(target=self._paced, args=(on_block,), daemon=True)
            self._thread.start()
            return
        import sounddevice as sd

        def callback(outdata, frames, _time, _status):
            chunk = self._next(frames)
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            if len(chunk):
                on_block(chunk.mean(axis=1))
            if len(chunk) < frames:
                raise sd.CallbackStop

        self._stream = sd.OutputStream(samplerate=self.sr, channels=self.samples.shape[1], blocksize=BLOCK,
                                       dtype="float32", callback=callback, finished_callback=self.finished.set)
        self._stream.start()

    def _paced(self, on_block: OnBlock) -> None:
        start = time.perf_counter()
        while not self.finished.is_set():
            chunk = self._next(BLOCK)
            if not len(chunk):
                break
            on_block(chunk.mean(axis=1))
            delay = start + self._pos / self.sr - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self.finished.set()

    def stop(self) -> None:
        self.finished.set()
        if self._stream is not None:
            self._stream.close()


class MicSource:
    def __init__(self, device: int | str | None = None):
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        self.device, self.sr, self.name = device, int(info["default_samplerate"]), f"mic: {info['name']}"
        self.finished = threading.Event()
        self._stream = None

    def start(self, on_block: OnBlock, mute: bool = False) -> None:
        import sounddevice as sd
        self._stream = sd.InputStream(device=self.device, samplerate=self.sr, channels=1, blocksize=BLOCK,
                                      dtype="float32", callback=lambda data, *_: on_block(data[:, 0].copy()))
        self._stream.start()

    def stop(self) -> None:
        self.finished.set()
        if self._stream is not None:
            self._stream.close()
