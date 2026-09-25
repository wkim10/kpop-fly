import numpy as np
import pytest

from kpop_fly.audio import click_track
from kpop_fly.beat import OnsetDetector

SR = 44100


def run(samples: np.ndarray) -> tuple[OnsetDetector, list]:
    det = OnsetDetector(SR)
    events = []
    for i in range(0, len(samples), 512):                     # the block size the audio thread uses
        events += det.process(samples[i:i + 512])
    return det, events


@pytest.mark.parametrize("bpm", [95, 120, 140])
def test_click_track_tempo(bpm):
    det, events = run(np.tile(click_track(bpm, SR)[:, 0], 4))  # 16 beats
    beats = [t for t, _, b in events if b]
    assert 14 <= len(beats) <= 16
    assert det.bpm() == pytest.approx(bpm, rel=0.03)


def test_silence_is_silent():
    det, events = run(np.zeros(SR * 3, np.float32))
    assert all(s == 0 and not b for _, s, b in events)
    assert det.bpm() is None


def test_quiet_noise_is_gated():
    noise = np.random.default_rng(0).normal(scale=1e-4, size=SR * 3).astype(np.float32)
    _, events = run(noise)
    assert not any(b for _, _, b in events)


def test_strength_is_normalized():
    _, events = run(np.tile(click_track(120, SR)[:, 0], 2))
    strengths = np.array([s for _, s, _ in events])
    assert strengths.min() >= 0 and strengths.max() == pytest.approx(1.0)
