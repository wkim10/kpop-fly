"""Loads the real connectome (~260 MB in ~/fly-data, calibration cached in ~/.cache/kpop-fly)."""
import numpy as np
import pytest

from kpop_fly.audio import click_track
from kpop_fly.brain import CHANNELS, ListeningBrain
from kpop_fly.engine import Pipeline, analyze

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def brain():
    return ListeningBrain(log=lambda *_: None)


def test_calibration_finds_listening_descending_neurons(brain):
    cal = brain.cal
    assert len(cal.dn) > 20
    assert set(cal.tier.tolist()) == {0, 1, 2}
    assert (cal.peak > cal.base).all()
    assert cal.members.shape == (len(CHANNELS), len(cal.dn))


def test_silence_keeps_the_fly_still(brain):
    brain.brain.reset(seed=1)
    levels = np.array([brain.step(0.0).level("all") for _ in range(100)])
    assert levels[20:].mean() < 0.1


def test_brain_locks_to_the_beat(brain):
    brain.brain.reset(seed=1)
    pipe = Pipeline(brain, 44100)
    report = analyze(pipe, np.tile(click_track(120)[:, 0], 3), 44100)   # 6 s
    assert report.beats >= 10
    assert report.beat_lock > 2.0
    assert report.seconds / report.wall_s > 1.0                          # faster than real time
