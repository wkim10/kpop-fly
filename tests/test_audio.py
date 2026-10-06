import numpy as np
import pytest

from kpop_fly.audio import InputConditioner, click_track

SR = 44100


def at_db(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20) / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def run(cond: InputConditioner, x: np.ndarray, block: int = 512) -> np.ndarray:
    return np.concatenate([cond(x[i:i + block]) for i in range(0, len(x), block)])


def db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def music(seconds: float) -> np.ndarray:
    """A beat over a steady bed of sound, like a song (a bare click track is mostly silence)."""
    beat = np.tile(click_track(120, SR)[:, 0], int(seconds / 2) + 1)[:int(seconds * SR)]
    bed = np.random.default_rng(1).normal(size=len(beat)).astype(np.float32)
    return beat + 0.3 * bed


def test_quiet_music_is_brought_up_to_a_normal_level():
    cond = InputConditioner(SR, silence_db=-65)
    out = run(cond, at_db(music(10), -55))
    assert cond.level_db == pytest.approx(-55, abs=2)                # the meter shows the raw level
    assert cond.gain_db == pytest.approx(35, abs=2)
    assert db(out[-3 * SR:]) == pytest.approx(InputConditioner.TARGET_DB, abs=2)


def test_loud_music_is_left_alone():
    cond = InputConditioner(SR)
    x = at_db(music(6), -15)
    out = run(cond, x)
    assert cond.gain_db == 0.0
    np.testing.assert_allclose(out[-SR:], x[-SR:], atol=1e-6)


def test_silence_is_gated_and_does_not_wind_up_the_gain():
    cond = InputConditioner(SR, silence_db=-65)
    run(cond, at_db(music(8), -50))
    gain = cond.gain_db
    noise = at_db(np.random.default_rng(0).normal(size=5 * SR).astype(np.float32), -80)
    out = run(cond, noise)
    assert not cond.loud
    assert cond.gain_db == pytest.approx(gain)                        # held through the silence
    assert db(out[-SR:]) < -90                                        # amplified, the noise would be ~-50 dBFS


def test_output_never_clips_past_full_scale():
    cond = InputConditioner(SR)
    run(cond, at_db(music(6), -60))                                  # gain winds up to +40
    out = cond(np.ones(512, np.float32) * 0.5)                       # then something loud
    assert np.abs(out).max() <= 1.0


def test_threshold_is_adjustable():
    x = at_db(music(4), -58)
    assert not _ends_loud(InputConditioner(SR, silence_db=-50), x)
    assert _ends_loud(InputConditioner(SR, silence_db=-65), x)


def _ends_loud(cond, x):
    run(cond, x)
    return cond.loud
