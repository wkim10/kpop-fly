import numpy as np
import pytest

from kpop_fly.choreo import ONSET_RATE, Choreo
from kpop_fly.match import Match, Segment, align, by_video_id, match, sliding_ncc

R = ONSET_RATE
HOP_TOL = 6.0                     # segment edges are only known to about a window hop + half a window


def song(seconds: float, seed: int) -> np.ndarray:
    """A synthetic onset envelope: a groove plus random accents, like a pop song's."""
    rng = np.random.default_rng(seed)
    n = int(seconds * R)
    env = np.zeros(n)
    beat = int(R * 60 / rng.uniform(95, 130))
    env[::beat] = 1.0
    accents = rng.random(n) < 0.08
    env[accents] = rng.uniform(0.3, 0.9, accents.sum())
    return env


def with_chorus(seconds: float, seed: int) -> np.ndarray:
    """Verse / chorus / verse / chorus / bridge / chorus: the chorus repeats exactly."""
    chorus = song(20, seed + 100)
    parts = [song(25, seed), chorus, song(25, seed + 1), chorus, song(15, seed + 2), chorus]
    return np.concatenate(parts)[:int(seconds * R)]


def choreo(env: np.ndarray, video_id: str = "abcdefghijk") -> Choreo:
    T = 3
    return Choreo(t=np.arange(T, dtype=np.float32), pose=np.zeros((T, 17, 2), np.float32),
                  visible=np.ones((T, 17), np.float32), spread=np.zeros(T, np.float32),
                  n_dancers=np.full(T, 9, np.int16), people=np.full((T, 1, 17, 3), np.nan, np.float16),
                  onset=env.astype(np.float16), meta={"slug": "x", "source": {"id": video_id}})


def noisy(env: np.ndarray, seed: int = 9, jitter: float = 0.15) -> np.ndarray:
    """Another recording of the same song: different mix, some onsets louder or softer."""
    rng = np.random.default_rng(seed)
    return np.clip(env * rng.uniform(0.6, 1.2, len(env)) + jitter * rng.random(len(env)) * (rng.random(len(env)) < 0.05), 0, 1)


def test_sliding_ncc_finds_an_exact_copy():
    ref = song(60, 1)
    c = sliding_ncc(ref[1000:1500], ref)
    assert int(c.argmax()) == 1000 and c.max() == pytest.approx(1.0)


def test_music_video_with_an_intro_maps_onto_the_song():
    ref = with_chorus(130, 2)
    video = np.concatenate([song(28, 50) * 0.3, noisy(ref)])            # 28 s drama intro, then the song
    segments = align(video, ref)
    assert len(segments) == 1
    s = segments[0]
    assert s.offset == pytest.approx(-28.0, abs=0.1)
    assert s.start == pytest.approx(28.0, abs=HOP_TOL) and s.end == pytest.approx(len(video) / R, abs=HOP_TOL)



def test_repeated_choruses_dont_make_the_path_jump():
    ref = with_chorus(130, 3)
    segments = align(noisy(ref, seed=4), ref)
    assert len(segments) == 1 and segments[0].offset == pytest.approx(0.0, abs=0.1)


def test_a_different_song_does_not_match():
    choreos = {"a": choreo(with_chorus(130, 5)), "b": choreo(with_chorus(130, 6))}
    best, candidates = match(with_chorus(130, 7), choreos)
    assert best is None and all(not c.ok() for c in candidates)


def test_the_right_song_wins_among_several():
    a, b = with_chorus(130, 8), with_chorus(130, 9)
    best, _ = match(noisy(b), {"a": choreo(a), "b": choreo(b)})
    assert best is not None and best.slug == "b" and best.cover > 0.9


def test_time_map():
    m = Match("x", [Segment(10, 50, -8.0, 0.7), Segment(60, 90, 30.0, 0.6)], duration=100, how="audio")
    assert m.song_time(5) is None and m.song_time(55) is None
    assert m.song_time(20) == 12.0 and m.song_time(70) == 100.0
    assert m.cover == pytest.approx(0.7) and m.longest == 40
    assert m.ok()
    assert not Match("x", [Segment(0, 20, 0, 0.7)], duration=200, how="audio").ok()      # too little of it
    assert Match("x", [Segment(0, 20, 0, 0.7)], duration=30, how="audio").ok()           # a short clip can match


def test_video_id_shortcut():
    m = by_video_id("abcdefghijk", {"x": choreo(song(10, 1))}, duration=100)
    assert m is not None and m.how == "video id" and m.song_time(42.0) == 42.0
    assert by_video_id("zzzzzzzzzzz", {"x": choreo(song(10, 1))}, duration=100) is None
