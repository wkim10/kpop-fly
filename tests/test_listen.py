"""LiveDance end to end on a synthetic song: chords (for chroma) over a beat (for onsets), with a
fake recognizer in place of AudD. Checks run synchronously inside feed(), on the mic's own clock."""
import numpy as np
import pytest
import soundfile as sf

from kpop_fly.choreo import Choreo, onset_envelope
from kpop_fly.listen import LiveDance, Ring
from kpop_fly.recognize import AudDError, Recognition
from kpop_fly.retarget import ChoreoTrack

SR = 22050
NOTES = 220.0 * 2 ** (np.arange(24) / 12)


def make_song(seconds: float, seed: int) -> np.ndarray:
    """A chord per 2 s bar (3 random notes) and a kick on every beat at 120 BPM."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for bar in range(int(seconds / 2)):
        span = (t >= bar * 2) & (t < bar * 2 + 2)
        for f in rng.choice(NOTES, 3, replace=False):
            x[span] += 0.15 * np.sin(2 * np.pi * f * t[span])
    kick_t = np.arange(int(0.1 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 25)
    for b in np.arange(0, seconds, 0.5):
        i = int(b * SR)
        x[i:i + len(kick)] += 0.6 * kick[:len(x) - i]
    return (x / np.abs(x).max() * 0.5).astype(np.float32)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """Two saved 'songs' with real audio files, like choreo/."""
    folder = tmp_path_factory.mktemp("choreo")
    tracks, audio = {}, {}
    for slug, title, seed in (("alpha", "ALPHA", 1), ("beta", "BETA", 2)):
        x = make_song(120, seed)
        sf.write(folder / f"{slug}.wav", x, SR)
        T = 3
        c = Choreo(t=np.arange(T, dtype=np.float32), pose=np.zeros((T, 17, 2), np.float32),
                   visible=np.ones((T, 17), np.float32), spread=np.zeros(T, np.float32),
                   n_dancers=np.full(T, 9, np.int16), people=np.full((T, 1, 17, 3), np.nan, np.float16),
                   onset=onset_envelope(x, SR).astype(np.float16),
                   meta={"slug": slug, "artist": "TEST", "title": title, "audio": f"{slug}.wav", "sample_fps": 1.0})
        c.save(folder / f"{slug}.npz")
        tracks[slug] = ChoreoTrack(Choreo.load(folder / f"{slug}.npz"), folder / f"{slug}.npz")
        audio[slug] = x
    return tracks, audio


def play(live: LiveDance, x: np.ndarray, seconds: float, block: int = 1024) -> None:
    for i in range(0, int(seconds * SR), block):
        live.feed(x[i:i + block])


def listener(tracks, recognizer, token="t", **kw) -> LiveDance:
    return LiveDance(SR, tracks, token, recognizer=recognizer, clock=lambda: 0.0, log=lambda *_: None,
                     threaded=False, **kw)


def test_audd_hint_then_exact_lock(library):
    tracks, audio = library
    calls = []

    def audd(clip, sr, token):                 # AudD: right song, position 3 s off (its own recording)
        calls.append(len(clip) / sr)
        return Recognition("TEST", "ALPHA", timecode=40.0 + (live.samples / SR - 10.0) + 3.0)

    live = listener(tracks, audd)
    play(live, audio["alpha"][40 * SR:], 25)    # the "mic" hears alpha from 0:40
    assert live.state == "synced" and live.track is tracks["alpha"]
    assert calls == [10.0] and live.requests == 1                  # one 10 s clip, no further requests
    assert live.song_time() == pytest.approx(40 + 25, abs=0.05)


def test_not_recognized_falls_back_to_the_saved_songs(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: None)
    x = audio["beta"][30 * SR:]
    play(live, x, 11)
    assert live.state == "synced" and live.track is tracks["beta"] and "locally" in live.detail
    play(live, x[11 * SR:], 11)                                     # then it keeps resyncing
    assert live.song_time() == pytest.approx(30 + 22, abs=0.05)


def test_audd_errors_fall_back_too(library):
    tracks, audio = library

    def broken(*a):
        raise AudDError("couldn't reach AudD: offline")

    live = listener(tracks, broken)
    play(live, audio["alpha"][50 * SR:], 14)
    assert live.state == "synced" and live.track is tracks["alpha"] and "offline" in live.detail


def test_no_token_never_calls_audd(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: pytest.fail("AudD called without a token"), token=None)
    play(live, audio["beta"][10 * SR:], 14)
    assert live.state == "synced" and live.requests == 0


def test_known_song_without_a_dance_is_brain_only(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: Recognition("TWICE", "LIKEY", 30.0, ["K-Pop"]))
    play(live, make_song(30, 9), 14)
    assert live.state == "no dance" and live.track is None and live.song_time() is None
    assert "LIKEY (K-pop)" in live.status


def test_unknown_audio_is_not_found(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: None)
    play(live, make_song(30, 9), 14)
    assert live.state == "not found" and live.track is None


def test_silence_is_not_sent_to_audd(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: pytest.fail("AudD called on silence"))
    play(live, np.zeros(20 * SR, np.float32), 20)
    assert live.state == "listening" and live.requests == 0


def test_request_cap(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: None, max_requests=1)
    play(live, make_song(120, 9), 100)                              # unrecognizable for 100 s
    assert live.requests == 1 and "cap" in live.detail


def test_song_change_is_noticed(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: None)
    x = np.concatenate([audio["alpha"][20 * SR:50 * SR], audio["beta"][60 * SR:100 * SR]])
    play(live, x, 30)
    assert live.track is tracks["alpha"]
    play(live, x[30 * SR:], 25)
    assert live.track is tracks["beta"]
    assert live.song_time() == pytest.approx(60 + 25, abs=0.05)


def test_ring_buffer_wraps():
    r = Ring(5)
    r.write(np.arange(3))
    r.write(np.arange(3, 8))
    assert r.last(5).tolist() == [3, 4, 5, 6, 7] and r.last(2).tolist() == [6, 7]
    r.write(np.arange(8, 20))
    assert r.last(5).tolist() == [15, 16, 17, 18, 19]
