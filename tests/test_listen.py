"""LiveDance end to end on a synthetic song: chords (for chroma) over a beat (for onsets), with a
fake recognizer in place of AudD. Checks run synchronously inside feed(), on the mic's own clock."""
import numpy as np
import pytest
import soundfile as sf

from kpop_fly.choreo import Choreo, onset_envelope
from kpop_fly.listen import LISTEN_S, STOP_AFTER_S, LiveDance, Ring
from kpop_fly.recognize import AudDError, Recognition
from kpop_fly.retarget import ChoreoTrack

SR = 22050
NOTES = 220.0 * 2 ** (np.arange(24) / 12)


def make_song(seconds: float, seed: int) -> np.ndarray:
    """Like a pop song: its own key and tempo, a chord from that key per bar, a kick on every beat.
    (Random chords from too small a palette repeat by chance, which real songs rarely do.)"""
    rng = np.random.default_rng(seed)
    root = 220.0 * 2 ** (rng.integers(12) / 12)
    scale = root * 2 ** (np.array([0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]) / 12)
    beat = 60 / rng.uniform(95, 140)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for bar in range(int(seconds / (4 * beat)) + 1):
        span = (t >= bar * 4 * beat) & (t < (bar + 1) * 4 * beat)
        degree = rng.integers(7)
        for f in scale[[degree, degree + 2, degree + 4, rng.integers(14)]]:   # a triad plus a colour note
            x[span] += 0.15 * np.sin(2 * np.pi * f * t[span])
    kick_t = np.arange(int(0.1 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 25)
    for b in np.arange(0, seconds, beat):
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

    def audd(clip, sr, token):                 # AudD's timecode: ~2.3 s before the clip's end, whole seconds
        calls.append(len(clip) / sr)
        return Recognition("TEST", "ALPHA", timecode=float(np.floor(40.0 + live.samples / SR - 2.3)))

    live = listener(tracks, audd)
    play(live, audio["alpha"][40 * SR:], 7)     # the "mic" hears alpha from 0:40
    assert calls == [LISTEN_S]                  # recognized from a LISTEN_S clip, not 10 s
    assert live.state == "synced" and live.track is tracks["alpha"]
    play(live, audio["alpha"][47 * SR:], 18)
    assert live.requests == 1                   # no further requests while it stays in sync
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
    play(live, audio["alpha"][50 * SR:], 7)
    assert live.state == "synced" and live.track is tracks["alpha"] and "offline" in live.detail


def test_no_token_never_calls_audd(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: pytest.fail("AudD called without a token"), token=None)
    play(live, audio["beta"][10 * SR:], 14)
    assert live.state == "synced" and live.requests == 0


def test_known_song_without_a_dance_is_brain_only(library):
    tracks, audio = library
    live = listener(tracks, lambda *a: Recognition("TWICE", "LIKEY", 30.0, ["K-Pop"]))
    play(live, make_song(30, 7), 14)
    assert live.state == "no dance" and live.track is None and live.song_time() is None
    assert "LIKEY (K-pop)" in live.status


def test_unknown_audio_is_not_found(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: None)
    play(live, make_song(30, 7), 14)
    assert live.state == "not found" and live.track is None


def test_silence_is_not_sent_to_audd(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: pytest.fail("AudD called on silence"))
    play(live, np.zeros(20 * SR, np.float32), 20)
    assert live.state == "listening" and live.requests == 0


def test_request_cap(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: None, max_requests=1)
    play(live, make_song(120, 7), 100)                              # unrecognizable for 100 s
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


def at_db(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20) / np.sqrt(np.mean(x ** 2) + 1e-20)).astype(np.float32)


def noise(seconds: float, db: float) -> np.ndarray:
    return at_db(np.random.default_rng(0).normal(size=int(seconds * SR)), db)


def time_to(live: LiveDance, x: np.ndarray, condition, block: int = 1024) -> float | None:
    """Feed x until condition(live) holds; seconds it took (None: never)."""
    for i in range(0, len(x), block):
        live.feed(x[i:i + block])
        if condition(live):
            return (i + block) / SR
    return None


@pytest.mark.parametrize("room_db", [-120, -72, -62])
def test_the_fly_stops_dancing_when_the_song_stops(library, room_db):
    """-62: room noise above the -65 silence line, only 7 dB under the song. The level alone can't tell."""
    tracks, audio = library
    live = listener(tracks, lambda *a: pytest.fail("AudD asked"), token=None)
    play(live, at_db(audio["alpha"][30 * SR:55 * SR], -55), 25)
    assert live.state == "synced"
    took = time_to(live, noise(15, room_db), lambda lv: lv.song_time() is None)
    assert took is not None and took <= 4.0                     # "more than a few seconds" of quiet
    assert live.state == "stopped" and live.track is None and not live.input_open
    play(live, noise(10, room_db), 10)
    assert live.state == "stopped"                              # stays stopped through the quiet


def test_a_paused_song_resumes_without_asking_audd(library):
    tracks, audio = library
    calls = []
    live = listener(tracks, lambda *a: calls.append(1) or Recognition("TEST", "ALPHA",
                    timecode=float(np.floor(30.0 + live.samples / SR - 2.3))))
    x = audio["alpha"]
    play(live, x[30 * SR:50 * SR], 20)                          # 0:30-0:50, then paused for 8 s
    assert calls == [1] and live.state == "synced"
    play(live, noise(8, -72), 8)
    assert live.state == "stopped"
    took = time_to(live, x[50 * SR:80 * SR], lambda lv: lv.state == "synced")   # resumed where it paused
    assert took is not None and took <= 6.0
    assert calls == [1] and "resumed" in live.detail
    assert live.song_time() == pytest.approx(50 + took, abs=0.05)


def test_a_different_song_is_recognized_from_after_the_change(library):
    tracks, audio = library
    clips = []

    def audd(clip, sr, token):
        clips.append(clip.copy())
        return None                                             # make it use the local search

    live = listener(tracks, audd)
    play(live, audio["alpha"][20 * SR:45 * SR], 25)
    assert live.track is tracks["alpha"]
    took = time_to(live, audio["beta"][60 * SR:100 * SR], lambda lv: lv.track is tracks["beta"])
    assert took is not None and took <= STOP_AFTER_S + LISTEN_S + 2.0
    assert live.input_open                                      # it never stopped: music kept playing
    assert live.song_time() == pytest.approx(60 + took, abs=0.05)


def test_quiet_before_any_song_is_just_listening(library):
    tracks, _ = library
    live = listener(tracks, lambda *a: pytest.fail("AudD asked"))
    play(live, noise(10, -80), 10)
    assert live.state == "listening" and live.input_open


def test_a_skip_within_the_song_is_followed_without_asking_audd(library):
    tracks, audio = library
    calls = []
    live = listener(tracks, lambda *a: calls.append(1) or Recognition("TEST", "ALPHA",
                    timecode=float(np.floor(20.0 + live.samples / SR - 2.3))))
    x = audio["alpha"]
    play(live, x[20 * SR:40 * SR], 20)
    assert calls == [1] and live.state == "synced"
    took = time_to(live, x[80 * SR:110 * SR], lambda lv: lv.state == "synced" and "again" in lv.detail)
    assert took is not None and took <= 6.0
    assert calls == [1] and live.input_open                    # no new request; never stopped
    assert live.song_time() == pytest.approx(80 + took, abs=0.05)
