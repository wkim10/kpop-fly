"""Live listening: microphone audio -> which song, and exactly where in it, kept up to date.

  listening    collecting ~10 s of sound (silence doesn't count)
  recognizing  the clip is with AudD (on a worker thread: the window keeps running)
  synced       a saved dance is playing and the clip was located in it; the fly dances it
  approximate  a saved dance, but the clip couldn't confirm AudD's position: dance from AudD's
               timecode while retrying
  no dance     AudD knows the song, but there's no saved dance for it: brain only
  not found    neither AudD nor a search of the saved songs recognized it: brain only

Locating a clip in a song takes two stages: chroma (harmony) finds the bar, then the onset
envelope finds the exact 20 ms frame. Through a speaker and a room, rhythm alone fits one beat or
one bar off about half the time (features.py has the numbers).

AudD's timecode is a position in *its* recording of the song, which can sit a few seconds off
the dance practice video's audio, and it's whole seconds. So it's only a hint: the clip is
located within +-LOCK_SPAN_S of it, narrow enough to rule out the song's other choruses.

When AudD doesn't know the recording (a live stage with live vocals, say), or there's no token or
no network, every saved song is searched end to end instead, using as much recent audio as there
is (up to LONG_S). Without AudD's timecode nothing tells a song's repeated sections apart, so a
local lock can land on another chorus (in tests, about 1 time in 10). Usually the choreography
repeats too; the resyncs keep it there until the section ends, then it finds its place again.
(Re-checking the whole song once more audio had been heard was tried, and removed: in a live
recording a later chorus can match the practice video's chorus better than the true one, so the
check moved correct locks to wrong ones.)

After a lock, every RESYNC_S the latest audio is re-located around where the song should be by
now. That costs no API requests; AudD is asked again only when it fails LOSE_AFTER times running
(the song changed, or someone skipped ahead).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .choreo import ONSET_RATE, Choreo, onset_envelope
from .features import CHROMA_RATE, chroma, chroma_ncc, song_chroma
from .match import sliding_ncc, smooth
from .recognize import TOKEN_ENV, AudDError, Recognition, recognize, saved_dance
from .retarget import ChoreoTrack

CLIP_S = 10.0             # what AudD hears, and the onset alignment window
LONG_S = 18.0             # the most audio a whole-song search uses (repeats need context)
LEAD_S = 2.0              # extra audio before a window so the onset detector is warmed up
BUFFER_S = LONG_S + LEAD_S + 2.0
SILENCE_DB = -65.0        # quieter than this (dBFS, averaged over LOUDNESS_S) is silence
LOUDNESS_S = 0.5
LOCK_SPAN_S = 20.0        # search +-this around AudD's timecode
RESYNC_SPAN_S = 3.0       # ...and +-this around the predicted position afterwards
FINE_SPAN_S = 0.2         # the onset stage searches +-this around the chroma position
LOCK_SCORE = 0.40         # chroma alignment needed to trust a position (room audio: real 0.55+, others <= 0.22)
LOCAL_SCORE = 0.35        # to say "it's this saved song" from a whole-song search (room audio: real 0.38+,
LOCAL_MARGIN = 0.15       # others <= 0.27), when it also beats the next saved song by this (real 0.20+, others <= 0.09)
LOCAL_SCORE_ALONE = 0.45  # with only one saved song there's no runner-up to beat: be stricter
RESYNC_S = 6.0
RETRY_MISS_S = 2.0        # after a failed resync, look again this soon
LOSE_AFTER = 3
RETRY_NOT_FOUND_S = 30.0  # wait before asking AudD again about an unrecognized song
RETRY_NO_DANCE_S = 60.0   # ...or about a recognized song with no saved dance (has the song changed?)
RETRY_LOCAL_S = 10.0      # without AudD, looking again only costs a little CPU


class Ring:
    """The last n samples of a stream."""

    def __init__(self, n: int):
        self.buf = np.zeros(n, np.float32)
        self.pos = 0
        self.filled = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).ravel()[-len(self.buf):]
        end = self.pos + len(x)
        if end <= len(self.buf):
            self.buf[self.pos:end] = x
        else:
            k = len(self.buf) - self.pos
            self.buf[self.pos:], self.buf[:end - len(self.buf)] = x[:k], x[k:]
        self.pos = end % len(self.buf)
        self.filled = min(len(self.buf), self.filled + len(x))

    def last(self, n: int) -> np.ndarray:
        n = min(n, self.filled)
        return np.roll(self.buf, -self.pos)[len(self.buf) - n:]


def _peak(scores: np.ndarray, rate: float, around: float, span: float) -> tuple[float | None, float]:
    lo, hi = max(0, int(round((around - span) * rate))), min(len(scores), int(round((around + span) * rate)) + 1)
    if lo >= hi:
        return None, 0.0
    j = lo + int(scores[lo:hi].argmax())
    return j / rate, float(scores[j])


def locate(audio: np.ndarray, sr: int, onset_ref: np.ndarray, chroma_ref: np.ndarray, around: float, span: float,
           window_s: float = CLIP_S) -> tuple[float | None, float]:
    """The song time at the END of `audio`, searched for within +-span of `around`, and the chroma
    score. Chroma uses the last `window_s` of the audio, the onset refinement the last CLIP_S;
    `audio` should hold LEAD_S more than the longer of the two."""
    start, score = _peak(chroma_ncc(chroma(audio[-int(window_s * sr):], sr), chroma_ref), CHROMA_RATE,
                         around - window_s, span)
    if start is None:
        return None, 0.0
    end = start + window_s
    env = onset_envelope(audio, sr)[-int(CLIP_S * ONSET_RATE):]
    fine, _ = _peak(sliding_ncc(smooth(env.astype(float)), smooth(onset_ref.astype(float))), ONSET_RATE,
                    end - CLIP_S, FINE_SPAN_S)
    return (fine + CLIP_S if fine is not None else end), score


class LiveDance:
    """Feed it audio blocks; read `track`, `song_time()` and `status` from the display."""

    def __init__(self, sr: int, tracks: dict[str, ChoreoTrack], api_token: str | None, max_requests: int = 50,
                 recognizer: Callable[[np.ndarray, int, str], Recognition | None] = recognize,
                 clock: Callable[[], float] = time.perf_counter, log: Callable[[str], None] = print,
                 threaded: bool = True, silence_db: float = SILENCE_DB,
                 level_db: Callable[[], float] | None = None):
        self.sr = sr
        self.tracks = tracks
        self.songs = {slug: t.choreo.meta for slug, t in tracks.items()}
        self.token = api_token
        self.max_requests = max_requests
        self.recognizer = recognizer
        self.clock = clock
        self.log = log
        self.threaded = threaded                 # False: checks run inside feed() (deterministic, for tests)
        self.silence_db = silence_db
        self.level_db = level_db                 # the raw input level, if the audio we're fed has been gained
        self.ring = Ring(int(BUFFER_S * sr))
        self.samples = 0                         # mic samples received: the mic clock
        self._stamp = (0, clock())               # (samples, wall time) at the last block
        self._energy = 0.0                       # mean square, smoothed over LOUDNESS_S
        self.loud_s = 0.0                        # seconds of sound in a row
        self.quiet_s = 0.0                       # seconds of silence in a row
        self.slug: str | None = None
        self.offset: float | None = None         # song time = mic time + offset
        self.state = "listening"
        self.detail = "" if api_token else f"no {TOKEN_ENV}: only the saved songs can be recognized"
        self.recognition: Recognition | None = None
        self.requests = 0
        self.misses = 0
        self.next_check = 0.0                    # mic time of the next check
        self._busy = False
        self._lock = threading.Lock()
        self._chroma: dict[str, np.ndarray] = {}

    # ---- the audio thread --------------------------------------------------------------------

    def feed(self, mono: np.ndarray) -> None:
        mono = np.asarray(mono, np.float32)
        self.ring.write(mono)
        self.samples += len(mono)
        self._stamp = (self.samples, self.clock())
        dur = len(mono) / self.sr
        a = min(1.0, dur / LOUDNESS_S)
        self._energy += a * (float(np.mean(mono ** 2)) - self._energy)
        level = self.level_db() if self.level_db else 10 * np.log10(self._energy + 1e-12)
        loud = level > self.silence_db
        self.loud_s, self.quiet_s = (self.loud_s + dur, 0.0) if loud else (0.0, self.quiet_s + dur)
        if not self._busy and self.mic_time() >= self.next_check:
            self._busy = True
            if self.threaded:
                threading.Thread(target=self._check, daemon=True).start()
            else:
                self._check()

    # ---- the display thread ------------------------------------------------------------------

    @property
    def track(self) -> ChoreoTrack | None:
        return self.tracks.get(self.slug) if self.slug else None

    def mic_time(self) -> float:
        samples, at = self._stamp
        return samples / self.sr + (self.clock() - at)

    def song_time(self) -> float | None:
        with self._lock:
            if self.slug is None or self.offset is None:
                return None
            return self.mic_time() + self.offset

    @property
    def status(self) -> str:
        quota = f"   AudD requests: {self.requests}" if self.requests else ""
        return f"{self.state}{': ' + self.detail if self.detail else ''}{quota}"

    # ---- the worker --------------------------------------------------------------------------

    def _set(self, state: str, detail: str = "", slug: str | None = None, offset: float | None = None,
             wait: float = RESYNC_S) -> None:
        with self._lock:
            changed = (state, detail) != (self.state, self.detail)
            self.state, self.detail, self.slug, self.offset = state, detail, slug, offset
            self.next_check = self.mic_time() + wait
        if changed:
            self.log(f"[{self.mic_time():6.1f}s] {self.status}")

    def _check(self) -> None:
        try:
            if self.quiet_s >= 3.0:                                   # music stopped
                if self.state != "listening" or self.detail != "quiet":
                    self.misses = 0
                    self._set("listening", "quiet", wait=1.0)
                else:
                    self.next_check = self.mic_time() + 1.0
            elif self.slug is not None:
                self._resync()
            elif self.loud_s >= CLIP_S:
                self._recognize()
            else:
                self.next_check = self.mic_time() + 1.0
        except Exception as e:                                       # never take the dance down with us
            self._set("error", f"{type(e).__name__}: {e}", wait=RETRY_NOT_FOUND_S)
        finally:
            self._busy = False

    def _audio(self, window_s: float = CLIP_S) -> tuple[np.ndarray, float]:
        """The last window_s of audio (plus LEAD_S before it), and the mic time of its last sample."""
        return self.ring.last(int((window_s + LEAD_S) * self.sr)), self.samples / self.sr

    def _locate(self, slug: str, audio: np.ndarray, around: float, span: float,
                window_s: float = CLIP_S) -> tuple[float | None, float]:
        if slug not in self._chroma:
            self._chroma[slug] = song_chroma(self.tracks[slug].audio)
        return locate(audio, self.sr, self.tracks[slug].choreo.onset, self._chroma[slug], around, span, window_s)

    def _whole_song(self, slug: str, audio: np.ndarray, window_s: float) -> tuple[float | None, float]:
        if slug not in self._chroma:
            self._chroma[slug] = song_chroma(self.tracks[slug].audio)
        duration = len(self._chroma[slug]) / CHROMA_RATE              # the audio's length, not the dance's
        return self._locate(slug, audio, duration / 2, duration / 2 + window_s, window_s)

    def _name(self) -> str:
        if self.recognition:
            return self.recognition.name
        return self.tracks[self.slug].title if self.slug else ""

    def _recognize(self) -> None:
        audio, end = self._audio(CLIP_S)
        rec, why, asked = None, "", False
        if not self.token:
            why = f"no {TOKEN_ENV}"
        elif self.requests >= self.max_requests:
            why = f"AudD request cap reached ({self.max_requests}, see --max-requests)"
        else:
            self._set("recognizing", "", wait=1e9)
            self.requests += 1
            asked = True
            try:
                rec = self.recognizer(audio[-int(CLIP_S * self.sr):], self.sr, self.token)
                why = "" if rec else "AudD didn't recognize this"
            except AudDError as e:
                why = str(e)
        self.recognition = rec
        self.misses = 0

        if rec is not None:
            slug = saved_dance(rec, self.songs)
            if slug is None:
                kpop = " (K-pop)" if rec.is_kpop else ""
                self._set("no dance", f"{rec.name}{kpop}: no saved dance", wait=RETRY_NO_DANCE_S)
                return
            if rec.timecode is None:
                at, score = self._whole_song(slug, audio, CLIP_S)
            else:                                                    # AudD's timecode is the clip's start
                at, score = self._locate(slug, audio, rec.timecode + CLIP_S, LOCK_SPAN_S)
            if at is not None and score >= LOCK_SCORE:
                self._set("synced", f"{rec.name} at {_clock(at)} (alignment {score:.2f})", slug, at - end)
            elif rec.timecode is not None:
                self._set("approximate", f"{rec.name}: dancing from AudD's position", slug,
                          rec.timecode + CLIP_S - end, wait=RETRY_MISS_S)
            else:
                self._set("no dance", f"{rec.name}: couldn't find the position", wait=RETRY_NOT_FOUND_S)
            return

        # AudD couldn't say (live vocals, no network, no token...): is it one of our own songs?
        window = min(LONG_S, self.loud_s)
        audio, end = self._audio(window)
        found = sorted(((score, slug, at) for slug in self.tracks
                        for at, score in [self._whole_song(slug, audio, window)] if at is not None), reverse=True)
        score, slug, at = found[0] if found else (0.0, None, None)
        if len(found) >= 2:
            ok = score >= LOCAL_SCORE and score - found[1][0] >= LOCAL_MARGIN
        else:
            ok = score >= LOCAL_SCORE_ALONE
        if slug is not None and ok:
            self._set("synced", f"{self.tracks[slug].title} at {_clock(at)} (recognized locally: {why})", slug,
                      at - end)
        else:
            self._set("not found", f"{why}; not one of the saved songs either",
                      wait=RETRY_NOT_FOUND_S if asked else RETRY_LOCAL_S)

    def _resync(self) -> None:
        slug, offset = self.slug, self.offset
        audio, end = self._audio(CLIP_S)
        span = RESYNC_SPAN_S if self.state == "synced" else LOCK_SPAN_S
        at, score = self._locate(slug, audio, end + offset, span)
        if at is not None and score >= LOCK_SCORE:
            self.misses = 0
            self._set("synced", f"{self._name()} at {_clock(at)} (alignment {score:.2f})", slug, at - end)
            return
        self.misses += 1
        # synced: a few misses mean the song changed. approximate: alignment hasn't worked yet,
        # which is expected with a poor mic, so keep dancing from AudD's position for a while
        limit = LOSE_AFTER if self.state == "synced" else int(RETRY_NO_DANCE_S / RETRY_MISS_S)
        if self.misses >= limit:                                      # lost it: recognize again
            self.misses = 0
            self.recognition = None
            self._set("listening", "lost the song; recognizing again", wait=0.0)
        else:
            with self._lock:
                self.next_check = self.mic_time() + RETRY_MISS_S


def _clock(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load_tracks(choreo_dir) -> dict[str, ChoreoTrack]:
    return {p.stem: ChoreoTrack(Choreo.load(p), p) for p in sorted(choreo_dir.glob("*.npz"))}
