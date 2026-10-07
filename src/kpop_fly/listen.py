"""Live listening: microphone audio -> which song, and exactly where in it, kept up to date.

  listening    collecting LISTEN_S of sound (silence doesn't count)
  recognizing  the clip is with AudD (on a worker thread: the window keeps running)
  synced       a saved dance is playing and the clip was located in it; the fly dances it
  approximate  a saved dance, but the clip couldn't confirm AudD's position: dance from AudD's
               timecode while retrying
  stopped      the song stopped: brain only, and the brain hears silence (not amplified room
               noise) until sound comes back
  no dance     AudD knows the song, but there's no saved dance for it: brain only
  not found    neither AudD nor a search of the saved songs recognized it: brain only

Locating a clip in a song takes two stages: chroma (harmony) finds the bar, then the onset
envelope finds the exact 20 ms frame. Through a speaker and a room, rhythm alone fits one beat or
one bar off about half the time (features.py has the numbers).

AudD's timecode, measured against the studio tracks the dances are timed to, sits 1.6-2.9 s
before the END of the clip it was sent, whatever the clip's length (whole seconds). So the clip's
end is searched for within +-LOCK_SPAN_S of timecode + AUDD_LAG_S: narrow enough that a 4-7 s
clip can't land on a repeated phrase (60/60 correct through simulated room audio at each length).

When AudD doesn't know the recording (a live stage with live vocals, say), or there's no token or
no network, every saved song is searched end to end instead. Without AudD's timecode nothing tells
a song's repeated sections apart, so a local lock can land on another chorus (in tests, about 1
time in 10); usually the choreography repeats too.

While a dance plays, every STILL_EVERY_S the last STILL_S of audio is checked against the song
where it should be by now (room audio: the song scores 0.89+, noise and other songs <= 0.31).
When that fails for STOP_AFTER_S:
  - and the level has dropped too (or gone under the silence line): the song STOPPED. A pause
    or the end of the song; room noise can still sit above the silence line, which is why
    the check doesn't rely on the level alone. When sound returns, the same song is looked for
    first (a resumed pause, any position) without asking AudD.
  - and it's still loud: first the same song is searched end to end (a skip within it, or a lock
    on the wrong repeat of a chorus whose repeats have now diverged: AudD's own timecode can point
    at either repeat). Only if it isn't there has the song CHANGED: recognize again, from audio
    after the change.
Every RESYNC_S a longer window also re-locates the song, to keep the position exact. None of this
costs API requests.
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

LISTEN_S = 6.0            # sound needed before recognizing (what AudD hears); --listen-seconds
CLIP_S = 10.0             # the onset alignment window, once there's that much
LONG_S = 18.0             # the most audio a whole-song search uses (repeats need context)
LEAD_S = 2.0              # extra audio before a window so the onset detector is warmed up
BUFFER_S = LONG_S + LEAD_S + 2.0
SILENCE_DB = -65.0        # quieter than this (dBFS, averaged over LOUDNESS_S) is silence
LOUDNESS_S = 0.5
AUDD_LAG_S = 2.3          # AudD's timecode ~ the clip's end minus this (measured 1.6-2.9 s)
LOCK_SPAN_S = 2.5         # search +-this around that for the first lock
RESYNC_SPAN_S = 3.0       # ...and +-this around the predicted position afterwards
WIDE_SPAN_S = 20.0        # approximate mode keeps looking this widely
FINE_SPAN_S = 0.2         # the onset stage searches +-this around the chroma position
LOCK_SCORE = 0.40         # chroma alignment needed to trust a position (room audio: real 0.55+, others <= 0.22)
LOCAL_SCORE = 0.35        # to say "it's this saved song" from a whole-song search (room audio: real 0.38+,
LOCAL_MARGIN = 0.15       # others <= 0.27), when it also beats the next saved song by this (real 0.20+, others <= 0.09)
LOCAL_SCORE_ALONE = 0.45  # with only one saved song there's no runner-up to beat: be stricter
STILL_S = 3.0             # "still playing?": this much recent audio...
STILL_EVERY_S = 1.0       # ...checked this often...
STILL_SCORE = 0.60        # ...must score this at the predicted position (song 0.89+, anything else <= 0.31)
STILL_SPAN_S = 0.3
REFIND_SCORE = 0.60       # a whole-song search for the same song after a jump (room audio: it 0.87+, others <= 0.34)
STOP_AFTER_S = 2.0        # failing for this long (on top of the window still holding the song): stopped or changed
DROP_DB = 5.0             # stopped (not changed) if the level fell this far below the song's
BACK_DB = 4.0             # in "stopped", sound is back when the level rises this far above the quiet
RESYNC_S = 6.0
RETRY_MISS_S = 2.0        # approximate mode: look again this soon
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
    fine_s = min(CLIP_S, len(audio) / sr - LEAD_S)          # what the onset stage has (less than CLIP_S early on)
    env = onset_envelope(audio, sr)[-int(fine_s * ONSET_RATE):]
    fine, _ = _peak(sliding_ncc(smooth(env.astype(float)), smooth(onset_ref.astype(float))), ONSET_RATE,
                    end - fine_s, FINE_SPAN_S)
    return (fine + fine_s if fine is not None else end), score


class LiveDance:
    """Feed it audio blocks; read `track`, `song_time()`, `status` and `input_open` from the display."""

    def __init__(self, sr: int, tracks: dict[str, ChoreoTrack], api_token: str | None, max_requests: int = 50,
                 recognizer: Callable[[np.ndarray, int, str], Recognition | None] = recognize,
                 clock: Callable[[], float] = time.perf_counter, log: Callable[[str], None] = print,
                 threaded: bool = True, silence_db: float = SILENCE_DB,
                 level_db: Callable[[], float] | None = None, listen_s: float = LISTEN_S):
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
        self.listen_s = listen_s
        self.ring = Ring(int(BUFFER_S * sr))
        self.samples = 0                         # mic samples received: the mic clock
        self._stamp = (0, clock())               # (samples, wall time) at the last block
        self._energy = 0.0                       # mean square, smoothed over LOUDNESS_S
        self.level = -120.0                      # dBFS, raw
        self.loud_s = 0.0                        # seconds of sound in a row
        self.quiet_s = 0.0                       # seconds of silence in a row
        self.slug: str | None = None
        self.offset: float | None = None         # song time = mic time + offset
        self.state = "listening"
        self.detail = "" if api_token else f"no {TOKEN_ENV}: only the saved songs can be recognized"
        self.recognition: Recognition | None = None
        self.requests = 0
        self.misses = 0
        self.fresh_from = 0.0                    # mic time from which audio counts for recognizing (after a change)
        self.failing_since: float | None = None  # when "still playing?" started failing
        self.song_db: float | None = None        # the song's level while it plays in sync
        self.quiet_db = -120.0                   # the level in "stopped"
        self.next_check = 0.0                    # mic time of the next check
        self.next_resync = 0.0
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
        self.level = self.level_db() if self.level_db else 10 * np.log10(self._energy + 1e-12)
        loud = self.level > self.silence_db
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
        return self.tracks.get(self.slug) if self.slug and self.state in ("synced", "approximate") else None

    @property
    def input_open(self) -> bool:
        """False while the song is stopped: the brain should hear silence, not the room."""
        return self.state != "stopped"

    def mic_time(self) -> float:
        samples, at = self._stamp
        return samples / self.sr + (self.clock() - at)

    def song_time(self) -> float | None:
        with self._lock:
            if self.slug is None or self.offset is None or self.state not in ("synced", "approximate"):
                return None
            return self.mic_time() + self.offset

    @property
    def status(self) -> str:
        quota = f"   AudD requests: {self.requests}" if self.requests else ""
        return f"{self.state}{': ' + self.detail if self.detail else ''}{quota}"

    # ---- the worker --------------------------------------------------------------------------

    def _set(self, state: str, detail: str = "", slug: str | None = None, offset: float | None = None,
             wait: float = STILL_EVERY_S) -> None:
        with self._lock:
            changed = (state, detail) != (self.state, self.detail)
            self.state, self.detail, self.slug, self.offset = state, detail, slug, offset
            self.next_check = self.mic_time() + wait
        if changed:
            self.log(f"[{self.mic_time():6.1f}s] {self.status}")

    def _again(self, wait: float) -> None:
        with self._lock:
            self.next_check = self.mic_time() + wait

    def _check(self) -> None:
        try:
            now = self.samples / self.sr
            if self.state == "stopped":
                self._stopped(now)
            elif self.state == "synced":
                self._playing(now)
            elif self.state == "approximate":
                self._approximate()
            elif self.quiet_s >= STOP_AFTER_S:
                if self.detail != "quiet":
                    self._set("listening", "quiet")
                else:
                    self._again(1.0)
            elif self.loud_s >= self.listen_s and now - self.fresh_from >= self.listen_s:
                self._recognize()
            else:
                self._again(0.5)
        except Exception as e:                                       # never take the dance down with us
            self._set("error", f"{type(e).__name__}: {e}", wait=RETRY_NOT_FOUND_S)
        finally:
            self._busy = False

    def _audio(self, window_s: float) -> tuple[np.ndarray, float]:
        """The last window_s of audio (plus LEAD_S before it), and the mic time of its last sample."""
        return self.ring.last(int((window_s + LEAD_S) * self.sr)), self.samples / self.sr

    def _chroma_of(self, slug: str) -> np.ndarray:
        if slug not in self._chroma:
            self._chroma[slug] = song_chroma(self.tracks[slug].audio)
        return self._chroma[slug]

    def _locate(self, slug: str, audio: np.ndarray, around: float, span: float,
                window_s: float) -> tuple[float | None, float]:
        return locate(audio, self.sr, self.tracks[slug].choreo.onset, self._chroma_of(slug), around, span, window_s)

    def _whole_song(self, slug: str, audio: np.ndarray, window_s: float) -> tuple[float | None, float]:
        duration = len(self._chroma_of(slug)) / CHROMA_RATE         # the audio's length, not the dance's
        return self._locate(slug, audio, duration / 2, duration / 2 + window_s, window_s)

    def _name(self) -> str:
        if self.recognition:
            return self.recognition.name
        return self.tracks[self.slug].title if self.slug else ""

    def _sound_s(self, now: float) -> float:
        """Seconds of usable sound: loud, and since the last song change."""
        return min(self.loud_s, now - self.fresh_from)

    # -- recognizing --

    def _recognize(self) -> None:
        window = min(self.listen_s, self._sound_s(self.samples / self.sr))
        audio, end = self._audio(window)
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
                rec = self.recognizer(audio[-int(window * self.sr):], self.sr, self.token)
                why = "" if rec else "AudD didn't recognize this"
            except AudDError as e:
                why = str(e)
        self.recognition = rec

        if rec is not None:
            slug = saved_dance(rec, self.songs)
            if slug is None:
                kpop = " (K-pop)" if rec.is_kpop else ""
                self._set("no dance", f"{rec.name}{kpop}: no saved dance", wait=RETRY_NO_DANCE_S)
                return
            if rec.timecode is None:
                at, score = self._whole_song(slug, audio, window)
            else:
                at, score = self._locate(slug, audio, rec.timecode + AUDD_LAG_S, LOCK_SPAN_S, window)
            if at is not None and score >= LOCK_SCORE:
                self._lock_on(slug, at, end, f"{rec.name} at {_clock(at)} (alignment {score:.2f})")
            elif rec.timecode is not None:
                self.misses = 0
                self._set("approximate", f"{rec.name}: dancing from AudD's position", slug,
                          rec.timecode + AUDD_LAG_S - end, wait=RETRY_MISS_S)
            else:
                self._set("no dance", f"{rec.name}: couldn't find the position", wait=RETRY_NOT_FOUND_S)
            return

        # AudD couldn't say (live vocals, no network, no token...): is it one of our own songs?
        window = min(LONG_S, self._sound_s(end))
        audio, end = self._audio(window)
        found = sorted(((score, slug, at) for slug in self.tracks
                        for at, score in [self._whole_song(slug, audio, window)] if at is not None), reverse=True)
        score, slug, at = found[0] if found else (0.0, None, None)
        if len(found) >= 2:
            ok = score >= LOCAL_SCORE and score - found[1][0] >= LOCAL_MARGIN
        else:
            ok = score >= LOCAL_SCORE_ALONE
        if slug is not None and ok:
            self._lock_on(slug, at, end, f"{self.tracks[slug].title} at {_clock(at)} (recognized locally: {why})")
        else:
            self._set("not found", f"{why}; not one of the saved songs either",
                      wait=RETRY_NOT_FOUND_S if asked else RETRY_LOCAL_S)

    def _lock_on(self, slug: str, at: float, end: float, detail: str) -> None:
        self.misses, self.failing_since, self.song_db = 0, None, None
        self.next_resync = end + RESYNC_S
        self._set("synced", detail, slug, at - end)

    # -- dancing --

    def _playing(self, now: float) -> None:
        slug, offset = self.slug, self.offset
        if now >= self.next_resync and self._sound_s(now) >= CLIP_S:      # keep the position exact
            self.next_resync = now + RESYNC_S
            audio, end = self._audio(CLIP_S)
            at, score = self._locate(slug, audio, end + offset, RESYNC_SPAN_S, CLIP_S)
            if at is not None and score >= LOCK_SCORE:
                offset = at - end
                self._set("synced", f"{self._name()} at {_clock(at)} (alignment {score:.2f})", slug, offset)

        audio, end = self._audio(STILL_S)                                    # still playing?
        expect = end + offset - STILL_S
        sc = chroma_ncc(chroma(audio[-int(STILL_S * self.sr):], self.sr), self._chroma_of(slug))
        _, still = _peak(sc, CHROMA_RATE, expect, STILL_SPAN_S)
        if still >= STILL_SCORE and self.quiet_s < STOP_AFTER_S:
            self.failing_since = None
            self.song_db = self.level if self.song_db is None else self.song_db + 0.1 * (self.level - self.song_db)
            self._again(STILL_EVERY_S)
            return
        if self.failing_since is None:
            self.failing_since = now
        if now - self.failing_since < STOP_AFTER_S and self.quiet_s < STOP_AFTER_S:
            self._again(STILL_EVERY_S)
            return
        dropped = self.quiet_s >= STOP_AFTER_S or (self.song_db is not None and self.level < self.song_db - DROP_DB)
        if dropped:
            self.quiet_db = self.level
            with self._lock:                                                  # keep slug/offset to resume
                self.state, self.detail = "stopped", f"{self._name()} stopped; brain only"
                self.next_check = self.mic_time() + STILL_EVERY_S
            self.log(f"[{self.mic_time():6.1f}s] {self.status}")
        else:
            # still loud: did it jump within the same song (a skip, or it was on the other chorus all
            # along and the repeats just diverged)? Look there first; it costs no request
            window = min(CLIP_S, max(STILL_S, now - self.failing_since + 0.5))   # audio from after the jump
            audio, end = self._audio(window)
            at, score = self._whole_song(slug, audio, window)
            if at is not None and score >= REFIND_SCORE:
                self._lock_on(slug, at, end, f"{self._name()}: found its place again at {_clock(at)} "
                                             f"(alignment {score:.2f})")
                return
            self.fresh_from = self.failing_since                            # recognize from after the change
            self.recognition = None
            self._set("listening", "the song changed; recognizing again", wait=0.0)

    def _approximate(self) -> None:
        audio, end = self._audio(min(CLIP_S, self._sound_s(self.samples / self.sr)))
        window = min(CLIP_S, self._sound_s(end))
        at, score = self._locate(self.slug, audio, end + self.offset, WIDE_SPAN_S, window)
        if at is not None and score >= LOCK_SCORE:
            self._lock_on(self.slug, at, end, f"{self._name()} at {_clock(at)} (alignment {score:.2f})")
            return
        self.misses += 1
        if self.misses >= int(RETRY_NO_DANCE_S / RETRY_MISS_S):            # never confirmed: ask again
            self.misses, self.recognition = 0, None
            self._set("listening", "couldn't confirm the position; recognizing again", wait=0.0)
        else:
            self._again(RETRY_MISS_S)

    # -- stopped --

    def _stopped(self, now: float) -> None:
        if self.level < self.quiet_db:
            self.quiet_db = self.level                                      # follow the quiet down
        back = self.level > self.silence_db and self.level >= self.quiet_db + BACK_DB
        if not back:
            self.fresh_from = now                                          # nothing to recognize from yet
            self._again(STILL_EVERY_S)
            return
        heard = now - self.fresh_from
        if heard < STILL_S + 1.0:
            self._again(STILL_EVERY_S)
            return
        slug = self.slug                                                    # the same song, resumed?
        window = min(CLIP_S, heard)
        audio, end = self._audio(window)
        at, score = self._locate(slug, audio, end + self.offset, RESYNC_SPAN_S, window)   # carried on playing
        threshold = STILL_SCORE                                             # (a quiet passage, not a pause)
        if at is None or score < STILL_SCORE:
            at, score = self._whole_song(slug, audio, window)                 # paused and resumed
            threshold = REFIND_SCORE
        if at is not None and score >= threshold:
            self._lock_on(slug, at, end, f"{self._name()} resumed at {_clock(at)} (alignment {score:.2f})")
        else:
            self.recognition = None
            self._set("listening", "something else is playing; recognizing it", wait=0.0)


def _clock(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load_tracks(choreo_dir) -> dict[str, ChoreoTrack]:
    return {p.stem: ChoreoTrack(Choreo.load(p), p) for p in sorted(choreo_dir.glob("*.npz"))}
