"""Which saved dance is this recording, and where in the song is each moment of it?

A music video or a live stage isn't the dance practice video: it has its own intro, cuts and
mix. So a recording is matched on its onset envelope (the same 50 Hz signal every choreo file
stores for its song):

  1. Slide 10 s windows of the recording along the song's envelope (normalized
     cross-correlation) to score every window at every offset.
  2. Collect the offsets that recur across windows as hypotheses.
  3. Viterbi: pick one hypothesis (or "not the song") per window, paying a penalty to jump.
     Pop songs repeat their choruses, and a chorus window matches every chorus about equally
     well; the jump penalty keeps the path on one consistent offset instead of hopping between
     repeats. Intros and inserted scenes fall to "not the song".
  4. Offsets the path proved can also claim weaker stretches elsewhere (a quieter mix, sound
     effects over the music), if they hold up for REENTRY_S at REENTRY. A song with no proven
     offset gets nothing from this step, so it can't turn a non-match into a match.

The result is a time map: segments of the recording, each at a fixed offset into the song.
On the test set (3 MVs + 3 TV stages of the saved songs; TWICE "What is Love?", "LIKEY" and
BTS "Dynamite" as non-matches), every real match came out as one clean segment and every
non-match scored zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np

from .choreo import ONSET_RATE, Choreo

WINDOW_S = 10.0
HOP_S = 2.5
NOT_THE_SONG = 0.40       # the score a window gets for "not the song": real matches score ~0.45-0.75
JUMP = 1.5                # cost of changing offset (half of it to enter or leave "not the song")
MIN_RUN_S = 40.0          # a match needs one consistent stretch this long (or half the recording)
MIN_COVER = 0.25          # ...and to cover this much of the recording
REENTRY = 0.42            # a proven offset also claims other stretches averaging this (non-matches: 90th pct)
REENTRY_S = 10.0          # ...that last at least this long


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Onsets in two recordings of a song land within a sample or so of each other; a little blur helps."""
    return np.convolve(x, np.ones(k) / k, "same")


def sliding_ncc(window: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of `window` against every position in `ref` (FFT, O(n log n))."""
    W, N = len(window), len(ref)
    if N < W:
        return np.zeros(0)
    w = window - window.mean()
    norm = np.sqrt((w ** 2).sum())
    if norm < 1e-9:
        return np.zeros(N - W + 1)
    n = 1 << int(np.ceil(np.log2(N + W)))
    corr = np.fft.irfft(np.fft.rfft(ref, n) * np.fft.rfft(w[::-1], n), n)[W - 1:N]
    c1 = np.concatenate([[0.0], np.cumsum(ref)])
    c2 = np.concatenate([[0.0], np.cumsum(ref ** 2)])
    s, s2 = c1[W:] - c1[:-W], c2[W:] - c2[:-W]
    return corr / (norm * np.sqrt(np.maximum(s2 - s * s / W, 1e-9)))


@dataclass
class Segment:
    start: float              # seconds into the recording
    end: float
    offset: float             # song time = recording time + offset
    score: float              # mean window score


@dataclass
class Match:
    slug: str
    segments: list[Segment]
    duration: float           # of the recording
    how: str                  # "video id" or "audio"

    @property
    def cover(self) -> float:
        return sum(s.end - s.start for s in self.segments) / max(self.duration, 1e-9)

    @property
    def longest(self) -> float:
        return max((s.end - s.start for s in self.segments), default=0.0)

    @property
    def score(self) -> float:
        total = sum(s.end - s.start for s in self.segments)
        return sum(s.score * (s.end - s.start) for s in self.segments) / total if total else 0.0

    def ok(self) -> bool:
        return self.longest >= min(MIN_RUN_S, 0.5 * self.duration) and self.cover >= MIN_COVER

    def song_time(self, t: float) -> float | None:
        """Where in the song recording time `t` is, or None if this moment isn't the song."""
        for s in self.segments:
            if s.start <= t < s.end:
                return t + s.offset
        return None

    def describe(self) -> str:
        parts = []
        for s in self.segments:
            span = f"{_clock(s.start)}-{_clock(s.end)}"
            parts.append(f"{span} -> song {_clock(s.start + s.offset)}")
        return f"{self.cover:.0%} of it matches ({'; '.join(parts)})"


def _clock(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def align(env: np.ndarray, ref: np.ndarray) -> list[Segment]:
    """Segments of `env` (recording) that line up with `ref` (song), both onset envelopes at ONSET_RATE."""
    W, H = int(WINDOW_S * ONSET_RATE), int(HOP_S * ONSET_RATE)
    env, ref = smooth(env.astype(float)), smooth(ref.astype(float))
    starts = list(range(0, len(env) - W + 1, H))
    if not starts:
        return []
    scores = [sliding_ncc(env[i:i + W], ref) for i in starts]

    bin_ = 5                                               # hypotheses in 0.1 s bins
    votes: dict[int, int] = {}
    for i, c in zip(starts, scores):
        for j in np.argsort(c)[::-1][:3]:
            key = int(round((j - i) / bin_))
            votes[key] = votes.get(key, 0) + 1
    hyps: list[int] = []
    for key, v in sorted(votes.items(), key=lambda kv: -kv[1]):
        if v >= 3 and all(abs(key - h) > 2 for h in hyps):
            hyps.append(key)
    hyps = hyps[:40]
    if not hyps:
        return []

    tol = 6                                                # +-0.12 s around each hypothesis
    S = np.full((len(starts), len(hyps) + 1), NOT_THE_SONG)
    for w, (i, c) in enumerate(zip(starts, scores)):
        for h, key in enumerate(hyps):
            j = i + key * bin_
            lo, hi = max(0, j - tol), min(len(c), j + tol + 1)
            S[w, h + 1] = c[lo:hi].max() if lo < hi else -1.0

    m = S.shape[1]
    penalty = np.full((m, m), JUMP)
    penalty[0, :] = penalty[:, 0] = JUMP / 2
    np.fill_diagonal(penalty, 0.0)
    best, back = S[0].copy(), np.zeros(S.shape, int)
    for w in range(1, len(starts)):
        total = best[:, None] - penalty
        back[w] = total.argmax(0)
        best = total.max(0) + S[w]
    path = [int(best.argmax())]
    for w in range(len(starts) - 1, 0, -1):
        path.append(int(back[w, path[-1]]))
    path.reverse()

    segments: list[Segment] = []
    first = 0
    for state, group in groupby(path):
        last = first + len(list(group)) - 1
        if state:
            segments.append(Segment(start=starts[first] / ONSET_RATE, end=(starts[last] + W) / ONSET_RATE,
                                    offset=hyps[state - 1] * bin_ / ONSET_RATE,
                                    score=float(S[first:last + 1, state].mean())))
        first = last + 1

    claimed = np.zeros(len(starts), bool)
    for seg in segments:
        claimed[int(seg.start * ONSET_RATE) // H:int((seg.end * ONSET_RATE - W) // H) + 1] = True
    need = int(np.ceil(REENTRY_S / HOP_S))
    for state in {p for p in path if p}:
        steady = np.convolve(S[:, state], np.ones(3) / 3, "same") >= REENTRY
        first = 0
        for ok, group in groupby(steady & ~claimed):
            n = len(list(group))
            if ok and n >= need:
                segments.append(Segment(start=starts[first] / ONSET_RATE, end=(starts[first + n - 1] + W) / ONSET_RATE,
                                        offset=hyps[state - 1] * bin_ / ONSET_RATE,
                                        score=float(S[first:first + n, state].mean())))
            first += n
    segments.sort(key=lambda seg: seg.start)
    for a, b in zip(segments, segments[1:]):               # windows overlap: split the difference
        if b.start < a.end:
            a.end = b.start = (a.end + b.start) / 2
    return segments


def match(env: np.ndarray, choreos: dict[str, Choreo]) -> tuple[Match | None, list[Match]]:
    """The best acceptable match among the saved dances (or None), and every candidate for reporting."""
    duration = len(env) / ONSET_RATE
    candidates = [Match(slug, align(env, c.onset), duration, "audio") for slug, c in choreos.items()]
    candidates.sort(key=lambda m: m.cover * m.score, reverse=True)
    ok = [m for m in candidates if m.ok()]
    return (ok[0] if ok else None), candidates


def by_video_id(video_id: str, choreos: dict[str, Choreo], duration: float) -> Match | None:
    """The recording *is* one of the dance practice videos: no alignment needed."""
    for slug, c in choreos.items():
        if (c.meta.get("source") or {}).get("id") == video_id:
            return Match(slug, [Segment(0.0, duration, 0.0, 1.0)], duration, "video id")
    return None
