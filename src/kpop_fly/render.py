"""Offline render: the fly dancing a song, written to an mp4 with the song as its soundtrack.

Runs the same pipeline as a live session (audio -> onsets -> brain -> display), but on the
song's own clock instead of the wall clock: the brain steps every 20 ms of audio and the
display draws a frame every 1/fps s, so the result is deterministic and exactly in sync.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import load_audio
from .brain import ListeningBrain
from .engine import Pipeline
from .retarget import ChoreoTrack


def render(track: ChoreoTrack, brain: ListeningBrain, out: str | Path, start: float = 0.0,
           seconds: float | None = None, fps: float = 30.0, warmup: float = 2.0, mode: str = "blend") -> Path:
    from .viz import H, W, Display

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found; install it (e.g. `brew install ffmpeg`)")
    samples, sr = load_audio(track.audio)
    mono = samples.mean(axis=1)
    end = min(len(mono) / sr, start + seconds if seconds else np.inf)
    out = Path(out)

    pipe = Pipeline(brain, sr)
    display = Display(pipe, track.title, headless=True, choreo=track, mode=mode)
    hop = int(round(brain.dt * sr))
    cursor = max(0, int((start - warmup) * sr))       # let the brain hear a little before the first frame

    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}",
         "-i", "-", "-ss", f"{start}", "-t", f"{end - start}", "-i", str(track.audio),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)],
        stdin=subprocess.PIPE)
    frames = int((end - start) * fps)
    t0, last = time.perf_counter(), 0.0
    try:
        for i in range(frames):
            t = start + i / fps
            while cursor / sr < t:                     # brain catches up to this frame's audio
                pipe.feed(mono[cursor:cursor + hop])
                pipe.step()
                cursor += hop
            display.frame(dt=1 / fps, t=t)
            proc.stdin.write(display.pg.image.tobytes(display.screen, "RGB"))
            if time.perf_counter() - last > 10:
                last = time.perf_counter()
                print(f"  {t - start:.0f}/{end - start:.0f} s rendered")
    finally:
        proc.stdin.close()
        proc.wait()
        display.close()
    took = time.perf_counter() - t0
    print(f"saved {out} ({end - start:.0f} s of video in {took:.0f} s)")
    return out


def render_side_by_side(track: ChoreoTrack, make_brain, out: str | Path, start: float = 0.0,
                        seconds: float | None = None, fps: float = 30.0) -> Path:
    """Dance only (left) and dance + brain (right): the stage of each render, side by side.
    Each side gets its own brain, started from the same seed, so both hear the song identically."""
    from .viz import STAGE_W

    out = Path(out)
    parts = [out.with_name(f".{out.stem}-{mode}.mp4") for mode in ("dance", "blend")]
    try:
        for mode, part in zip(("dance", "blend"), parts):
            render(track, make_brain(), part, start=start, seconds=seconds, fps=fps, mode=mode)
        subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-i", str(parts[0]), "-i", str(parts[1]),
                        "-filter_complex", f"[0:v]crop={STAGE_W}:in_h:0:0[a];[1:v]crop={STAGE_W}:in_h:0:0[b];[a][b]hstack",
                        "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-c:a", "copy", str(out)], check=True)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    print(f"saved {out}")
    return out


def compare(track: ChoreoTrack, brain: ListeningBrain, start: float = 0.0, seconds: float | None = None,
            fps: float = 30.0, warmup: float = 2.0) -> dict[str, dict[str, float]]:
    """Run every mode on the same brain simulation and measure how the fly moves.

    For each mode: limb speed (px per frame, over all limb joints); hit punch (limb speed in the
    33-100 ms after the song's strongest onsets, the top 10%, relative to other frames: do moves
    land harder on the hits?); jerkiness (mean acceleration / mean speed: higher is twitchier);
    and how far the pose differs from dance-only (px, mean over
    joints): the brain's visible effect."""
    from .blend import MODES, Blender
    from .brain import CHANNELS
    from .fly import Dancer, skeleton

    samples, sr = load_audio(track.audio)
    mono = samples.mean(axis=1)
    end = min(len(mono) / sr, start + seconds if seconds else np.inf)
    pipe = Pipeline(brain, sr)
    hop = int(round(brain.dt * sr))
    cursor = max(0, int((start - warmup) * sr))
    dancers = {m: Dancer() for m in MODES}
    blenders = {m: Blender() for m in MODES}
    joints = {m: [] for m in MODES}
    onsets, beats = [], 0
    for i in range(int((end - start) * fps)):
        t = start + i / fps
        onset = 0.0
        while cursor / sr <= t:
            pipe.feed(mono[cursor:cursor + hop])
            frame, strength, b = pipe.step()
            onset, beats = max(onset, strength), beats + b
            cursor += hop
        onsets.append(onset)
        levels = dict(zip(CHANNELS, frame.levels))
        choreo = track.targets_at(t)
        for m in MODES:
            goals, k = blenders[m](m, levels, frame.drive, choreo, 1 / fps)
            sk = skeleton(dancers[m].follow(goals, 1 / fps, k), 0, 0)
            joints[m].append([p for leg in sk.legs for p in leg[1:]] + [sk.head[0]])
    onsets = np.array(onsets)[1:]
    hits = onsets >= np.percentile(onsets, 90)
    after = np.zeros_like(hits)
    for lag in range(1, 4):
        after[lag:] |= hits[:-lag]
    report = {}
    for m in MODES:
        j = np.array(joints[m])                                    # (frames, joints, 2)
        speed = np.linalg.norm(np.diff(j, axis=0), axis=2).mean(1)
        accel = np.linalg.norm(np.diff(j, n=2, axis=0), axis=2).mean(1)
        report[m] = {"speed": float(speed.mean()), "punch": float(speed[after].mean() / speed[~after].mean()),
                     "jerk": float(accel.mean() / max(speed.mean(), 1e-9)),
                     "vs_dance": float(np.linalg.norm(j - np.array(joints["dance"]), axis=2).mean())}
    report["_"] = {"beats": int(beats), "seconds": end - start}
    return report
