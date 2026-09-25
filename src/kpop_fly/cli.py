"""kpop-fly command line.

  kpop-fly mic                   dance to the microphone
  kpop-fly file song.mp3         play a file and dance to it
  kpop-fly dance fancy           dance an extracted song's choreography, with its audio
  kpop-fly render fancy          render that to an mp4 (offline)
  kpop-fly compare fancy         measure dance-only vs brain-only vs blended
  kpop-fly metronome --bpm 120   dance to a click track
  kpop-fly analyze song.mp3      no window, no audio: run a recording through the brain and report
  kpop-fly calibrate             (re)measure which descending neurons hear sound
  kpop-fly extract fancy         turn a dance practice video into choreo/fancy.npz
  kpop-fly songs                 list songs and whether they're extracted
  kpop-fly devices               list audio devices
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from .blend import MODES
from .brain import HEARING, ListeningBrain


def _check_arch() -> None:
    if sys.platform == "darwin" and platform.machine() != "arm64":
        import subprocess
        translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
        if translated.stdout.strip() == "1":
            sys.exit("This Python is x86_64 running under Rosetta. numba/MediaPipe/MuJoCo need a native "
                     "arm64 Python:\n  uv sync   (the repo's .python-version pins cpython-3.13-macos-aarch64)")


def _brain(args) -> ListeningBrain:
    return ListeningBrain(hearing=args.hearing, gain=args.gain, recalibrate=getattr(args, "recalibrate", False))


def _sparkline(values: np.ndarray, beats: np.ndarray, width: int = 100) -> str:
    bars = " ▁▂▃▄▅▆▇█"
    v = values[:width]
    line = "".join(bars[min(8, int(x * 8.999))] for x in v)
    marks = "".join("|" if b else " " for b in beats[:width])
    return f"  beats {marks}\n  DNs   {line}"


def cmd_analyze(args) -> None:
    from .audio import click_track, load_audio
    from .engine import Pipeline, analyze

    if args.path:
        samples, sr = load_audio(args.path)
        name = args.path
    else:
        sr = 44100
        samples = np.tile(click_track(args.bpm, sr), (int(np.ceil(args.seconds * args.bpm / 240)), 1))
        name = f"metronome {args.bpm:g} BPM"
    if args.seconds:
        samples = samples[:int(args.seconds * sr)]
    pipe = Pipeline(_brain(args), sr)
    r = analyze(pipe, samples.mean(axis=1), sr)
    print(f"\n{name}: {r.seconds:.1f} s of audio in {r.wall_s:.1f} s ({r.seconds / r.wall_s:.1f}x realtime)")
    print(f"beats detected: {r.beats}   tempo: {f'{r.bpm:.1f} BPM' if r.bpm else 'n/a'}")
    print(f"responsive descending neurons: {r.n_dn}   top types: "
          + ", ".join(f"{t} x{c}" for t, c in r.top_types))
    print(f"beat lock: DN activity 20-100 ms after a beat is {r.beat_lock:.1f}x its level elsewhere")
    print("first 2 s (one column per 20 ms brain step):")
    print(_sparkline(r.level_trace, r.beat_trace))


def cmd_live(args) -> None:
    from .audio import MicSource, PlaybackSource
    from .engine import BrainThread, Pipeline

    track = None
    if args.cmd == "mic":
        source = MicSource(args.device)
    elif args.cmd == "dance":
        from .retarget import ChoreoTrack
        track = ChoreoTrack.load(args.song)
        source = PlaybackSource.file(track.audio)
    elif args.cmd == "file":
        source = PlaybackSource.file(args.path)
        if args.choreo:
            from .retarget import ChoreoTrack
            track = ChoreoTrack.load(args.choreo)
    else:
        source = PlaybackSource.metronome(args.bpm)
    if getattr(args, "start", None):
        source.seek(args.start)

    pipe = Pipeline(_brain(args), source.sr)
    brain_thread = BrainThread(pipe)
    display = None
    if not args.no_window:
        from .viz import Display
        display = Display(pipe, track.title if track else source.name, headless=args.headless,
                          choreo=track, song_time=source.position if track else None, mode=args.mode)
    source.start(pipe.feed, mute=args.mute)
    brain_thread.start()
    print(f"listening to {source.name}. {'Close the window or press q to quit.' if display else 'Ctrl-C to quit.'}")

    start = last_print = time.perf_counter()
    try:
        while not source.finished.is_set():
            now = time.perf_counter()
            if args.seconds and now - start >= args.seconds:
                break
            if display:
                if not display.frame():
                    break
            else:
                time.sleep(0.02)
            if now - last_print >= 1.0 and (args.no_window or args.verbose):
                last_print = now
                snap = pipe.timeline.snapshot()
                f = snap["latest"]
                bpm = pipe.detector.bpm()
                print(f"t={now - start:5.1f}s  beats {pipe.beats:4d}  {f'{bpm:5.1f} BPM' if bpm else '  -- BPM'}  "
                      f"step {snap['step_ms']:.1f} ms  DNs firing {int(f.dn_fired.sum()) if f else 0:3d}  "
                      f"level {f.level('all') if f else 0:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        if display and args.screenshot:
            display.screenshot(args.screenshot)
            print(f"saved {args.screenshot}")
        brain_thread.stop()
        source.stop()
        if display:
            display.close()
        if brain_thread.late_steps:
            print(f"warning: the brain fell behind real time {brain_thread.late_steps} times")


def cmd_render(args) -> None:
    from .render import render
    from .retarget import ChoreoTrack

    track = ChoreoTrack.load(args.song)
    if args.compare:
        from .render import render_side_by_side
        out = args.out or track.path.with_name(f"{track.path.stem}-compare.mp4")
        render_side_by_side(track, lambda: _brain(args), out, start=args.start or 0.0, seconds=args.seconds, fps=args.fps)
        return
    suffix = "" if args.mode == "blend" else f"-{args.mode}"
    out = args.out or track.path.with_name(f"{track.path.stem}-fly{suffix}.mp4")
    render(track, _brain(args), out, start=args.start or 0.0, seconds=args.seconds, fps=args.fps, mode=args.mode)


def cmd_compare(args) -> None:
    from .render import compare
    from .retarget import ChoreoTrack

    track = ChoreoTrack.load(args.song)
    r = compare(track, _brain(args), start=args.start or 0.0, seconds=args.seconds)
    info = r.pop("_")
    print(f"\n{track.title}: {info['seconds']:.0f} s, {info['beats']} beats detected")
    print(f"{'mode':<8}{'limb speed':>12}{'hit punch':>11}{'jerkiness':>11}{'differs from dance-only':>26}")
    for mode, m in r.items():
        print(f"{mode:<8}{m['speed']:>9.2f} px{m['punch']:>10.2f}x{m['jerk']:>11.2f}{m['vs_dance']:>23.1f} px")


def cmd_calibrate(args) -> None:
    b = ListeningBrain(hearing=args.hearing, recalibrate=True)
    cal = b.cal
    print(f"{'type':<22}{'side':<6}{'latency ms':>11}{'effect':>8}  tier")
    for i in np.lexsort((-cal.effect, cal.latency)):
        print(f"{cal.cell_type[i]:<22}{cal.side[i]:<6}{cal.latency[i] * b.dt * 1000:>11.0f}{cal.effect[i]:>8.2f}  "
              f"{('front', 'mid', 'hind')[cal.tier[i]]}")


def cmd_extract(args) -> None:
    from .choreo import CHOREO_DIR, Choreo, extract, load_songs, summarize

    songs = load_songs()
    if args.all:
        targets = list(songs)
    elif args.song in songs:
        targets = [args.song]
    else:
        targets = [args.song]                        # a YouTube URL or id not in songs.toml
    for slug in targets:
        song = songs.get(slug)
        name = slug if song else (args.slug or slug.rsplit("=", 1)[-1].rsplit("/", 1)[-1])
        path = CHOREO_DIR / f"{name}.npz"
        if path.exists() and not args.force:
            print(f"{path.name} already extracted (use --force to redo)")
            continue
        path = extract(name, song["youtube"] if song else slug, song, target_fps=args.fps, preview=args.preview,
                       max_seconds=args.seconds)
        print(f"saved {path}\n  {summarize(Choreo.load(path))}")


def cmd_songs(_args) -> None:
    from .choreo import CHOREO_DIR, Choreo, load_songs, summarize

    for slug, song in load_songs().items():
        path = CHOREO_DIR / f"{slug}.npz"
        status = summarize(Choreo.load(path)) if path.exists() else "not extracted"
        print(f"{slug:<10} {song['artist']} - {song['title']}  youtu.be/{song['youtube']}\n           {status}")


def cmd_devices(_args) -> None:
    import sounddevice as sd
    print(sd.query_devices())


def main(argv: list[str] | None = None) -> None:
    _check_arch()
    p = argparse.ArgumentParser(prog="kpop-fly", description="A fruit fly connectome that dances to K-pop.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def brain_opts(sp):
        sp.add_argument("--hearing", choices=HEARING, default="all",
                        help="which Johnston's organ neurons hear: all 672, or only the sound-tuned JO-A/B")
        sp.add_argument("--gain", type=float, default=1.0, help="voltage into the antennae at full onset strength")

    def mode_opt(sp):
        sp.add_argument("--mode", choices=MODES, default="blend",
                        help="with a choreography: dance + brain (blend), the dance alone, or the brain alone")

    def live_opts(sp):
        brain_opts(sp)
        mode_opt(sp)
        sp.add_argument("--seconds", type=float, help="stop after this long")
        sp.add_argument("--no-window", action="store_true", help="print status lines instead of opening a window")
        sp.add_argument("--screenshot", help="save the last frame to this PNG on exit")
        sp.add_argument("--headless", action="store_true", help="with --screenshot: render off-screen")
        sp.add_argument("--verbose", "-v", action="store_true", help="print status lines alongside the window")
        sp.set_defaults(func=cmd_live, mute=False)

    sp = sub.add_parser("mic", help="dance to the microphone")
    sp.add_argument("--device", help="input device index or name (see `kpop-fly devices`)")
    live_opts(sp)
    sp = sub.add_parser("dance", help="dance an extracted song's choreography, with its audio")
    sp.add_argument("song", help="a slug in choreo/ (e.g. fancy) or a path to its .npz")
    sp.add_argument("--start", type=float, help="start this many seconds into the song")
    sp.add_argument("--mute", action="store_true", help="don't play the sound, just dance in time")
    live_opts(sp)
    sp = sub.add_parser("file", help="play an audio file and dance to it")
    sp.add_argument("path")
    sp.add_argument("--choreo", help="also dance this choreography (slug or .npz); the file must be its song")
    sp.add_argument("--start", type=float, help="start this many seconds into the file")
    sp.add_argument("--mute", action="store_true", help="don't play the sound, just dance in time")
    live_opts(sp)
    sp = sub.add_parser("metronome", help="dance to a click track")
    sp.add_argument("--bpm", type=float, default=120)
    sp.add_argument("--mute", action="store_true", help="don't play the clicks, just dance in time")
    live_opts(sp)

    sp = sub.add_parser("render", help="render the fly dancing a song to an mp4 (offline, with audio)")
    sp.add_argument("song", help="a slug in choreo/ (e.g. fancy) or a path to its .npz")
    sp.add_argument("--out", help="output path (default: choreo/<song>-fly.mp4)")
    sp.add_argument("--start", type=float, help="start this many seconds into the song")
    sp.add_argument("--seconds", type=float, help="only render this long")
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--compare", action="store_true",
                    help="dance only and dance + brain side by side (default: choreo/<song>-compare.mp4)")
    brain_opts(sp)
    mode_opt(sp)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("compare", help="measure how each mode moves the fly on a song (offline)")
    sp.add_argument("song", help="a slug in choreo/ (e.g. fancy) or a path to its .npz")
    sp.add_argument("--start", type=float, help="start this many seconds into the song")
    sp.add_argument("--seconds", type=float, help="only this long")
    brain_opts(sp)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("analyze", help="run a recording through the brain offline and report")
    sp.add_argument("path", nargs="?", help="audio file (omit to use a metronome)")
    sp.add_argument("--bpm", type=float, default=120)
    sp.add_argument("--seconds", type=float, default=None, help="only the first N seconds")
    brain_opts(sp)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("calibrate", help="re-measure which descending neurons respond to sound")
    sp.add_argument("--hearing", choices=HEARING, default="all")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("extract", help="extract choreography from a dance practice video")
    sp.add_argument("song", nargs="?", help="a slug from songs.toml, or a YouTube URL/id")
    sp.add_argument("--all", action="store_true", help="every song in songs.toml")
    sp.add_argument("--slug", help="file name for a URL that isn't in songs.toml")
    sp.add_argument("--fps", type=float, default=30.0, help="poses per second to sample")
    sp.add_argument("--preview", action="store_true", help="also write choreo/<slug>.preview.mp4 to check the result")
    sp.add_argument("--force", action="store_true", help="re-extract even if the file exists")
    sp.add_argument("--seconds", type=float, help="only the first N seconds (for a quick test)")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("songs", help="list songs and their extraction status")
    sp.set_defaults(func=cmd_songs)

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    args = p.parse_args(argv)
    if args.cmd == "extract" and not (args.song or args.all):
        p.error("extract needs a song slug, a YouTube URL, or --all")
    if args.cmd == "analyze" and not args.path and not args.seconds:
        args.seconds = 10.0
    args.func(args)


if __name__ == "__main__":
    main()
