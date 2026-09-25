"""kpop-fly command line.

  kpop-fly mic                   dance to the microphone
  kpop-fly file song.mp3         play a file and dance to it
  kpop-fly metronome --bpm 120   dance to a click track
  kpop-fly analyze song.mp3      no window, no audio: run a recording through the brain and report
  kpop-fly calibrate             (re)measure which descending neurons hear sound
  kpop-fly devices               list audio devices
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

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

    if args.cmd == "mic":
        source = MicSource(args.device)
    elif args.cmd == "file":
        source = PlaybackSource.file(args.path)
    else:
        source = PlaybackSource.metronome(args.bpm)

    pipe = Pipeline(_brain(args), source.sr)
    brain_thread = BrainThread(pipe)
    display = None
    if not args.no_window:
        from .viz import Display
        display = Display(pipe, source.name, headless=args.headless)
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


def cmd_calibrate(args) -> None:
    b = ListeningBrain(hearing=args.hearing, recalibrate=True)
    cal = b.cal
    print(f"{'type':<22}{'side':<6}{'latency ms':>11}{'effect':>8}  tier")
    for i in np.lexsort((-cal.effect, cal.latency)):
        print(f"{cal.cell_type[i]:<22}{cal.side[i]:<6}{cal.latency[i] * b.dt * 1000:>11.0f}{cal.effect[i]:>8.2f}  "
              f"{('front', 'mid', 'hind')[cal.tier[i]]}")


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

    def live_opts(sp):
        brain_opts(sp)
        sp.add_argument("--seconds", type=float, help="stop after this long")
        sp.add_argument("--no-window", action="store_true", help="print status lines instead of opening a window")
        sp.add_argument("--screenshot", help="save the last frame to this PNG on exit")
        sp.add_argument("--headless", action="store_true", help="with --screenshot: render off-screen")
        sp.add_argument("--verbose", "-v", action="store_true", help="print status lines alongside the window")
        sp.set_defaults(func=cmd_live, mute=False)

    sp = sub.add_parser("mic", help="dance to the microphone")
    sp.add_argument("--device", help="input device index or name (see `kpop-fly devices`)")
    live_opts(sp)
    sp = sub.add_parser("file", help="play an audio file and dance to it")
    sp.add_argument("path")
    sp.add_argument("--mute", action="store_true", help="don't play the sound, just dance in time")
    live_opts(sp)
    sp = sub.add_parser("metronome", help="dance to a click track")
    sp.add_argument("--bpm", type=float, default=120)
    sp.add_argument("--mute", action="store_true", help="don't play the clicks, just dance in time")
    live_opts(sp)

    sp = sub.add_parser("analyze", help="run a recording through the brain offline and report")
    sp.add_argument("path", nargs="?", help="audio file (omit to use a metronome)")
    sp.add_argument("--bpm", type=float, default=120)
    sp.add_argument("--seconds", type=float, default=None, help="only the first N seconds")
    brain_opts(sp)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("calibrate", help="re-measure which descending neurons respond to sound")
    sp.add_argument("--hearing", choices=HEARING, default="all")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    args = p.parse_args(argv)
    if args.cmd == "analyze" and not args.path and not args.seconds:
        args.seconds = 10.0
    args.func(args)


if __name__ == "__main__":
    main()
