# kpop-fly

A fruit fly that dances to K-pop, and whose moves are timed by a simulation of its real brain.

Sound drives the antennal hearing neurons (Johnston's organ, JO) of the complete male fruit fly
central nervous system, the [MaleCNS v1.0 connectome](https://male-cns.janelia.org) (166,700
neurons, 25.6 M synapses), simulated as a spiking network by [flybrain](https://github.com/alextitonis/fly.ai).
The descending neurons (DNs), the brain's commands to the body, are read out every 20 ms and move
a cartoon fly.

**Phase 1 (this):** brain → beat. Any audio → onsets → antennae → connectome → DNs → 2D fly.
**Phase 2:** song recognition + choreography extracted from the music video, performed with the brain's timing.

## Setup

Needs Apple Silicon or Linux, and a **native arm64 Python on Macs**: numba, MediaPipe and MuJoCo no
longer ship x86_64 macOS wheels, so an Intel Homebrew Python running under Rosetta can't install
them. `.python-version` pins `cpython-3.13-macos-aarch64`, so uv picks the right one:

```sh
uv sync
uv run kpop-fly metronome            # quickest check: a 120 BPM click track
```

The first run downloads the connectome (~260 MB, to `~/fly-data`, or `$FLY_DATA`) and calibrates
(~4 s, cached in `~/.cache/kpop-fly`).

## Usage

```sh
uv run kpop-fly mic                  # dance to the microphone (allow mic access for your terminal)
uv run kpop-fly file song.mp3        # play a file (wav/flac/ogg/mp3) and dance to it
uv run kpop-fly metronome --bpm 128
uv run kpop-fly analyze song.mp3     # no window or sound: run it through the brain and report
uv run kpop-fly calibrate            # show which DNs hear sound, with latency and effect size
uv run kpop-fly devices              # list audio devices, for `mic --device N`
```

Useful flags: `--hearing sound` drives only the sound-tuned JO-A/B neurons instead of all 672
(far fewer DNs respond); `--gain` scales the antennal drive; `--mute` dances without playing
audio; `--no-window` prints status lines; `--headless --screenshot out.png --seconds 5` renders
off-screen.

## How it works

```
audio ──► OnsetDetector ──► strength 0..1 ──► voltage into 672 JO neurons
  (mic / file / metronome)   spectral flux,      held 60 ms per hit, like calibration
                             adaptive range
                                                        │  flybrain: leaky integrate-and-fire over
                                                        ▼  the whole CNS, 20 ms steps, ~2 ms each
Dancer springs ◄── channel levels 0..1 ◄── which of the 252 sound-responsive DNs fired
 (overshoot = bounce)   all / front / mid / hind / left / right
```

- **Calibration** (`brain.calibrate`) plays the brain a 120 BPM pulse train and silence with the
  same random seeds, and keeps the DNs whose firing reliably rises after a pulse (3 seeds, 3-sigma
  test; with zero drive it finds none). Of 1,314 DNs, 252 respond, mostly DNg08, DNg07, DNg106,
  DNge091 and DNg12 types.
- **Channels:** responders are split into thirds by response latency (fast → arms, middle → mid
  legs, slow → knees), so each beat ripples down the body in the order the wiring sets. Left/right
  channels come from which side of the brain each DN sits on, and drive the lean.
- **Background noise off:** the brain runs with `sensory_input=False`. With the default sensory
  background noise (~13,000 spikes/step), sound barely changes DN output. Without it, a beat takes
  DN activity from ~5 to 100+ spikes in the next step.

Measured on an M4 MacBook: 10 s of 120 BPM clicks runs in about 1 s offline; DN activity 20–100 ms
after a beat is 2.9× its level elsewhere (1.6× on a dense pop track with off-beat hi-hats, which
the fly also dances to). Live with the window open, a step takes about 4 ms of each 20 ms.

### What's honest to claim

- **True:** a simulation of the real fly connectome hears the music through its antennal neurons,
  and *when* the fly moves, and which body part moves first, comes from its descending neurons'
  response.
- **Artistic:** *how far* each body part moves per spike, the cartoon body, the spring bounce and
  pressing sound onsets straight onto JO neurons (a crude stand-in for antenna mechanics).
- **Not claimed:** that these DNs control dancing (or these body parts) in a real fly, or that
  the simple neuron model reproduces real fly physiology.

## Layout

```
src/kpop_fly/
  beat.py     onset detection and tempo
  brain.py    connectome, calibration, per-step readout into body channels
  engine.py   Pipeline (audio → brain, clock-free), BrainThread (real time), analyze (offline)
  audio.py    mic / file / metronome sources
  fly.py      springs and 2D geometry (pure math, tested)
  viz.py      pygame window
  cli.py      commands
tests/        uv run pytest  (test_brain.py loads the real connectome; -m "not slow" skips it)
```

## Credits

Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory
of Molecular Biology and Google Research, [CC BY 4.0](https://male-cns.janelia.org/download/).
Berg, S. et al. (2026), *Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system*, *Cell*. Simulation: [flybrain](https://pypi.org/project/flybrain/) (MIT),
neuron model after [Fly64](https://github.com/ornata/fly).
