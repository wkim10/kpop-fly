# kpop-fly

A fruit fly that dances to K-pop, and whose moves are timed by a simulation of its real brain.

Sound drives the antennal hearing neurons (Johnston's organ, JO) of the complete male fruit fly
central nervous system, the [MaleCNS v1.0 connectome](https://male-cns.janelia.org) (166,700
neurons, 25.6 M synapses), simulated as a spiking network by [flybrain](https://github.com/alextitonis/fly.ai).
The descending neurons (DNs), the brain's commands to the body, are read out every 20 ms and move
a cartoon fly.

**Phase 1:** brain → beat. Any audio → onsets → antennae → connectome → DNs → 2D fly.
**Phase 2 (in progress):** choreography from dance practice videos, performed with the brain's timing.
Done: extraction (step 1), retargeting onto the fly (step 2), and blending the brain into the
moves (step 3). Next: YouTube URLs as input, then live song recognition.

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

### Choreography

```sh
uv run kpop-fly songs                    # the songs in songs.toml and whether they're extracted
uv run kpop-fly extract fancy --preview  # ~5 min per song on an M4; needs ffmpeg
uv run kpop-fly extract --all
```

```sh
uv run kpop-fly dance fancy              # the fly dances FANCY, with the song (choreo/fancy.opus)
uv run kpop-fly dance tt --start 60      # jump to 1:00
uv run kpop-fly render fancy             # write choreo/fancy-fly.mp4 offline (~4x real time)
uv run kpop-fly file choreo/tt.opus --choreo tt   # the same as `dance tt`, spelled out
```

`dance` finds the song's audio through the file name recorded in the choreography, so the two
always match. The fly's arms, legs, torso and head copy the consensus dancer (shown small in the
corner): upper-arm/forearm and thigh/shin directions carry over as-is, so elbows and knees bend
exactly as the dancer's do; torso tilt leans the upper body; the nose and ear line nod, shift and
roll the head. Everything stays in screen space, so the fly does what you'd see in the video.
Mid legs, wings and antennae have no human counterpart and are driven by the brain.

### Dance + brain

Three modes; press **m** in the window to cycle them, or pass `--mode`:

| mode | the moves | when they land, how hard |
|---|---|---|
| `blend` (default) | the dance | the brain: snaps into poses on DN bursts and follows loosely between them; fast responders exaggerate arm moves (70% quiet → 130% on a burst), slow responders scale leg moves and bounce the knees a beat later; all responders add a nod and flare the wings |
| `dance` | the dance | nothing: the brain runs (and shows in the panel) but drives nothing |
| `brain` | Phase 1's generic beat dance | the brain |

On pop songs the descending neurons never fall quiet (their level sits around 0.25–0.65,
against 0.1–0.8 for a metronome), so blended mode rescales each channel to its own range over
the last 4 s before using it.

```sh
uv run kpop-fly render fancy --compare   # choreo/fancy-compare.mp4: dance only | dance + brain
uv run kpop-fly compare fancy            # measure it
```

`compare` runs all three modes on one brain simulation. Across the three songs, blended mode moves
the limbs 70–95% more than the dance alone and lands harder on the song's strongest hits: limb
speed just after the top 10% of onsets is 1.08–1.22× its speed elsewhere, against 0.96–1.07× for
the dance alone. It stays about 16 px (mean, per joint) from the dance, so the choreography is
still recognizable, and it's somewhat jerkier (0.83 vs 0.73 acceleration per unit speed on FANCY).

`extract` downloads the dance practice video, finds every dancer in each frame (YOLOX-tiny +
RTMPose-m via rtmlib, pose model on CoreML), normalizes each dancer to their own body, and takes
the per-keypoint median across them: one consensus dancer, with mirror reflections and
off-count dancers outvoted. It writes `choreo/<slug>.npz` (consensus + every raw dancer + the
audio's onset envelope for syncing) and `choreo/<slug>.opus` (the song), then deletes the video.
`--preview` also writes `choreo/<slug>.preview.mp4`: the footage with detected skeletons next to
the consensus dancer, to check the result. `choreo/` is git-ignored because it's derived from
third-party videos. MediaPipe was the original plan, but it finds only 0–2 of 9 small dancers per
frame, and 1.0.x aborts on macOS ([#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)).

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
  choreo.py   dance practice video → consensus choreography (choreo/<slug>.npz)
  retarget.py consensus dancer → fly pose targets
  blend.py    dance-only / brain-only / blended modes
  render.py   offline mp4s of the fly dancing a song, and the mode comparison
  brain.py    connectome, calibration, per-step readout into body channels
  engine.py   Pipeline (audio → brain, clock-free), BrainThread (real time), analyze (offline)
  audio.py    mic / file / metronome sources
  fly.py      springs and 2D geometry (pure math, tested)
  viz.py      pygame window
  cli.py      commands
songs.toml    songs and their dance practice videos
tests/        uv run pytest  (test_brain.py loads the real connectome; -m "not slow" skips it)
```

## Credits

Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory
of Molecular Biology and Google Research, [CC BY 4.0](https://male-cns.janelia.org/download/).
Berg, S. et al. (2026), *Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system*, *Cell*. Simulation: [flybrain](https://pypi.org/project/flybrain/) (MIT),
neuron model after [Fly64](https://github.com/ornata/fly).
