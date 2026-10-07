# kpop-fly

A fruit fly that dances to K-pop, and whose moves are timed by a simulation of its real brain.

Sound drives the antennal hearing neurons (Johnston's organ, JO) of the complete male fruit fly
central nervous system, the [MaleCNS v1.0 connectome](https://male-cns.janelia.org) (166,700
neurons, 25.6 M synapses), simulated as a spiking network by [flybrain](https://github.com/alextitonis/fly.ai).
The descending neurons (DNs), the brain's commands to the body, are read out every 20 ms and move
a cartoon fly.

**Phase 1:** brain → beat. Any audio → onsets → antennae → connectome → DNs → 2D fly.
**Phase 2 (in progress):** choreography from dance practice videos, performed with the brain's timing.
Done: extraction (step 1), retargeting onto the fly (step 2), blending the brain into the moves
(step 3), YouTube links as input (step 4), and live song recognition from the microphone (step 5).

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
uv run kpop-fly mic                  # listen, recognize the song, dance it (allow mic access for your terminal)
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
always match. The fly's arms, legs, torso and head copy the song's dancer (shown small in the
corner): upper-arm/forearm and thigh/shin directions carry over as-is, so elbows and knees bend
exactly as the dancer's do; torso tilt leans the upper body; the nose and ear line nod, shift and
roll the head. Everything stays in screen space, so the fly does what you'd see in the video.
Mid legs, wings and antennae have no human counterpart and are driven by the brain.

### Live from the microphone

```sh
export AUDD_API_TOKEN=...            # from dashboard.audd.io (300 free requests)
uv run kpop-fly mic                  # play a song anywhere near the mic
uv run kpop-fly mic --simulate choreo/tt.opus --start 60   # test without a mic: a file stands in for it
```

The fly dances to the brain alone while `mic` listens. Once it has heard 6 s of sound
(`--listen-seconds`, 5 also tested fine), it sends that clip to [AudD](https://audd.io) on a
background thread. If AudD names one of the saved songs, the clip is located in it and the fly
switches to the dance, about 7 s after the music starts. AudD's timecode lands 1.6-2.9 s before
the *end* of the clip it was sent, whatever the clip's length, so the search covers just +-2.5 s
around that. That's narrow enough that 4-7 s clips can't land on a repeated phrase (60/60 correct
at each length through simulated room audio). Locating takes two stages: chroma (which notes are
sounding) finds the bar, then the onset envelope finds the exact 20 ms frame. Rhythm alone,
through a speaker and a room, fits one beat or one bar off about half the time; harmony doesn't.

While the fly dances, about once a second it checks that the last 3 s still match the song where
it should be by now. Through a room the song scores 0.89+, room noise and other songs at most 0.31.
If that fails for ~2 s:

- **and the level dropped** (or went under the silence line): the song **stopped**. The fly goes
  back to the brain alone, about 3-4 s after the music stops, and the brain hears silence rather
  than amplified room noise. This works even when the room's noise sits above the silence line,
  which a loudness threshold alone can't handle. When sound returns, the same song is looked for
  first, so a resumed pause re-locks in ~4 s without asking AudD.
- **and it's still loud:** the same song is searched end to end first (a skip, or a lock on the
  wrong repeat of a chorus: AudD's own timecode sometimes names the other repeat). If it isn't
  there, the song **changed**, and it's recognized again from audio after the change.

Every 6 s a longer window also re-locates the song to keep the position exact. None of this costs
API requests. A song AudD names that has no saved dance shows as, for example, "TWICE - Likey
(K-pop): no saved dance", and the brain keeps dancing. `--max-requests` caps AudD requests per
session (default 50).

When AudD doesn't recognize a recording, or there's no token or no network, every saved song is
searched end to end instead. AudD fingerprints the studio recording, so it misses TV stages with
live vocals; the local search still finds them. A local match must score well and beat the next
saved song clearly (real matches by at least 0.20, other songs by at most 0.09). Without AudD's
timecode nothing tells a song's repeated sections apart, so a local lock can land on another
chorus. That happened in about 1 test clip in 10, usually where the choreography repeats too.

Tested with files standing in for the mic: TT's music video (one AudD request, then in sync to
about 0.1 s for the rest of the test), FANCY's TV stage (AudD didn't know it; recognized locally),
a FANCY-to-TT switch (noticed about 14 s after the change), LIKEY (recognized, no saved dance),
and runs without a token. With simulated laptop-speaker-to-room-to-mic audio, the two-stage
locate placed 55 of 55 clips correctly. A real microphone hasn't been tested yet.

### Any YouTube link

```sh
uv run kpop-fly url "https://www.youtube.com/watch?v=ePpPVE-GGJw"   # TWICE "TT" M/V
uv run kpop-fly url ePpPVE-GGJw --render                          # or write choreo/url-<id>-fly.mp4
```

`url` downloads the video's audio (cached in `~/.cache/kpop-fly/youtube/`), works out which saved
dance it is and where each moment falls in the song, then plays it with the fly dancing. The link
can be the dance practice video itself (recognized by its video id) or a different recording of
the song: a music video with a drama intro, a TV stage with live vocals. Moments that aren't the
song, such as an MV's intro, and links that match no saved dance, get the brain-only dance.

Matching uses the onset envelope each choreo file saved for its song: 10 s windows of the
recording are cross-correlated against it, and a Viterbi pass picks one consistent offset per
stretch. Choruses repeat, so each chorus window matches every chorus about equally well; a jump
penalty keeps the path on the true offset instead of hopping between repeats. Tested on the
three MVs and three TV stages of the saved songs (all matched as one clean segment; the MVs'
intros correctly left out) and on TWICE "What is Love?", "LIKEY" and BTS "Dynamite" (no match).
The weakest real match is the CHEER UP MV (61% of it; its mix differs most from the practice
video).

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

`extract` downloads a dance video, finds every dancer in each frame (YOLOX-tiny + RTMPose-m via
rtmlib, pose model on CoreML) and normalizes each to their own body. With several dancers it takes
the per-keypoint median, one consensus dancer; with one, it's that dancer. It writes
`choreo/<slug>.npz` (poses + the song's onset envelope for syncing) and `choreo/<slug>.opus` (the
song), then deletes the video. `--preview` also writes `choreo/<slug>.preview.mp4`: the footage
with detected skeletons next to the extracted dancer. `choreo/` is git-ignored because it's derived
from third-party videos. MediaPipe was the original plan, but it finds only 0–2 of 9 small dancers
per frame, and 1.0.x aborts on macOS ([#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)).

**Solo dancers on the studio track.** Each song in `songs.toml` now comes from one dancer: Momo's
official FANCY dance video, and Lisa Rhee's covers of TT and CHEER UP. Each also names a `studio`
track, the official audio from YouTube Music. `extract` aligns the video's soundtrack to it (to
20 ms) and moves every pose onto the studio timeline, so `choreo/<slug>.opus` is the exact
recording Spotify plays. Compared with the 9-member group practices used before (kept in
`choreo/group/`):

- **Mic matching:** room-degraded studio audio aligns at 0.95–0.96 (median) instead of 0.75–0.84.
  The CHEER UP practice video's soundtrack was a smeared copy of the song.
- **Choreography:** the solo poses carry less high-frequency jitter than the 9-dancer median
  (above 8 Hz: 0.6–2.2% of motion vs 1.4–3.1%) and keep moves the median blurred or invented in
  split formations. They agree with the group's limb directions at 0.80–0.85 as-is and 0.68–0.74
  flipped, so none is mirrored.

`mirrored = true` in `songs.toml` flips mirrored videos (tutorials) back before posing.

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
  choreo.py   dance video → choreography on the studio track's timeline (choreo/<slug>.npz)
  retarget.py the dancer's poses → fly pose targets
  blend.py    dance-only / brain-only / blended modes
  render.py   offline mp4s of the fly dancing a song, and the mode comparison
  youtube.py  audio from a YouTube link (cached)
  match.py    which saved dance a recording is, and its time map onto the song
  features.py chroma: harmony over time, for locating room audio in a song
  recognize.py  AudD client and title matching
  listen.py   live listening: recognize, lock on, stay in sync
  brain.py    connectome, calibration, per-step readout into body channels
  engine.py   Pipeline (audio → brain, clock-free), BrainThread (real time), analyze (offline)
  audio.py    mic / file / metronome sources
  fly.py      springs and 2D geometry (pure math, tested)
  viz.py      pygame window
  cli.py      commands
songs.toml    songs: their dance videos, dancers and official studio tracks
tests/        uv run pytest  (test_brain.py loads the real connectome; -m "not slow" skips it)
```

## Credits

Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory
of Molecular Biology and Google Research, [CC BY 4.0](https://male-cns.janelia.org/download/).
Berg, S. et al. (2026), *Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system*, *Cell*. Simulation: [flybrain](https://pypi.org/project/flybrain/) (MIT),
neuron model after [Fly64](https://github.com/ornata/fly).
