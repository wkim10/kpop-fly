"""Choreography extraction: a dance practice video in, one consensus dancer out.

  1. Download the video (720p, video only) and the audio track from YouTube.
  2. Find every dancer in every sampled frame (YOLOX-tiny person detector) and estimate their
     17 COCO keypoints (RTMPose-m, on CoreML when available). MediaPipe's pose landmarker only
     finds 0-2 of 9 small, distant dancers, and 1.0.x aborts on macOS, so it isn't used.
  3. Normalize each dancer to their own body (hip center at the origin, torso length 1, y up)
     and take the per-keypoint median across dancers. Group choreography is mostly unison, so
     the median is the move; a mirror reflection or a dancer off-count gets outvoted.
  4. Save choreo/<slug>.npz (poses + the audio's onset envelope, for syncing later) and
     choreo/<slug>.opus (the song, for playback). The video itself is deleted.

Coordinates in the consensus are the dancer's own: "left_wrist" is their left hand, which
appears on screen right because they face the camera.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle")
KP = {name: i for i, name in enumerate(KEYPOINTS)}
BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_hip", "left_knee"),
         ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
TORSO = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]

MODELS = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DETECTOR = MODELS + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
POSE = MODELS + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
MAX_PEOPLE = 12
ONSET_RATE = 50                   # onset envelope samples per second: one per 20 ms brain step
FORMAT_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]
SONGS_FILE = ROOT / "songs.toml"
CHOREO_DIR = ROOT / "choreo"


def load_songs(path: Path = SONGS_FILE) -> dict[str, dict]:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---- consensus -------------------------------------------------------------------------------

def normalize(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,17,2) pixel keypoints -> body-relative coords (hip center origin, torso length 1, y up),
    plus each dancer's hip center (px) and torso length (px)."""
    hip = (kp[:, KP["left_hip"]] + kp[:, KP["right_hip"]]) / 2
    shoulder = (kp[:, KP["left_shoulder"]] + kp[:, KP["right_shoulder"]]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    rel = (kp - hip[:, None]) / np.maximum(torso, 1e-6)[:, None, None]
    rel[..., 1] *= -1
    return rel, hip, torso


def dancer_mask(kp: np.ndarray, sc: np.ndarray, width: int, height: int,
                min_torso: float = 0.04, edge: float = 0.04, min_score: float = 0.3) -> np.ndarray:
    """Which detections are usable dancers: torso clearly seen, big enough, not at the frame edge
    (where practice-room wall mirrors show reflections)."""
    if not len(kp):
        return np.zeros(0, bool)
    _, hip, torso = normalize(kp)
    return ((sc[:, TORSO].min(1) >= min_score) & (torso >= min_torso * height)
            & (hip[:, 0] >= edge * width) & (hip[:, 0] <= (1 - edge) * width))


def consensus(kp: np.ndarray, sc: np.ndarray, valid: np.ndarray, kp_score: float = 0.3):
    """Median body-relative pose across valid dancers.

    Returns (pose (17,2) with NaN where nobody saw the keypoint, visible (17,) fraction of dancers
    that saw it, spread: median distance of dancers from the consensus, in torso lengths)."""
    nan = np.full((len(KEYPOINTS), 2), np.nan)
    if not valid.any():
        return nan, np.zeros(len(KEYPOINTS)), np.nan
    rel, _, _ = normalize(kp[valid])
    seen = sc[valid] >= kp_score
    rel[~seen] = np.nan
    visible = seen.mean(0)
    if not seen.any():
        return nan, visible, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns are expected
        pose = np.nanmedian(rel, axis=0)
        dist = np.linalg.norm(rel - pose, axis=2)
        spread = float(np.nanmedian(dist)) if np.isfinite(dist).any() else np.nan
    return pose, visible, spread


def onset_envelope(samples: np.ndarray, sr: int) -> np.ndarray:
    """Onset strength per 20 ms (the brain's step), from the same detector the live loop uses."""
    from .beat import OnsetDetector
    det = OnsetDetector(sr)
    events = det.process(samples)
    n = int(len(samples) / sr * ONSET_RATE)
    env = np.zeros(n, np.float32)
    for t, s, _ in events:
        i = min(n - 1, int(t * ONSET_RATE))
        env[i] = max(env[i], s)
    return env


def sample_frames(fps: float, n_frames: int, target_fps: float) -> np.ndarray:
    """Indices of the frames to analyze so samples land as close to target_fps as the video allows."""
    if fps <= target_fps * 1.05:
        return np.arange(n_frames)
    step = fps / target_fps
    return np.unique(np.round(np.arange(0, n_frames - 0.5, step)).astype(int))


# ---- the saved file ---------------------------------------------------------------------------

@dataclass
class Choreo:
    t: np.ndarray              # (T,) seconds into the video
    pose: np.ndarray           # (T,17,2) consensus, body-relative (torso = 1, y up); NaN = unseen
    visible: np.ndarray        # (T,17) fraction of dancers who showed that keypoint
    spread: np.ndarray         # (T,) how much the dancers disagree (torso lengths); high = not unison
    n_dancers: np.ndarray      # (T,) usable dancers found
    people: np.ndarray         # (T,MAX_PEOPLE,17,3) float16 raw px x, y, score; NaN padded
    onset: np.ndarray          # (N,) audio onset strength at ONSET_RATE Hz, same clock as t
    meta: dict

    @property
    def fps(self) -> float:
        return self.meta["sample_fps"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items() if k != "meta"}
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: Path) -> Choreo:
        with np.load(path) as z:
            return cls(meta=json.loads(str(z["meta"])), **{k: z[k] for k in z.files if k != "meta"})


# ---- extraction -------------------------------------------------------------------------------

def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found; install it (e.g. `brew install {tool}`)")
    return path


def download(youtube: str, workdir: Path, log: Callable[[str], None]) -> tuple[Path, Path, dict]:
    """Video-only (<=720p, H.264 so OpenCV decodes it) and audio-only streams; no merging needed."""
    from yt_dlp import YoutubeDL
    url = youtube if youtube.startswith("http") else f"https://www.youtube.com/watch?v={youtube}"
    common = {"quiet": True, "no_warnings": True, "noprogress": True}
    with YoutubeDL({**common, "format": "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]",
                    "outtmpl": str(workdir / "video.%(ext)s")}) as y:
        log(f"downloading video {url}")
        info = y.extract_info(url, download=True)
        video = Path(y.prepare_filename(info))
    with YoutubeDL({**common, "format": "ba[acodec=opus]/ba", "outtmpl": str(workdir / "audio.%(ext)s")}) as y:
        log("downloading audio")
        audio = Path(y.prepare_filename(y.extract_info(url, download=True)))
    keep = ("id", "title", "channel", "upload_date", "duration", "webpage_url")
    return video, audio, {k: info.get(k) for k in keep}


def save_audio(src: Path, dst: Path) -> None:
    """Rewrap YouTube's Opus into an .opus (Ogg) file libsndfile can read; transcode otherwise."""
    ffmpeg = _need("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for codec in (["-c:a", "copy"], ["-c:a", "libopus", "-b:a", "128k"]):
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn", *codec, str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError(f"ffmpeg couldn't convert {src}: {r.stderr}")


class PoseEstimator:
    def __init__(self, device: str = "mps"):
        import os
        os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
        from rtmlib import RTMPose, YOLOX
        self.detector = YOLOX(DETECTOR, model_input_size=(416, 416), backend="onnxruntime", device="cpu")
        self.pose = RTMPose(POSE, model_input_size=(192, 256), backend="onnxruntime", device=device)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes = self.detector(frame)
        if not len(boxes):
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        return self.pose(frame, boxes)


def extract_poses(video: Path, target_fps: float, log: Callable[[str], None],
                  preview: Callable | None = None, max_seconds: float | None = None) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = sample_frames(fps, n, target_fps)
    if max_seconds:
        wanted = wanted[wanted / fps < max_seconds]
    estimator = PoseEstimator()
    T = len(wanted)
    out = {"t": wanted / fps, "pose": np.full((T, 17, 2), np.nan, np.float32),
           "visible": np.zeros((T, 17), np.float32), "spread": np.full(T, np.nan, np.float32),
           "n_dancers": np.zeros(T, np.int16), "people": np.full((T, MAX_PEOPLE, 17, 3), np.nan, np.float16)}
    span = (wanted[-1] - wanted[0] + 1) / fps if T else 0
    log(f"{video.name}: {width}x{height} @ {fps:.2f} fps, analyzing {T} frames ({span:.0f} s at {T / max(span, 1e-9):.1f} fps)")

    t0, last_log, idx = time.perf_counter(), 0.0, 0
    for frame_no in range(n):
        if idx >= T:
            break
        if frame_no != wanted[idx]:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        kp, sc = estimator(frame)
        valid = dancer_mask(kp, sc, width, height)
        out["pose"][idx], out["visible"][idx], out["spread"][idx] = consensus(kp, sc, valid)
        out["n_dancers"][idx] = valid.sum()
        order = np.argsort(kp[:, :, 0].mean(1))[:MAX_PEOPLE] if len(kp) else []
        for j, p in enumerate(order):
            out["people"][idx, j, :, :2], out["people"][idx, j, :, 2] = kp[p], sc[p]
        if preview:
            preview(frame, kp, sc, valid, out["pose"][idx], idx, out)
        idx += 1
        now = time.perf_counter()
        if now - last_log > 10:
            last_log = now
            rate = idx / (now - t0)
            log(f"  {idx}/{T} frames ({rate:.1f}/s, ~{(T - idx) / rate / 60:.1f} min left)")
    cap.release()
    for k in ("pose", "visible", "spread", "n_dancers", "people"):
        out[k] = out[k][:idx]
    out["t"] = out["t"][:idx]
    out.update(width=width, height=height, video_fps=fps, seconds=time.perf_counter() - t0)
    return out


def extract(slug: str, youtube: str, song: dict | None = None, target_fps: float = 30.0,
            out_dir: Path = CHOREO_DIR, preview: bool = False, max_seconds: float | None = None,
            log: Callable[[str], None] = print) -> Path:
    _need("ffmpeg")
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="kpop-fly-") as tmp:
        video, audio, info = download(youtube, Path(tmp), log)
        audio_out = out_dir / f"{slug}.opus"
        save_audio(audio, audio_out)
        samples, sr = sf.read(str(audio_out), dtype="float32", always_2d=True)
        onset = onset_envelope(samples.mean(1), sr)

        writer = PreviewWriter(out_dir / f"{slug}.preview.mp4", audio_out) if preview else None
        try:
            poses = extract_poses(video, target_fps, log, writer.frame if writer else None, max_seconds)
        finally:
            if writer:
                writer.close()
        # the temporary directory, and the video in it, are deleted here

    meta = {"format": FORMAT_VERSION, "slug": slug, **(song or {}), "source": info,
            "keypoints": list(KEYPOINTS), "detector": DETECTOR.rsplit("/", 1)[1], "pose_model": POSE.rsplit("/", 1)[1],
            "sample_fps": len(poses["t"]) / (poses["t"][-1] - poses["t"][0]) if len(poses["t"]) > 1 else target_fps,
            "video_fps": poses["video_fps"], "frame_size": [poses["width"], poses["height"]],
            "onset_rate": ONSET_RATE, "audio": audio_out.name, "audio_seconds": len(samples) / sr,
            "extracted": time.strftime("%Y-%m-%d"), "extract_seconds": round(poses["seconds"], 1)}
    choreo = Choreo(t=poses["t"].astype(np.float32), pose=poses["pose"], visible=poses["visible"],
                    spread=poses["spread"], n_dancers=poses["n_dancers"], people=poses["people"],
                    onset=onset.astype(np.float16), meta=meta)
    path = out_dir / f"{slug}.npz"
    choreo.save(path)
    return path


def summarize(c: Choreo) -> str:
    found = np.isfinite(c.pose[:, TORSO, 0]).all(1)
    limbs = [KP[k] for k in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")]
    limb_seen = np.isfinite(c.pose[:, limbs, 0]).all(1)
    return (f"{c.meta.get('artist', '')} {c.meta.get('title', c.meta['slug'])}: {len(c.t)} poses over "
            f"{c.t[-1] - c.t[0]:.0f} s at {c.fps:.1f} fps; consensus found in {found.mean():.0%} of frames "
            f"(wrists+ankles in {limb_seen.mean():.0%}); dancers per frame median {int(np.median(c.n_dancers))}; "
            f"disagreement median {np.nanmedian(c.spread):.2f} torso lengths")


# ---- preview video ----------------------------------------------------------------------------

class PreviewWriter:
    """Side-by-side check video: the practice footage with every detected skeleton (valid dancers
    colored, rejected ones grey) next to the consensus dancer, with the song as its soundtrack."""
    W, H = 640, 360

    def __init__(self, path: Path, audio: Path):
        self.path = path
        self._raw = path.with_suffix(".raw.mp4")
        self._audio = audio
        self._proc = None
        self._fps = None

    def frame(self, frame, kp, sc, valid, pose, idx, out) -> None:
        import cv2
        if self._proc is None:
            t = out["t"]
            self._fps = 1 / float(np.median(np.diff(t[:50]))) if len(t) > 1 else 30
            self._proc = subprocess.Popen(
                [_need("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{2 * self.W}x{self.H}", "-r", f"{self._fps:.4f}", "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", str(self._raw)],
                stdin=subprocess.PIPE)
        sx, sy = self.W / frame.shape[1], self.H / frame.shape[0]
        left = cv2.resize(frame, (self.W, self.H))
        for p in range(len(kp)):
            color = (90, 220, 255) if valid[p] else (150, 150, 150)
            for a, b in BONES:
                a, b = KP[a], KP[b]
                if sc[p, a] >= 0.3 and sc[p, b] >= 0.3:
                    cv2.line(left, (int(kp[p, a, 0] * sx), int(kp[p, a, 1] * sy)),
                             (int(kp[p, b, 0] * sx), int(kp[p, b, 1] * sy)), color, 2)
        right = np.full((self.H, self.W, 3), (34, 22, 24), np.uint8)
        scale, cx, cy = 62, self.W // 2, int(self.H * 0.55)
        for a, b in BONES:
            pa, pb = pose[KP[a]], pose[KP[b]]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                color = (170, 92, 255) if "left" in a and "left" in b else (240, 220, 80) if "right" in a and "right" in b else (235, 235, 235)
                cv2.line(right, (int(cx + pa[0] * scale), int(cy - pa[1] * scale)),
                         (int(cx + pb[0] * scale), int(cy - pb[1] * scale)), color, 4)
        if np.isfinite(pose[KP["nose"]]).all():
            n = pose[KP["nose"]]
            cv2.circle(right, (int(cx + n[0] * scale), int(cy - n[1] * scale)), 14, (235, 235, 235), 3)
        cv2.putText(right, f"consensus of {int(valid.sum())} dancers   t={out['t'][idx]:.1f}s", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(right, "pink = dancer's left   cyan = dancer's right", (12, self.H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        self._proc.stdin.write(np.hstack([left, right]).tobytes())

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.stdin.close()
        self._proc.wait()
        subprocess.run([_need("ffmpeg"), "-y", "-loglevel", "error", "-i", str(self._raw), "-i", str(self._audio),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(self.path)], check=True)
        self._raw.unlink(missing_ok=True)
