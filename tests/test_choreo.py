import numpy as np
import pytest

from kpop_fly.audio import click_track
from kpop_fly.choreo import (KP, ONSET_RATE, Choreo, consensus, dancer_mask, load_songs, normalize,
                             onset_envelope, sample_frames)

W, H = 1280, 720


def dancer(x: float, y: float, size: float, arm_up: bool = False) -> np.ndarray:
    """A 17-keypoint pixel skeleton standing at (x, y) = hip center, torso `size` px tall."""
    body = {  # body-relative, y up, torso = 1
        "nose": (0, 1.45), "left_eye": (0.08, 1.52), "right_eye": (-0.08, 1.52), "left_ear": (0.15, 1.5),
        "right_ear": (-0.15, 1.5), "left_shoulder": (0.4, 1), "right_shoulder": (-0.4, 1),
        "left_elbow": (0.55, 0.45), "right_elbow": (-0.55, 0.45), "left_wrist": (0.6, -0.05),
        "right_wrist": (-0.6, -0.05), "left_hip": (0.25, 0), "right_hip": (-0.25, 0),
        "left_knee": (0.28, -0.9), "right_knee": (-0.28, -0.9), "left_ankle": (0.3, -1.8), "right_ankle": (-0.3, -1.8),
    }
    if arm_up:
        body["left_elbow"], body["left_wrist"] = (0.6, 1.6), (0.7, 2.2)
    kp = np.array([body[k] for k in KP], float)
    # dancers face the camera: their left is screen right; screen y grows down
    return np.stack([x + kp[:, 0] * size, y - kp[:, 1] * size], axis=1)


def group(n: int, arm_up_for: set[int] = frozenset(), mirror_at_edge: bool = False):
    kp = [dancer(200 + 100 * i, 500, 45 + 3 * i, i in arm_up_for) for i in range(n)]
    if mirror_at_edge:
        kp.append(dancer(20, 500, 45, arm_up=True))
    kp = np.stack(kp)
    return kp, np.full(kp.shape[:2], 0.9)


def test_normalize_is_position_and_size_invariant():
    a, _, _ = normalize(dancer(300, 400, 40)[None])
    b, hip, torso = normalize(dancer(900, 600, 90)[None])
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(hip[0], (900, 600))
    assert torso[0] == pytest.approx(90)
    assert b[0, KP["nose"], 1] > 0                          # y is up


def test_consensus_is_the_majority_move():
    kp, sc = group(7, arm_up_for={1, 3, 4, 6})              # 4 of 7 raise the left arm
    pose, visible, spread = consensus(kp, sc, np.ones(7, bool))
    assert pose[KP["left_wrist"], 1] == pytest.approx(2.2)
    assert pose[KP["right_wrist"], 1] == pytest.approx(-0.05)
    assert visible.min() == 1.0 and spread >= 0


def test_edge_reflections_are_rejected():
    kp, sc = group(5, mirror_at_edge=True)
    mask = dancer_mask(kp, sc, W, H)
    assert mask[:5].all() and not mask[5]


def test_unsure_and_tiny_dancers_are_rejected():
    kp, sc = group(3)
    sc[0, KP["left_hip"]] = 0.1                             # torso not clearly seen
    kp[1] = dancer(500, 500, 10)                            # far too small to trust
    assert dancer_mask(kp, sc, W, H).tolist() == [False, False, True]


def test_unseen_keypoints_are_nan_not_zero():
    kp, sc = group(3)
    sc[:, KP["left_ankle"]] = 0.1
    pose, visible, _ = consensus(kp, sc, np.ones(3, bool))
    assert np.isnan(pose[KP["left_ankle"]]).all() and visible[KP["left_ankle"]] == 0
    pose, _, spread = consensus(kp, sc, np.zeros(3, bool))
    assert np.isnan(pose).all() and np.isnan(spread)


@pytest.mark.parametrize("fps,target,expected", [(59.94, 30, 29.97), (24, 30, 24), (30, 30, 30)])
def test_sample_frames(fps, target, expected):
    n = int(fps * 10)
    idx = sample_frames(fps, n, target)
    assert len(idx) / 10 == pytest.approx(expected, rel=0.01)
    assert idx[0] == 0 and idx[-1] < n and (np.diff(idx) > 0).all()


def test_onset_envelope_marks_beats_at_brain_rate():
    sr = 44100
    env = onset_envelope(np.tile(click_track(120, sr)[:, 0], 2), sr)      # 8 beats over 4 s
    assert len(env) == 4 * ONSET_RATE
    for beat in np.arange(0.5, 4.0, 0.5):                  # the first beat is lost to the detector's warm-up
        i = int(beat * ONSET_RATE)
        assert env[i - 1:i + 3].max() > 0.9, f"no onset at {beat} s"
    assert env[int(0.3 * ONSET_RATE):int(0.45 * ONSET_RATE)].max() < 0.3   # quiet between beats


def test_choreo_round_trip(tmp_path):
    T = 5
    c = Choreo(t=np.arange(T, dtype=np.float32) / 30, pose=np.zeros((T, 17, 2), np.float32),
               visible=np.ones((T, 17), np.float32), spread=np.zeros(T, np.float32),
               n_dancers=np.full(T, 9, np.int16), people=np.full((T, 12, 17, 3), np.nan, np.float16),
               onset=np.zeros(10, np.float16), meta={"slug": "x", "sample_fps": 30.0})
    c.save(tmp_path / "x.npz")
    d = Choreo.load(tmp_path / "x.npz")
    assert d.meta == c.meta and d.fps == 30.0
    np.testing.assert_array_equal(d.n_dancers, c.n_dancers)


def test_songs_file_lists_the_three_twice_songs():
    songs = load_songs()
    assert {"fancy", "tt", "cheer-up"} <= songs.keys()
    assert all(len(s["youtube"]) == 11 for s in songs.values())
