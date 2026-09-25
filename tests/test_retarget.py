import math

import numpy as np
import pytest

from kpop_fly.choreo import KP, Choreo
from kpop_fly.fly import limb_key, skeleton
from kpop_fly.retarget import REST, ChoreoTrack, fill_gaps, retarget, smooth

NEUTRAL = np.array([0.0, 0.37])


def human(**moves) -> np.ndarray:
    """REST pose (body-relative, y up, facing camera) with some keypoints moved."""
    h = np.array([REST[k] for k in KP], float)
    for name, xy in moves.items():
        h[KP[name]] = xy
    return h


def test_rest_pose_is_a_symmetric_standing_fly():
    g = retarget(human(), NEUTRAL)
    for side in (-1, 1):
        assert g[limb_key("front", side, "uy")] < -0.9                  # arms hang down
        assert g[limb_key("hind", side, "uy")] < -0.9 and g[limb_key("hind", side, "ly")] < -0.9   # straight legs
    assert g[limb_key("front", -1, "ux")] == pytest.approx(-g[limb_key("front", 1, "ux")])
    assert g[limb_key("hind", -1, "ux")] == pytest.approx(-g[limb_key("hind", 1, "ux")])
    assert g["torso"] == pytest.approx(0) and g["head_roll"] == pytest.approx(0, abs=1e-6)


def test_dancers_left_arm_raises_the_screen_right_front_leg():
    g = retarget(human(left_elbow=(0.6, 1.4), left_wrist=(0.7, 2.0)), NEUTRAL)
    assert g[limb_key("front", 1, "uy")] > 0.5 and g[limb_key("front", 1, "ly")] > 0.5
    assert g[limb_key("front", -1, "uy")] < -0.9                          # the other arm stays down


def test_turning_around_does_not_cross_the_arms():
    back = human(left_elbow=(0.6, 1.4), left_wrist=(0.7, 2.0))
    for part in ("shoulder", "elbow", "wrist"):                           # from behind, the dancer's left is on screen left
        back[[KP[f"left_{part}"], KP[f"right_{part}"]]] = back[[KP[f"right_{part}"], KP[f"left_{part}"]]]
    g = retarget(back, NEUTRAL)
    assert g[limb_key("front", 1, "uy")] > 0.5                            # the raised arm is still the screen-right one
    assert g[limb_key("front", -1, "ux")] < 0 and g[limb_key("front", 1, "ux")] > 0   # neither arm crosses over


def test_torso_lean_and_head():
    tilt = human(left_shoulder=(0.49, 0.98), right_shoulder=(-0.09, 1.0), nose=(0.2, 1.35))
    assert retarget(tilt, NEUTRAL)["torso"] > 5                           # shoulders over to screen right
    assert retarget(human(nose=(0, 1.15)), NEUTRAL)["nod"] > 0.9           # head dropped
    rolled = retarget(human(left_ear=(0.1, 1.5), right_ear=(-0.1, 1.3)), NEUTRAL)["head_roll"]
    assert 30 < rolled <= 35


def test_straight_legs_stay_straight_and_bent_knees_bend():
    straight = skeleton(retarget(human(), NEUTRAL), cx=300, ground=600).legs[4]
    hip, knee, foot = (np.array(p) for p in straight[:3])
    assert np.linalg.norm(foot - hip) == pytest.approx(72 + 82, rel=0.02)
    squat = human(left_knee=(0.6, -0.4), right_knee=(-0.6, -0.4), left_ankle=(0.2, -1.0), right_ankle=(-0.2, -1.0))
    bent = skeleton(retarget(squat, NEUTRAL), cx=300, ground=600).legs[4]
    hip, knee, foot = (np.array(p) for p in bent[:3])
    assert np.linalg.norm(foot - hip) < 0.8 * (72 + 82) and knee[0] < hip[0]      # knee out to screen left


def test_a_kick_lifts_one_foot_and_keeps_the_other_on_the_floor():
    g = retarget(human(left_knee=(0.6, -0.4), left_ankle=(1.0, -0.9)), NEUTRAL)
    sk = skeleton(g, cx=300, ground=600)
    standing, kicking = sk.legs[4][2], sk.legs[5][2]                     # hind feet: screen left, right
    assert standing[1] == pytest.approx(600)
    assert kicking[1] < 560 and kicking[0] > standing[0] + 100


@pytest.mark.parametrize("kp", ["left_wrist", "right_ankle", "nose"])
def test_extreme_poses_give_finite_geometry(kp):
    wild = human(**{kp: (3.0, 3.0)})
    sk = skeleton(retarget(wild, NEUTRAL), cx=300, ground=600, scale=1.35)
    pts = np.array([p for leg in sk.legs for p in leg] + sk.thorax + [sk.head[0]])
    assert np.isfinite(pts).all()


def test_fill_gaps_interpolates_and_holds_ends():
    t = np.arange(5.0)
    pose = np.tile(np.array([REST[k] for k in KP], float), (5, 1, 1))
    pose[:, KP["nose"], 1] = [np.nan, 1.0, np.nan, 2.0, np.nan]
    pose[:, KP["left_ear"]] = np.nan                                      # never seen
    out = fill_gaps(t, pose)
    assert out[:, KP["nose"], 1].tolist() == [1.0, 1.0, 1.5, 2.0, 2.0]
    assert out[:, KP["left_ear"]].tolist() == [list(REST["left_ear"])] * 5


def test_smoothing_keeps_still_poses_still():
    pose = np.tile(np.array([REST[k] for k in KP], float), (20, 1, 1))
    np.testing.assert_allclose(smooth(pose), pose)


def test_track_interpolates_between_samples(tmp_path):
    T = 4
    pose = np.tile(np.array([REST[k] for k in KP], np.float32), (T, 1, 1))
    pose[2:, KP["left_wrist"], 1] = 2.0
    c = Choreo(t=np.arange(T, dtype=np.float32) / 2, pose=pose, visible=np.ones((T, 17), np.float32),
               spread=np.zeros(T, np.float32), n_dancers=np.full(T, 9, np.int16),
               people=np.full((T, 1, 17, 3), np.nan, np.float16), onset=np.zeros(8, np.float16),
               meta={"slug": "x", "artist": "A", "title": "B", "audio": "x.opus", "sample_fps": 2.0})
    c.save(tmp_path / "x.npz")
    track = ChoreoTrack.load(tmp_path / "x.npz")
    assert track.title == "A - B" and track.audio == tmp_path / "x.opus"
    a, b = track.human_at(0.5)[KP["left_wrist"], 1], track.human_at(1.0)[KP["left_wrist"], 1]
    assert a < track.human_at(0.75)[KP["left_wrist"], 1] < b              # rises smoothly in between
    assert not math.isnan(track.targets_at(99.0)["torso"])                # past the end: holds the last pose
