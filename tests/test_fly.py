import math

import numpy as np
import pytest

from kpop_fly.fly import POSE_KEYS, Dancer, Spring, skeleton, targets

REST = dict.fromkeys(["all", "front", "mid", "hind", "left", "right"], 0.0)


def points(sk):
    pts = [*sk.thorax, *sk.abdomen, sk.head[0]]
    for group in (sk.wings, sk.legs, sk.eyes, sk.antennae, sk.stripes):
        for poly in group:
            pts += poly
    return np.array(pts)


def test_rest_pose_is_symmetric_and_standing():
    sk = skeleton(dict.fromkeys(POSE_KEYS, 0.0), cx=300, ground=600)
    pts = points(sk)
    assert np.isfinite(pts).all()
    xs = pts[:, 0] - 300
    assert xs.min() == pytest.approx(-xs.max(), abs=1e-6)
    hind_feet = [leg[2] for leg in sk.legs[4:]]
    assert all(foot[1] == pytest.approx(600) for foot in hind_feet)


@pytest.mark.parametrize("value", [-0.3, 0.5, 1.4])        # springs overshoot past 0..1
def test_extreme_poses_stay_finite(value):
    sk = skeleton(dict.fromkeys(POSE_KEYS, value), cx=300, ground=600, scale=1.35)
    assert np.isfinite(points(sk)).all()


def test_raised_arms_go_up():
    down = skeleton(dict.fromkeys(POSE_KEYS, 0.0), 300, 600)
    up = skeleton({**dict.fromkeys(POSE_KEYS, 0.0), "arm_l": 1.0, "arm_r": 1.0}, 300, 600)
    for i in (0, 1):
        assert up.legs[i][2][1] < down.legs[i][2][1] - 50       # screen y grows downward


def test_left_brain_leans_left():
    t = targets({**REST, "left": 0.8, "right": 0.1, "front": 0.5}, drive=0.0)
    assert t["lean"] > 0 and t["arm_l"] > t["arm_r"]


def test_spring_overshoots_then_settles():
    s = Spring()
    xs = [s.update(1.0, 1 / 60) for _ in range(240)]
    assert max(xs) > 1.0
    assert xs[-1] == pytest.approx(1.0, abs=1e-3)


def test_dancer_moves_on_a_burst():
    d = Dancer()
    pose = d.update({**REST, "all": 1.0, "front": 1.0}, drive=1.0, dt=0.05)
    assert pose["arm_l"] > 0 and pose["nod"] > 0 and not any(math.isnan(v) for v in pose.values())
