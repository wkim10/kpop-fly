import math

import numpy as np
import pytest

from kpop_fly.blend import _REST, HIT, MAX_SWING, MODES, QUIET, SNAP, Blender, compose
from kpop_fly.brain import CHANNELS
from kpop_fly.choreo import KP
from kpop_fly.fly import limb_key, skeleton, targets
from kpop_fly.retarget import REST, retarget

NEUTRAL = np.array([0.0, 0.37])
QUIET_BRAIN = dict.fromkeys(CHANNELS, 0.0)
BURST = dict.fromkeys(CHANNELS, 1.0)


def arm_up() -> dict[str, float]:
    h = np.array([REST[k] for k in KP], float)
    h[KP["left_elbow"]], h[KP["left_wrist"]] = (0.75, 0.95), (1.2, 1.3)       # arm out to the side
    return retarget(h, NEUTRAL)


def elevation(goals: dict, part: str = "front", side: int = 1, seg: str = "u") -> float:
    """Degrees above hanging straight down."""
    return math.degrees(math.atan2(goals[limb_key(part, side, f"{seg}y")], goals[limb_key(part, side, f"{seg}x")])) + 90


def test_brain_mode_is_phase_one():
    goals, k = compose("brain", BURST, 0.5, arm_up())
    assert goals == targets(BURST, 0.5) and k is None


def test_dance_mode_ignores_the_brain():
    choreo = arm_up()
    quiet, k1 = compose("dance", QUIET_BRAIN, 0.0, choreo)
    burst, k2 = compose("dance", BURST, 1.0, choreo)
    assert quiet == burst and k1 is k2 is None
    assert quiet["wings"] == quiet["antenna"] == quiet["mid"] == 0.0


def test_blend_scales_moves_with_the_brain():
    choreo = arm_up()
    rest = elevation(_REST)                                               # scaled around the neutral pose
    dance = elevation(choreo) - rest
    quiet, k_quiet = compose("blend", QUIET_BRAIN, 0.0, choreo)
    burst, k_burst = compose("blend", BURST, 1.0, choreo)
    assert elevation(quiet) - rest == pytest.approx(dance * QUIET, rel=0.01)
    assert elevation(burst) - rest == pytest.approx(dance * HIT, rel=0.01)
    assert (k_quiet, k_burst) == SNAP                                     # loose when quiet, snaps on a burst
    assert burst["wings"] == pytest.approx(1.0) and quiet["wings"] == 0.0


def test_neutral_pose_stays_neutral_but_knees_bounce():
    rest = retarget(np.array([REST[k] for k in KP], float), NEUTRAL)
    burst, _ = compose("blend", BURST, 1.0, rest)
    assert elevation(burst, "front", -1) == pytest.approx(elevation(rest, "front", -1), abs=0.5)
    sk_rest = skeleton(compose("blend", QUIET_BRAIN, 0.0, rest)[0], 300, 600)
    sk_burst = skeleton(burst, 300, 600)
    for leg in (4, 5):                                                    # hind legs
        hip_rest, knee_rest = sk_rest.legs[leg][:2]
        hip_burst, knee_burst = sk_burst.legs[leg][:2]
        assert abs(knee_burst[0] - hip_burst[0]) > abs(knee_rest[0] - hip_rest[0]) + 15   # knees flare out
        assert hip_burst[1] > hip_rest[1] + 5                                               # hips drop


def test_exaggeration_is_capped_so_limbs_dont_wrap_around():
    h = np.array([REST[k] for k in KP], float)
    h[KP["left_elbow"]], h[KP["left_wrist"]] = (0.35, 1.55), (0.36, 2.1)     # arm straight up, ~174 from neutral
    burst, _ = compose("blend", BURST, 1.0, retarget(h, NEUTRAL))
    assert elevation(burst) - elevation(_REST) == pytest.approx(MAX_SWING, abs=0.5)


def test_contrast_uses_the_recent_range():
    b = Blender(window_s=2.0)
    rng = np.random.default_rng(0)
    for _ in range(100):                                                 # a busy song: levels 0.3..0.6
        b.contrast(dict.fromkeys(CHANNELS, float(rng.uniform(0.3, 0.6))), 1 / 30)
    assert b.contrast(dict.fromkeys(CHANNELS, 0.62), 1 / 30)["all"] > 0.95
    assert b.contrast(dict.fromkeys(CHANNELS, 0.3), 1 / 30)["all"] < 0.1


def test_contrast_does_not_amplify_silence():
    b = Blender()
    for _ in range(100):
        out = b.contrast(dict.fromkeys(CHANNELS, 0.02), 1 / 30)
    assert out["all"] < 0.2


def test_blender_only_contrasts_blend_mode():
    b = Blender()
    levels = dict.fromkeys(CHANNELS, 0.4)
    for m in MODES:
        goals, _ = b(m, levels, 0.0, arm_up(), 1 / 30)
        if m == "brain":
            assert goals == targets(levels, 0.0)
