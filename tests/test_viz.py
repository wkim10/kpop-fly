"""The window, rendered off-screen. Loads the real connectome and the extracted FANCY choreography."""
import pytest

from kpop_fly.choreo import CHOREO_DIR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (CHOREO_DIR / "fancy.npz").exists(), reason="run `kpop-fly extract fancy` first")]


@pytest.fixture(scope="module")
def display():
    from kpop_fly.brain import ListeningBrain
    from kpop_fly.engine import Pipeline
    from kpop_fly.retarget import ChoreoTrack
    from kpop_fly.viz import Display

    track = ChoreoTrack.load("fancy")
    d = Display(Pipeline(ListeningBrain(log=lambda *_: None), 48000), track.title, headless=True, choreo=track)
    yield d
    d.close()


def test_m_cycles_the_modes(display):
    pg = display.pg
    seen = [display.mode]
    for _ in range(3):
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_m))
        assert display.frame(dt=1 / 30, t=99.0)
        seen.append(display.mode)
    assert seen == ["blend", "dance", "brain", "blend"]


def test_q_closes_the_window(display):
    pg = display.pg
    pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_q))
    assert display.frame(dt=1 / 30, t=99.0) is False
