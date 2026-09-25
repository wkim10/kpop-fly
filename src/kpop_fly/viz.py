"""pygame window: the dancing fly on the left, the evidence on the right.

The right panel shows the chain end to end: sound onsets -> voltage into the antennae -> a
raster of the descending neurons that listen -> the channel levels that move the body.

With a choreography, the fly's arms, legs, torso and head follow the song's consensus dancer
(drawn small in the corner for reference), mixed with the brain according to the mode (see
blend.py). Press m to cycle blend -> dance -> brain.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from .brain import CHANNELS
from .engine import Pipeline
from .blend import MODES, Blender
from .choreo import BONES, KP
from .fly import Dancer, skeleton
from .retarget import ChoreoTrack

W, H = 1200, 680
STAGE_W = 620
BG = (14, 13, 22)
PANEL = (24, 22, 36)
INK = (230, 228, 240)
DIM = (130, 126, 150)
PINK = (255, 92, 170)
CYAN = (80, 220, 240)
GOLD = (255, 200, 90)
TIER_COLORS = [(255, 92, 170), (170, 120, 255), (80, 220, 240)]   # front, mid, hind
BODY, BODY_DARK, EYE, WING = (120, 88, 60), (70, 50, 36), (200, 40, 50), (200, 220, 255, 90)


class Display:
    def __init__(self, pipeline: Pipeline, source_name: str, headless: bool = False,
                 choreo: ChoreoTrack | None = None, song_time: Callable[[], float] | None = None,
                 mode: str = "blend"):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("kpop-fly")
        self.font = pygame.font.SysFont("menlo,monaco,courier", 13)
        self.big = pygame.font.SysFont("menlo,monaco,courier", 22, bold=True)
        self.pipeline = pipeline
        self.source_name = source_name
        self.dancer = Dancer()
        self.blender = Blender()
        self.clock = pygame.time.Clock()
        self.choreo = choreo
        self.mode = mode if choreo else "brain"
        self.headless = headless
        self.song_time = song_time
        self.t = 0.0
        cal = pipeline.brain.cal
        order = np.lexsort((cal.latency, cal.tier))
        self.row_order = order
        self.row_colors = np.array([TIER_COLORS[t] for t in cal.tier[order]], np.uint8)

    def frame(self, dt: float | None = None, t: float | None = None) -> bool:
        """Draw one frame, `dt` after the last, at song time `t` (default: ask `song_time`, which
        may say None: this moment isn't part of the dance). Returns False when the window closes."""
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_q)):
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_m and self.choreo:
                self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        if dt is None:
            dt = self.clock.tick(60) / 1000
        self.t = t if t is not None else self.song_time() if self.song_time else self.t + dt
        in_dance = self.choreo is not None and self.t is not None
        snap = self.pipeline.timeline.snapshot()
        latest = snap["latest"]
        levels = dict(zip(CHANNELS, latest.levels)) if latest else dict.fromkeys(CHANNELS, 0.0)
        choreo = self.choreo.targets_at(self.t) if in_dance else None
        goals, stiffness = self.blender(self.mode, levels, latest.drive if latest else 0.0, choreo, dt)
        pose = self.dancer.follow(goals, dt, stiffness)

        self.screen.fill(BG)
        self._stage(pose, snap)
        self._panel(snap, levels)
        pg.display.flip()
        return True

    def screenshot(self, path: str | Path) -> None:
        self.pg.image.save(self.screen, str(path))

    def close(self) -> None:
        self.pg.quit()

    def _stage(self, pose, snap) -> None:
        pg, s = self.pg, self.screen
        ground = H - 70
        glow = max(0.0, snap["levels"][-1, 0]) if len(snap["levels"]) else 0.0
        spot = pg.Surface((STAGE_W, H), pg.SRCALPHA)
        for r, a in ((260, 18), (190, 22), (120, 26)):
            pg.draw.ellipse(spot, (*PINK, int(a * (0.4 + glow))), (STAGE_W // 2 - r, ground - r * 1.6, 2 * r, 2 * r * 1.6))
        s.blit(spot, (0, 0))
        pg.draw.line(s, DIM, (40, ground), (STAGE_W - 40, ground), 2)
        pg.draw.ellipse(s, (8, 7, 14), (STAGE_W // 2 - 110, ground - 8, 220, 16))

        sk = skeleton(pose, STAGE_W / 2, ground, scale=1.35)
        wings = pg.Surface((W, H), pg.SRCALPHA)
        for w in sk.wings:
            pg.draw.polygon(wings, WING, w)
            pg.draw.polygon(wings, (220, 235, 255, 160), w, 2)
        s.blit(wings, (0, 0))
        for leg in sk.legs[2:]:                                     # mid and hind legs behind the body
            self._leg(leg)
        pg.draw.polygon(s, (190, 150, 70), sk.abdomen)
        for stripe in sk.stripes:
            pg.draw.line(s, BODY_DARK, *stripe, 7)
        pg.draw.polygon(s, BODY, sk.thorax)
        pg.draw.polygon(s, BODY_DARK, sk.thorax, 2)
        (hx, hy), hr = sk.head
        pg.draw.circle(s, BODY, (hx, hy), hr)
        for eye in sk.eyes:
            pg.draw.polygon(s, EYE, eye)
            pg.draw.polygon(s, (255, 120, 120), eye, 2)
        for ant in sk.antennae:
            pg.draw.lines(s, BODY_DARK, False, ant, 4)
            pg.draw.circle(s, GOLD, ant[-1], 3)
        for leg in sk.legs[:2]:                                     # front legs (arms) in front, like a dancer's
            self._leg(leg, width=5)

        s.blit(self.big.render("kpop-fly", True, INK), (24, 18))
        subtitle = f"dancing {self.choreo.title}" if self.choreo else "brain -> beat"
        s.blit(self.font.render(subtitle, True, DIM), (26, 46))
        if self.choreo:
            names = {"blend": "dance + brain", "dance": "dance only", "brain": "brain only"}
            shown = self.mode if self.t is not None else "brain"      # outside the dance only the brain moves it
            tag = self.big.render(names[shown], True, PINK if shown == "blend" else CYAN if shown == "dance" else GOLD)
            s.blit(tag, (24, 70))
            if not self.headless:
                s.blit(self.font.render("press m to switch", True, DIM), (26, 98))
        bpm = self.pipeline.detector.bpm()
        if not self.choreo:
            clock = ""
        elif self.t is None:
            clock = "not in the dance: brain only   "
        else:
            clock = f"{int(self.t // 60)}:{self.t % 60:04.1f}   "
        s.blit(self.font.render(f"{self.source_name}   {clock}{f'heard {bpm:.0f} BPM' if bpm else 'listening...'}",
                                True, INK), (26, H - 40))
        if self.choreo and self.t is not None:
            self._reference(self.choreo.human_at(self.t), self.choreo.dancers_at(self.t))

    def _leg(self, points, width: int = 4) -> None:
        self.pg.draw.lines(self.screen, BODY_DARK, False, points, width + 3)
        self.pg.draw.lines(self.screen, BODY, False, points, width)
        for joint in points[1:-1]:
            self.pg.draw.circle(self.screen, BODY_DARK, joint, width // 2 + 2)

    def _reference(self, human: np.ndarray, n_dancers: int) -> None:
        """The consensus dancer the fly is copying, small, top right of the stage."""
        pg, s = self.pg, self.screen
        cx, cy, k = STAGE_W - 80, 150, 34
        pt = lambda i: (cx + human[i, 0] * k, cy - human[i, 1] * k)
        for a, b in BONES:
            color = PINK if a.startswith("left") and b.startswith("left") else CYAN if a.startswith("right") and b.startswith("right") else DIM
            pg.draw.line(s, color, pt(KP[a]), pt(KP[b]), 2)
        pg.draw.circle(s, DIM, pt(KP["nose"]), 7, 2)
        label = self.font.render(f"the moves ({n_dancers} dancers)", True, DIM)
        s.blit(label, (min(cx - label.get_width() // 2, STAGE_W - label.get_width() - 8), cy + 72))

    def _panel(self, snap, levels) -> None:
        pg, s = self.pg, self.screen
        x0, pad = STAGE_W, 18
        pg.draw.rect(s, PANEL, (x0, 0, W - x0, H))
        x, w = x0 + pad, W - x0 - 2 * pad
        brain = self.pipeline.brain
        latest = snap["latest"]
        rt = brain.dt * 1000 / snap["step_ms"] if snap["step_ms"] else 0
        stats = (f"{brain.brain.n:,} neurons  {snap['step_ms']:.1f} ms/step ({rt:.0f}x realtime)  "
                 f"{latest.total_spikes if latest else 0:,} spikes/step")
        s.blit(self.font.render(stats, True, DIM), (x, 14))

        y = 44
        y = self._trace("sound onsets", snap["strength"], snap["beat"], x, y, w, 56, GOLD)
        y = self._trace(f"antenna drive -> {len(brain.jo)} JO neurons", snap["drive"], snap["beat"], x, y, w, 56, CYAN)

        n = brain.n_responsive
        s.blit(self.font.render(f"{n} descending neurons that respond to sound (rows; fast->slow)", True, INK), (x, y))
        y += 18
        rh = 250
        raster = snap["raster"][self.row_order]                       # (n, T)
        img = np.full((*raster.shape, 3), PANEL, np.uint8)
        img[raster] = np.broadcast_to(self.row_colors[:, None, :], img.shape)[raster]
        surf = pg.surfarray.make_surface(img.transpose(1, 0, 2))      # surfarray wants (x=time, y=row)
        s.blit(pg.transform.scale(surf, (w, rh)), (x, y))
        self._beat_ticks(snap["beat"], x, y, w, rh)
        pg.draw.rect(s, DIM, (x, y, w, rh), 1)
        y += rh + 14

        labels = {
            "blend": {"all": "all -> snap, nod", "front": "fast -> arm hits", "mid": "mid -> mid legs",
                      "hind": "slow -> knees"},
            "dance": {"all": "all (unused)", "front": "fast (unused)", "mid": "mid (unused)", "hind": "slow (unused)"},
            "brain": {"all": "all", "front": "fast -> arms", "mid": "mid -> mid legs", "hind": "slow -> knees"},
        }[self.mode] | {"left": "left brain", "right": "right brain"}
        colors = {"all": INK, "front": TIER_COLORS[0], "mid": TIER_COLORS[1], "hind": TIER_COLORS[2],
                  "left": GOLD, "right": GOLD}
        for i, ch in enumerate(CHANNELS):
            cx, cy = x + (i % 2) * (w // 2), y + (i // 2) * 24
            s.blit(self.font.render(labels[ch], True, DIM), (cx, cy))
            bw = w // 2 - 150
            pg.draw.rect(s, BG, (cx + 130, cy + 2, bw, 12))
            pg.draw.rect(s, colors[ch], (cx + 130, cy + 2, int(bw * levels[ch]), 12))
        y += 3 * 24 + 8
        foot = {
            "blend": ("the moves: the dance practice's consensus dancer.",
                      "when they land, how hard: the brain's descending neurons."),
            "dance": ("the moves: the dance practice's consensus dancer, alone.",
                      "the brain is running but not driving the fly."),
            "brain": ("when each part moves: the brain's descending neurons.",
                      "how far it moves: an artistic mapping."),
        }[self.mode] + ("wiring: MaleCNS v1.0 (Berg et al. 2026, CC BY 4.0) via flybrain",)
        for line in foot:
            s.blit(self.font.render(line, True, DIM), (x, y))
            y += 16

    def _trace(self, label, values, beats, x, y, w, h, color) -> int:
        pg, s = self.pg, self.screen
        s.blit(self.font.render(label, True, INK), (x, y))
        y += 18
        pg.draw.rect(s, BG, (x, y, w, h))
        self._beat_ticks(beats, x, y, w, h)
        n = len(values)
        pts = [(x + i * w / (n - 1), y + h - 2 - min(1.0, v) * (h - 4)) for i, v in enumerate(values)]
        pg.draw.lines(s, color, False, pts, 2)
        return y + h + 12

    def _beat_ticks(self, beats, x, y, w, h) -> None:
        n = len(beats)
        for i in np.flatnonzero(beats):
            bx = x + i * w / (n - 1)
            self.pg.draw.line(self.screen, (70, 66, 90), (bx, y), (bx, y + h), 1)
