"""pygame window: the dancing fly on the left, the evidence on the right.

The right panel shows the chain end to end: sound onsets -> voltage into the antennae -> a
raster of the descending neurons that listen -> the channel levels that move the body.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .brain import CHANNELS
from .engine import Pipeline
from .fly import Dancer, skeleton

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
    def __init__(self, pipeline: Pipeline, source_name: str, headless: bool = False):
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
        self.clock = pygame.time.Clock()
        cal = pipeline.brain.cal
        order = np.lexsort((cal.latency, cal.tier))
        self.row_order = order
        self.row_colors = np.array([TIER_COLORS[t] for t in cal.tier[order]], np.uint8)

    def frame(self) -> bool:
        """Draw one frame. Returns False when the user closes the window."""
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_q)):
                return False
        dt = self.clock.tick(60) / 1000
        snap = self.pipeline.timeline.snapshot()
        latest = snap["latest"]
        levels = dict(zip(CHANNELS, latest.levels)) if latest else dict.fromkeys(CHANNELS, 0.0)
        pose = self.dancer.update(levels, latest.drive if latest else 0.0, dt)

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
        for leg in sk.legs:
            pg.draw.lines(s, BODY_DARK, False, leg, 7)
            pg.draw.lines(s, BODY, False, leg, 4)
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

        s.blit(self.big.render("kpop-fly", True, INK), (24, 18))
        s.blit(self.font.render("phase 1: brain -> beat", True, DIM), (26, 46))
        bpm = self.pipeline.detector.bpm()
        s.blit(self.font.render(f"{self.source_name}   {f'heard {bpm:.0f} BPM' if bpm else 'listening...'}", True, INK),
               (26, H - 40))

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

        labels = {"all": "all", "front": "fast -> arms", "mid": "mid -> mid legs", "hind": "slow -> knees",
                  "left": "left brain", "right": "right brain"}
        colors = {"all": INK, "front": TIER_COLORS[0], "mid": TIER_COLORS[1], "hind": TIER_COLORS[2],
                  "left": GOLD, "right": GOLD}
        for i, ch in enumerate(CHANNELS):
            cx, cy = x + (i % 2) * (w // 2), y + (i // 2) * 24
            s.blit(self.font.render(labels[ch], True, DIM), (cx, cy))
            bw = w // 2 - 150
            pg.draw.rect(s, BG, (cx + 130, cy + 2, bw, 12))
            pg.draw.rect(s, colors[ch], (cx + 130, cy + 2, int(bw * levels[ch]), 12))
        y += 3 * 24 + 8
        foot = ("when each part moves: the brain's descending neurons.",
                "how far it moves: an artistic mapping.",
                "wiring: MaleCNS v1.0 (Berg et al. 2026, CC BY 4.0) via flybrain")
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
