"""Car-on-a-track environment of Chapter 4 (Section 4.5.1), with the Gym interface.

* state   : position (x, y) of the car, discretised on a grid (the heading and the speed are
            NOT part of the state, as in the thesis - see the limits discussed in Section 4.5.4)
* actions : 0 accelerate, 1 brake, 2 turn left, 3 turn right
* reward  : -1 per step, -10 when the car hits a wall / leaves the track (it is then put back
            at the start line); 100 steps per episode
Rendering with Pygame is optional (render_mode="human"); training does not need it.
"""

from __future__ import annotations

import math
import random


class TrackEnv:
    metadata = {"render_modes": ["human"], "render_fps": 30}
    n_actions = 4
    action_names = ("accelerate", "brake", "left", "right")

    def __init__(self, width=600, height=400, margin=40, track_width=110, cell=20, max_steps=100,
                 v_max=12.0, accel=2.0, turn_deg=15.0, step_penalty=-1.0, crash_penalty=-10.0,
                 seed=None, render_mode=None):
        self.W, self.H, self.margin, self.tw = width, height, margin, track_width
        self.cell, self.max_steps = cell, max_steps
        self.v_max, self.accel, self.turn = v_max, accel, math.radians(turn_deg)
        self.step_penalty, self.crash_penalty = step_penalty, crash_penalty
        self.nx, self.ny = width // cell, height // cell
        self.n_states = self.nx * self.ny
        self.rng = random.Random(seed)
        self.render_mode = render_mode
        self._screen = None
        # outer wall rectangle and inner island: the drivable track is the ring between them
        self.outer = (margin, margin, width - margin, height - margin)
        self.inner = (margin + track_width, margin + track_width,
                      width - margin - track_width, height - margin - track_width)
        self.start = (width / 2, margin + track_width / 2, 0.0)   # bottom straight, heading east

    # ------------------------------------------------------------------ Gym API
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.x, self.y, self.theta = self.start
        self.v = 0.0
        self.t = 0
        self.crashes = 0
        return self._obs(), {}

    def step(self, action: int):
        if action == 0:
            self.v = min(self.v + self.accel, self.v_max)
        elif action == 1:
            self.v = max(self.v - self.accel, 0.0)
        elif action == 2:
            self.theta += self.turn
        elif action == 3:
            self.theta -= self.turn
        self.x += self.v * math.cos(self.theta)
        self.y += self.v * math.sin(self.theta)
        reward = self.step_penalty
        crashed = not self.on_track(self.x, self.y)
        if crashed:
            reward = self.crash_penalty
            self.crashes += 1
            self.x, self.y, self.theta = self.start
            self.v = 0.0
        self.t += 1
        truncated = self.t >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, False, truncated, {"crashed": crashed}

    def on_track(self, x, y) -> bool:
        x0, y0, x1, y1 = self.outer
        i0, j0, i1, j1 = self.inner
        inside_outer = x0 <= x <= x1 and y0 <= y <= y1
        inside_inner = i0 < x < i1 and j0 < y < j1
        return inside_outer and not inside_inner

    def _obs(self) -> int:
        i = min(max(int(self.x // self.cell), 0), self.nx - 1)
        j = min(max(int(self.y // self.cell), 0), self.ny - 1)
        return j * self.nx + i

    # ------------------------------------------------------------------ rendering
    def render(self):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install pygame to render the track") from exc
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.W, self.H))
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        s = self._screen
        s.fill((60, 140, 60))
        flip = lambda r: pygame.Rect(r[0], self.H - r[3], r[2] - r[0], r[3] - r[1])  # noqa: E731
        pygame.draw.rect(s, (90, 90, 90), flip(self.outer))
        pygame.draw.rect(s, (60, 140, 60), flip(self.inner))
        cx, cy = int(self.x), int(self.H - self.y)
        pygame.draw.circle(s, (220, 40, 40), (cx, cy), 8)
        hx, hy = cx + 14 * math.cos(self.theta), cy - 14 * math.sin(self.theta)
        pygame.draw.line(s, (255, 255, 255), (cx, cy), (hx, hy), 2)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
