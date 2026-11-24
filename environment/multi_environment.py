"""Multi-intersection environment: the Saiss network, eight agents (Chapter 7).

Local state of agent i (Table 7.2 and Eq. 7.5):
    s_t^i = (q_t^i, w_t^i, v_t^i, p_t^i, d_t^i, phi_i(t))
    q, w  : min-max scaling on [0, QUEUE_MAX] and [0, WAIT_MAX]
    v     : divided by the free-flow speed
    p     : current phase, binary (0 = horizontal / East-West, 1 = vertical / North-South)
    d     : time since the last phase change, divided by g_min
    phi_i : mean (q, w) over the direct neighbours N_i (Eq. 7.4), same normalisation
Actions: A^i = {0, 1} = request of green for the horizontal / vertical movement.
Safety constraint (Eq. 7.15): the request is executed only if d >= g_min, otherwise the
current phase p is kept; the EXECUTED action is the one used in the updates.
"""

from __future__ import annotations

import numpy as np

from .intersection import (IntersectionMonitor, SignalController, bin_index, build_green_states,
                           multi_objective_reward, nearest_tls)
from .sumo_utils import SumoRunner


AXIS_OF_ACTION = {0: "EW", 1: "NS"}     # 0 = horizontal, 1 = vertical


def axis_bit(c: SignalController) -> int:
    """Binary phase p of Table 7.2: 0 if the active (or target) green serves East-West, 1 North-South."""
    return 1 if c.axis == "NS" else 0


class MultiIntersectionEnv:
    n_actions = 2

    def __init__(self, cfg, label: str = "multi"):
        self.cfg = cfg
        self.runner = SumoRunner(cfg, label=label)
        self.agent_ids = sorted(cfg.TLS_COORDS) if cfg.TLS_IDS is None else sorted(cfg.TLS_IDS)
        self.n_agents = len(self.agent_ids)
        self.neighbors = {i: list(cfg.NEIGHBORS.get(i, [])) for i in self.agent_ids}
        self.use_neighbors = bool(cfg.USE_NEIGHBORS)
        self.control = cfg.ALGORITHM != "fixed"
        nb = cfg.NEIGHBOR_BINS if self.use_neighbors else 1
        self.state_dims = (cfg.QUEUE_BINS, cfg.WAIT_BINS, cfg.SPEED_BINS, 2, cfg.DURATION_BINS, nb, nb)
        self.n_states = int(np.prod(self.state_dims))
        self.obs_dim = 7 if self.use_neighbors else 5     # continuous state for DQN / PPO
        self.traci = None
        self.tls: dict[int, str] = {}
        self.controllers: dict[int, SignalController] = {}
        self.monitors: dict[int, IntersectionMonitor] = {}
        self.t = 0.0
        self.arrived = 0
        self.n_steps = 0
        self.horizon = int(round(cfg.EPISODE_LENGTH / cfg.STEP_LENGTH))   # T decision steps

    # ------------------------------------------------------------------ episode
    def reset(self, episode: int = 0):
        cfg = self.cfg
        seed = cfg.SEED + episode if cfg.SUMO_SEED_PER_EPISODE else cfg.SEED
        self.traci = self.runner.start(seed)
        self.tls = self._map_agents_to_tls()
        self.controllers, self.monitors = {}, {}
        for i, tls_id in self.tls.items():
            if self.control:
                greens, axes = build_green_states(self.traci, tls_id)
                self.controllers[i] = SignalController(self.traci, tls_id, greens, axes, cfg.MIN_GREEN,
                                                       cfg.MAX_GREEN, cfg.YELLOW, cfg.ALL_RED,
                                                       cfg.STEP_LENGTH)
            self.monitors[i] = IntersectionMonitor(self.traci, tls_id, cfg.STEP_LENGTH)
        self.t, self.arrived = 0.0, 0
        self._advance()
        self.n_steps = 0
        return self.observe()

    def _map_agents_to_tls(self) -> dict[int, str]:
        cfg = self.cfg
        if cfg.TLS_IDS is not None:
            return dict(cfg.TLS_IDS)
        mapping, used = {}, set()
        for i in self.agent_ids:
            tls_id = nearest_tls(self.traci, cfg.TLS_COORDS[i], exclude=used)
            mapping[i] = tls_id
            used.add(tls_id)
        return mapping

    def _advance(self) -> dict[int, dict]:
        self.traci.simulationStep()
        for c in self.controllers.values():
            c.tick()
        self.t += self.cfg.STEP_LENGTH
        self.arrived += self.traci.simulation.getArrivedNumber()
        self.measures = {i: m.measure() for i, m in self.monitors.items()}
        return self.measures

    # ------------------------------------------------------------------ actions
    def current_phase(self, i: int) -> int:
        c = self.controllers.get(i)
        return axis_bit(c) if c else 0

    def project(self, i: int, requested: int) -> int:
        """Safety projection C(a, s) of Eq. 7.15."""
        c = self.controllers.get(i)
        if c is None:
            return requested
        return int(requested) if c.can_switch else axis_bit(c)

    def step(self, executed: dict[int, int]):
        """Apply the executed joint action; returns (obs, rewards, done, info)."""
        for i, a in executed.items():
            c = self.controllers.get(i)
            if c is not None and a != axis_bit(c):
                c.request_axis(AXIS_OF_ACTION[a])
        self._advance()
        self.n_steps += 1
        rewards = {i: multi_objective_reward(self.measures[i], self.cfg.REWARD_COEFS)
                   for i in self.agent_ids}
        done = self.n_steps >= self.horizon or self.traci.simulation.getMinExpectedNumber() == 0
        return self.observe(), rewards, done, self.measures

    def close(self):
        self.runner.close()

    # ------------------------------------------------------------------ observation
    def features(self, i: int) -> np.ndarray:
        """Normalised continuous state (also the input of DQN / PPO)."""
        cfg = self.cfg
        m = self.measures[i]
        c = self.controllers.get(i)
        q = min(m["queue"] / cfg.QUEUE_MAX, 1.0)
        w = min(m["wait"] / cfg.WAIT_MAX, 1.0)
        v = min(m["speed"] / max(self.monitors[i].free_speed, 0.1), 1.0)
        p = float(axis_bit(c)) if c else 0.0
        d = min((c.elapsed if c and not c.in_transition else 0.0) / cfg.MIN_GREEN, 1.0) if c else 1.0
        x = [q, w, v, p, d]
        if self.use_neighbors:
            nb = self.neighbors[i]
            if nb:
                x.append(float(np.mean([min(self.measures[j]["queue"] / cfg.QUEUE_MAX, 1.0) for j in nb])))
                x.append(float(np.mean([min(self.measures[j]["wait"] / cfg.WAIT_MAX, 1.0) for j in nb])))
            else:
                x += [0.0, 0.0]
        return np.asarray(x, dtype=np.float64)

    def encode(self, x: np.ndarray) -> int:
        cfg = self.cfg
        digits = [
            bin_index(x[0], cfg.QUEUE_BINS, 1.0),
            bin_index(x[1], cfg.WAIT_BINS, 1.0),
            bin_index(x[2], cfg.SPEED_BINS, 1.0),
            int(x[3]),
            bin_index(x[4], cfg.DURATION_BINS, 1.0) if cfg.DURATION_BINS > 2 else int(x[4] >= 1.0),
        ]
        if self.use_neighbors:
            digits += [bin_index(x[5], cfg.NEIGHBOR_BINS, 1.0), bin_index(x[6], cfg.NEIGHBOR_BINS, 1.0)]
        else:
            digits += [0, 0]
        idx = 0
        for dgt, base in zip(digits, self.state_dims):
            idx = idx * base + dgt
        return idx

    def observe(self) -> dict[int, dict]:
        obs = {}
        for i in self.agent_ids:
            x = self.features(i)
            obs[i] = {"x": x, "s": self.encode(x)}
        return obs

    # ------------------------------------------------------------------ metrics
    def episode_summary(self) -> dict:
        """Network-level indicators: mean over the eight intersections (Section 7.5.4)."""
        per = {i: m.episode_summary() for i, m in self.monitors.items()}
        keys = ("queue", "wait", "speed", "co2", "fuel")
        out = {k: float(np.mean([p[k] for p in per.values()])) for k in keys}
        out["throughput"] = self.arrived * 3600.0 / max(self.n_steps * self.cfg.STEP_LENGTH, 1.0)
        out["per_agent"] = per
        return out
