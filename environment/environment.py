"""Single-intersection environment (Chapters 5 and 6).

MDP of Section 6.2:
    state   s_t = (q^(1..4), w, v, phi)  -> 5^4 x 5 x 5 x 4 = 62 500 discrete states
    actions 0: keep the current phase, 1: green North-South, 2: green East-West,
            3: next phase of the plan   (min green 10 s, max green 60 s)
    reward  multi-objective (Eq. 6.2), or minus the cumulative waiting time (Chapter 5)
"""

from __future__ import annotations

from .intersection import (IntersectionMonitor, SignalController, bin_index, build_green_states,
                           main_tls, multi_objective_reward, nearest_tls)
from .sumo_utils import SumoRunner

KEEP, GREEN_NS, GREEN_EW, NEXT = 0, 1, 2, 3


class TrafficSignalEnv:
    n_actions = 4
    action_names = ("keep", "green_NS", "green_EW", "next_phase")

    def __init__(self, cfg, label: str = "single"):
        self.cfg = cfg
        self.runner = SumoRunner(cfg, label=label)
        self.traci = None
        self.tls_id = None
        self.controller: SignalController | None = None
        self.monitor: IntersectionMonitor | None = None
        self.control = cfg.ALGORITHM != "fixed"   # "fixed": native SUMO program, no agent
        self.n_q, self.q_max = cfg.QUEUE_BINS, cfg.QUEUE_MAX
        self.n_w, self.w_max = cfg.WAIT_BINS, cfg.WAIT_MAX
        self.n_v, self.v_max = cfg.SPEED_BINS, cfg.SPEED_MAX
        self.n_phi = 4
        self.n_states = self.n_q ** 4 * self.n_w * self.n_v * self.n_phi
        self.t = 0.0
        self.arrived = 0
        self.n_steps = 0
        self.horizon = int(round(cfg.EPISODE_LENGTH / cfg.STEP_LENGTH))   # T decision steps

    # ------------------------------------------------------------------ episode
    def reset(self, episode: int = 0) -> int:
        cfg = self.cfg
        seed = cfg.SEED + episode if cfg.SUMO_SEED_PER_EPISODE else cfg.SEED
        self.traci = self.runner.start(seed)
        self.tls_id = self._select_tls()
        if self.control:
            greens, axes = build_green_states(self.traci, self.tls_id)
            self.controller = SignalController(self.traci, self.tls_id, greens, axes, cfg.MIN_GREEN,
                                               cfg.MAX_GREEN, cfg.YELLOW, cfg.ALL_RED, cfg.STEP_LENGTH)
        self.monitor = IntersectionMonitor(self.traci, self.tls_id, cfg.STEP_LENGTH)
        self.t, self.arrived, self.n_steps = 0.0, 0, 0
        self.traci.simulationStep()
        if self.controller:
            self.controller.tick()
        self.t += cfg.STEP_LENGTH
        return self.encode(self.monitor.measure())

    def _select_tls(self) -> str:
        cfg = self.cfg
        if cfg.TLS_ID:
            return cfg.TLS_ID
        if cfg.TLS_COORD:
            return nearest_tls(self.traci, cfg.TLS_COORD)
        return main_tls(self.traci)

    def step(self, action: int):
        """Apply `action` for one decision step; returns (state, reward, done, measures)."""
        if self.control:
            c = self.controller
            if action == GREEN_NS:
                c.request_axis("NS")
            elif action == GREEN_EW:
                c.request_axis("EW")
            elif action == NEXT:
                c.request_next()
            # KEEP: nothing to do (the max-green rule is enforced in tick())
        self.traci.simulationStep()
        if self.controller:
            self.controller.tick()
        self.t += self.cfg.STEP_LENGTH
        self.n_steps += 1
        self.arrived += self.traci.simulation.getArrivedNumber()
        m = self.monitor.measure()
        reward = self.reward(m)
        done = self.n_steps >= self.horizon or self.traci.simulation.getMinExpectedNumber() == 0
        return self.encode(m), reward, done, m

    def reward(self, m: dict) -> float:
        if self.cfg.REWARD_MODE == "waiting":          # Chapter 5
            return -m["wait_total"]
        return multi_objective_reward(m, self.cfg.REWARD_COEFS)   # Chapter 6

    def close(self):
        self.runner.close()

    # ------------------------------------------------------------------ state
    def phase_index(self) -> int:
        if self.controller:
            return self.controller.phase_index
        # native program: map its phase index on {0..3}
        return min(self.traci.trafficlight.getPhase(self.tls_id), 3)

    def encode(self, m: dict) -> int:
        """Mixed-radix index of the discretised state (Section 6.2.2)."""
        idx = 0
        for q in m["queues"]:
            idx = idx * self.n_q + bin_index(q, self.n_q, self.q_max)
        idx = idx * self.n_w + bin_index(m["wait"], self.n_w, self.w_max)
        idx = idx * self.n_v + bin_index(m["speed"], self.n_v, self.v_max)
        idx = idx * self.n_phi + self.phase_index()
        return idx

    def episode_summary(self) -> dict:
        s = self.monitor.episode_summary()
        s["throughput"] = self.arrived * 3600.0 / max(self.n_steps * self.cfg.STEP_LENGTH, 1.0)  # veh/h
        s["switches"] = self.controller.switches if self.controller else 0
        return s
