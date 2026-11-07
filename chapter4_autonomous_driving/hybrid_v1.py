"""First formulation of the hybrid method (variant V1, Chapter 4, Eq. 4.1).

Three estimates of Q(s, a) are maintained, each with its own rule: Q_Q (Q-learning), Q_S (SARSA)
and Q_MC (Monte Carlo, end of episode). The decision table is updated as their weighted mix

    Q_new(s,a) = (1 - alpha) Q(s,a) + alpha [w_Q Q_Q(s,a) + w_S Q_S(s,a) + w_MC Q_MC(s,a)],

with dynamic weights w = f(xi), xi = episode / N (SARSA dominant early, then Q-learning, Monte Carlo
growing at the end), renormalised so that w_Q + w_S + w_MC = 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base_agent import TabularAgent, monte_carlo_pass  # noqa: E402
from agents.hybrid import dynamic_weights  # noqa: E402


class HybridV1Agent(TabularAgent):
    name = "hybrid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.n_states, self.n_actions)
        self.QQ, self.QS, self.QMC = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.w_q, self.w_s, self.w_mc = dynamic_weights(0.0)

    def set_progress(self, xi: float):
        self.w_q, self.w_s, self.w_mc = dynamic_weights(xi)

    def _blend(self, s, a):
        mix = self.w_q * self.QQ[s, a] + self.w_s * self.QS[s, a] + self.w_mc * self.QMC[s, a]
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * mix

    def update(self, s, a, r, s_next, a_next):
        self.QQ[s, a] += self.alpha * (r + self.gamma * self.QQ[s_next].max() - self.QQ[s, a])
        self.QS[s, a] += self.alpha * (r + self.gamma * self.QS[s_next, a_next] - self.QS[s, a])
        self.buffer.append((s, a, r))
        self._blend(s, a)

    def end_episode(self):
        monte_carlo_pass(self.QMC, self.buffer, self.alpha, self.gamma, self.mc_visit)
        for s, a in {(s, a) for s, a, _ in self.buffer}:
            self._blend(s, a)
        self.buffer = []

    def tables(self):
        return {"Q": self.Q, "QQ": self.QQ, "QS": self.QS, "QMC": self.QMC}
