"""Hybrid Q-learning - SARSA - Monte Carlo methods.

HybridAgent       H-QSMC, single Q-table, fixed weights (w_Q, w_S, w_MC)
    every step   : Q(s,a) += alpha [w_Q d_QL + w_S d_SA]      (both errors from the SAME table,
                                                               applied simultaneously)
    end of episode: Q(s,a) += alpha w_MC [G - Q(s,a)]          (backward pass, last visit)

HybridMARLAgent   H-QSMC-MARL, three separate tables per agent
    Q^Q  updated by Q-learning, Q^S by SARSA (every step), Q^MC by Monte Carlo (end of episode);
    decision on  Q^H = w_Q Q^Q + w_S Q^S + w_MC Q^MC
"""

from __future__ import annotations

import numpy as np

from .base_agent import TabularAgent, monte_carlo_pass


class HybridAgent(TabularAgent):
    """H-QSMC: single table, TD errors every step, Monte Carlo correction at the end of the episode."""

    name = "hybrid"

    def __init__(self, *args, weights=(0.5, 0.3, 0.2), **kwargs):
        super().__init__(*args, **kwargs)
        self.w_q, self.w_s, self.w_mc = weights

    def update(self, s, a, r, s_next, a_next):
        q_sa = self.Q[s, a]                                    # value at the start of the step
        d_ql = r + self.gamma * self.Q[s_next].max() - q_sa
        d_sa = r + self.gamma * self.Q[s_next, a_next] - q_sa
        self.Q[s, a] = q_sa + self.alpha * (self.w_q * d_ql + self.w_s * d_sa)
        self.buffer.append((s, a, r))

    def end_episode(self):
        if self.w_mc > 0:
            monte_carlo_pass(self.Q, self.buffer, self.alpha * self.w_mc, self.gamma, self.mc_visit)
        self.buffer = []

    @property
    def weights(self):
        return self.w_q, self.w_s, self.w_mc


class HybridMARLAgent(TabularAgent):
    """H-QSMC-MARL agent: three independent value tables combined at decision time."""

    name = "hybrid"

    def __init__(self, *args, weights=(0.5, 0.3, 0.2), **kwargs):
        super().__init__(*args, **kwargs)
        self.w_q, self.w_s, self.w_mc = weights
        shape = (self.n_states, self.n_actions)
        self.QQ = np.zeros(shape)    # Q-learning table
        self.QS = np.zeros(shape)    # SARSA table
        self.QMC = np.zeros(shape)   # Monte Carlo table
        del self.Q

    def values(self, s):
        return self.w_q * self.QQ[s] + self.w_s * self.QS[s] + self.w_mc * self.QMC[s]

    def hybrid_table(self) -> np.ndarray:
        return self.w_q * self.QQ + self.w_s * self.QS + self.w_mc * self.QMC

    def update(self, s, a, r, s_next, a_next):
        a_ = self.alpha
        self.QQ[s, a] = (1 - a_) * self.QQ[s, a] + a_ * (r + self.gamma * self.QQ[s_next].max())
        self.QS[s, a] = (1 - a_) * self.QS[s, a] + a_ * (r + self.gamma * self.QS[s_next, a_next])
        self.buffer.append((s, a, r))

    def end_episode(self):
        monte_carlo_pass(self.QMC, self.buffer, self.alpha, self.gamma, self.mc_visit)
        self.buffer = []

    def tables(self):
        return {"QQ": self.QQ, "QS": self.QS, "QMC": self.QMC}
