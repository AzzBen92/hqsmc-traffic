"""Hybrid Q-learning - SARSA - Monte Carlo methods.

HybridAgent       single Q-table (variants V2 and V3, Table 7.1)
    every step   : Q(s,a) += alpha [w_Q d_QL + w_S d_SA]      (both errors from the SAME table,
                                                               applied simultaneously)
    end of episode: Q(s,a) += alpha w_MC [G - Q(s,a)]          (backward pass, last visit)
    V3 = H-QSMC (Chapter 6): fixed weights (0.5, 0.3, 0.2), Algorithm 6
    V2 (Chapter 5)       : weights f(xi), xi = episode / N, renormalised once per episode,
                           Algorithm 5

HybridMARLAgent   three separate tables per agent (variant V4 = H-QSMC-MARL, Chapter 7)
    Q^Q  updated by Q-learning, Q^S by SARSA (every step), Q^MC by Monte Carlo (end of episode);
    decision on  Q^H = w_Q Q^Q + w_S Q^S + w_MC Q^MC  (Eq. 7.11)
"""

from __future__ import annotations

import numpy as np

from .base_agent import TabularAgent, monte_carlo_pass


# Scheduling functions of the dynamic variant (Chapters 4 and 5). The thesis specifies them only
# qualitatively: f_S decreasing (SARSA dominates early), f_Q increasing, f_MC increasing slowly
# towards the end of training. The functions below satisfy these properties.
def f_q(xi: float) -> float:
    return 0.2 + 0.6 * xi


def f_s(xi: float) -> float:
    return 0.7 - 0.6 * xi


def f_mc(xi: float) -> float:
    return 0.1 + 0.3 * xi ** 2


def dynamic_weights(xi: float) -> tuple[float, float, float]:
    xi = min(max(xi, 0.0), 1.0)
    raw = np.array([f_q(xi), f_s(xi), f_mc(xi)])
    w = raw / raw.sum()
    return float(w[0]), float(w[1]), float(w[2])


class HybridAgent(TabularAgent):
    """H-QSMC: single table, TD errors every step, Monte Carlo correction at the end of the episode."""

    name = "hybrid"

    def __init__(self, *args, weights=(0.5, 0.3, 0.2), weight_mode: str = "fixed", **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_mode = weight_mode
        self.w_q, self.w_s, self.w_mc = weights

    def set_progress(self, xi: float) -> None:
        """Dynamic variant: update the weights once per episode (Eq. 5.13)."""
        if self.weight_mode == "dynamic":
            self.w_q, self.w_s, self.w_mc = dynamic_weights(xi)

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
        self.QQ = np.zeros(shape)    # Q-learning table   (Eq. 7.7)
        self.QS = np.zeros(shape)    # SARSA table        (Eq. 7.8)
        self.QMC = np.zeros(shape)   # Monte Carlo table  (Eq. 7.9)
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
