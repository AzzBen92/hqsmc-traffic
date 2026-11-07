"""Tabular Monte Carlo control with a constant step size.

The table is updated only at the end of the episode, from the complete returns
G_t = sum_k gamma^k r_{t+k+1}:  Q(s,a) <- Q(s,a) + alpha [G_t - Q(s,a)]
(constant alpha rather than 1/n, Section 6.3).
"""

from .base_agent import TabularAgent, monte_carlo_pass


class MonteCarloAgent(TabularAgent):
    name = "montecarlo"

    def update(self, s, a, r, s_next, a_next):
        self.buffer.append((s, a, r))

    def end_episode(self):
        monte_carlo_pass(self.Q, self.buffer, self.alpha, self.gamma, self.mc_visit)
        self.buffer = []
