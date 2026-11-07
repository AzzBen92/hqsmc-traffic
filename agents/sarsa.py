"""Tabular SARSA (on-policy):  Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]."""

from .base_agent import TabularAgent


class SarsaAgent(TabularAgent):
    name = "sarsa"

    def update(self, s, a, r, s_next, a_next):
        td = r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
