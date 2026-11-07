"""Tabular Q-learning (off-policy):  Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]."""

from .base_agent import TabularAgent


class QLearningAgent(TabularAgent):
    name = "qlearning"

    def update(self, s, a, r, s_next, a_next):
        td = r + self.gamma * self.Q[s_next].max() - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
