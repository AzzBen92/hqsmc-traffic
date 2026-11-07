"""Learning agents: tabular Q-learning, SARSA, Monte Carlo, H-QSMC (single table),
H-QSMC-MARL (three tables per agent) and NumPy DQN / PPO baselines."""

from .base_agent import TabularAgent
from .hybrid import HybridAgent, HybridMARLAgent
from .montecarlo import MonteCarloAgent
from .qlearning import QLearningAgent
from .sarsa import SarsaAgent

TABULAR = {
    "qlearning": QLearningAgent,
    "sarsa": SarsaAgent,
    "montecarlo": MonteCarloAgent,
}

__all__ = ["TabularAgent", "QLearningAgent", "SarsaAgent", "MonteCarloAgent",
           "HybridAgent", "HybridMARLAgent", "TABULAR"]
