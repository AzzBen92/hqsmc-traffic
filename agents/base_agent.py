"""Common interface of the tabular agents.

Every agent exposes the same methods, so the training loops do not depend on the method:
    select_action(s)                    epsilon-greedy choice
    start_episode()
    update(s, a, r, s_next, a_next)     called at every step, a_next = next (executed) action
    end_episode()                       Monte Carlo pass, if any
    decay_epsilon()                     epsilon <- max(eps_min, d * epsilon), once per episode
"""

from __future__ import annotations

import random

import numpy as np


class TabularAgent:
    name = "base"
    continuous = False      # tabular agents receive the discrete state index

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_decay: float = 0.9993, epsilon_min: float = 0.005,
                 seed: int = 42, mc_visit: str = "last"):
        self.n_states, self.n_actions = int(n_states), int(n_actions)
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon_start, epsilon_decay, epsilon_min
        self.mc_visit = mc_visit
        self.rng = random.Random(seed)   # Python generator: epsilon-greedy draws (Section 6.4)
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.buffer: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------ policy
    def values(self, s: int) -> np.ndarray:
        """Action values used for the decision."""
        return self.Q[s]

    def greedy(self, s: int) -> int:
        v = self.values(s)
        best = np.flatnonzero(v == v.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(list(best)))

    def select_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy(s)

    # ------------------------------------------------------------------ learning
    def start_episode(self) -> None:
        self.buffer = []

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        pass

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ persistence
    def tables(self) -> dict[str, np.ndarray]:
        return {"Q": self.Q}

    def state_dict(self) -> dict:
        return {"tables": {k: v.copy() for k, v in self.tables().items()},
                "epsilon": self.epsilon, "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["tables"].items():
            getattr(self, k)[...] = v
        self.epsilon = state["epsilon"]
        self.rng.setstate(state["rng"])


def monte_carlo_pass(table: np.ndarray, episode: list[tuple[int, int, float]], step: float,
                     gamma: float, visit: str = "last") -> None:
    """Backward pass  Q(s,a) <- Q(s,a) + step * (G_t - Q(s,a)).

    visit="last": the episode is traversed backwards and each (s, a) is updated only at its
    first encounter, i.e. with the return of its LAST visit (convention (ii), Section 6.3).
    visit="every": every visit is updated.
    """
    G = 0.0
    seen: set[tuple[int, int]] = set()
    for s, a, r in reversed(episode):
        G = r + gamma * G
        if visit == "last":
            if (s, a) in seen:
                continue
            seen.add((s, a))
        table[s, a] += step * (G - table[s, a])
