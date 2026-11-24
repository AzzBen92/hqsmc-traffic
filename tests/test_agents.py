"""Unit tests of the update rules (no SUMO needed):  pytest -q"""

import numpy as np
import pytest

from agents import HybridAgent, HybridMARLAgent, MonteCarloAgent, QLearningAgent, SarsaAgent
from agents.base_agent import monte_carlo_pass
from agents.hybrid import dynamic_weights

KW = dict(alpha=0.1, gamma=0.9, epsilon_start=0.0, seed=0)


def test_qlearning_update():
    ag = QLearningAgent(3, 2, **KW)
    ag.Q[1] = [2.0, 5.0]
    ag.update(0, 1, 1.0, 1, 0)
    assert ag.Q[0, 1] == pytest.approx(0.1 * (1.0 + 0.9 * 5.0))


def test_sarsa_uses_next_action():
    ag = SarsaAgent(3, 2, **KW)
    ag.Q[1] = [2.0, 5.0]
    ag.update(0, 1, 1.0, 1, 0)
    assert ag.Q[0, 1] == pytest.approx(0.1 * (1.0 + 0.9 * 2.0))


def test_monte_carlo_last_visit():
    Q = np.zeros((2, 1))
    # episode: (s0,a0,r=1), (s0,a0,r=2) -> backward: G=2 for the last visit (updated), G=1+0.9*2 skipped
    monte_carlo_pass(Q, [(0, 0, 1.0), (0, 0, 2.0)], step=1.0, gamma=0.9, visit="last")
    assert Q[0, 0] == pytest.approx(2.0)
    Q[:] = 0
    monte_carlo_pass(Q, [(0, 0, 1.0), (0, 0, 2.0)], step=1.0, gamma=0.9, visit="every")
    assert Q[0, 0] == pytest.approx(1.0 + 0.9 * 2.0)


def test_monte_carlo_agent_updates_only_at_episode_end():
    ag = MonteCarloAgent(2, 2, **KW)
    ag.start_episode()
    ag.update(0, 1, 1.0, 1, 0)
    assert ag.Q.sum() == 0
    ag.end_episode()
    assert ag.Q[0, 1] == pytest.approx(0.1)


def test_hybrid_simultaneous_td_errors():
    ag = HybridAgent(3, 2, weights=(0.5, 0.3, 0.2), **KW)
    ag.Q[0, 1] = 1.0
    ag.Q[1] = [2.0, 5.0]
    ag.start_episode()
    ag.update(0, 1, 1.0, 1, 0)
    d_ql = 1.0 + 0.9 * 5.0 - 1.0
    d_sa = 1.0 + 0.9 * 2.0 - 1.0
    assert ag.Q[0, 1] == pytest.approx(1.0 + 0.1 * (0.5 * d_ql + 0.3 * d_sa))
    before = ag.Q[0, 1]
    ag.end_episode()      # Monte Carlo correction with weight w_MC: G = 1
    assert ag.Q[0, 1] == pytest.approx(before + 0.1 * 0.2 * (1.0 - before))


def test_hybrid_degenerate_weights_equal_single_methods():
    rng = np.random.default_rng(0)
    transitions = [(int(rng.integers(5)), int(rng.integers(2)), float(rng.normal()), int(rng.integers(5)),
                    int(rng.integers(2))) for _ in range(200)]
    for w, cls in (((1, 0, 0), QLearningAgent), ((0, 1, 0), SarsaAgent), ((0, 0, 1), MonteCarloAgent)):
        h, ref = HybridAgent(5, 2, weights=w, **KW), cls(5, 2, **KW)
        for ag in (h, ref):
            ag.start_episode()
            for t in transitions:
                ag.update(*t)
            ag.end_episode()
        np.testing.assert_allclose(h.Q, ref.Q)


def test_marl_three_tables_and_aggregation():
    ag = HybridMARLAgent(4, 2, weights=(0.5, 0.3, 0.2), **KW)
    ag.start_episode()
    ag.update(0, 1, 2.0, 1, 0)
    ag.end_episode()
    assert ag.QQ[0, 1] == pytest.approx(0.2) and ag.QS[0, 1] == pytest.approx(0.2)
    assert ag.QMC[0, 1] == pytest.approx(0.2)
    assert ag.values(0)[1] == pytest.approx(0.5 * 0.2 + 0.3 * 0.2 + 0.2 * 0.2)


def test_dynamic_weights_properties():
    w0, w_mid, w1 = dynamic_weights(0.0), dynamic_weights(0.5), dynamic_weights(1.0)
    for w in (w0, w_mid, w1):
        assert sum(w) == pytest.approx(1.0) and min(w) >= 0
    assert w0[1] == max(w0)              # SARSA dominates at the beginning
    assert w1[0] > w0[0]                 # Q-learning weight increases
    assert w1[2] > w_mid[2] > w0[2]      # Monte Carlo weight increases towards the end


def test_epsilon_decay_and_floor():
    ag = QLearningAgent(2, 2, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_decay=0.5, epsilon_min=0.2)
    for _ in range(5):
        ag.decay_epsilon()
    assert ag.epsilon == pytest.approx(0.2)


def test_state_dict_roundtrip():
    ag = HybridMARLAgent(4, 2, **KW)
    ag.QQ[1, 1] = 3.0
    s = ag.state_dict()
    ag2 = HybridMARLAgent(4, 2, **KW)
    ag2.load_state_dict(s)
    assert ag2.QQ[1, 1] == 3.0
