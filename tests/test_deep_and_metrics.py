"""Gradient checks of the NumPy networks and tests of the evaluation helpers."""

import numpy as np
import pytest

from agents.deep import MLP, DQNAgent, PPOAgent, _softmax
from evaluation.metrics import convergence_episode, holm_bonferroni


def numeric_grad(f, params, eps=1e-6):
    grads = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            fp = f()
            p[i] = old - eps
            fm = f()
            p[i] = old
            g[i] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_mlp_backward_matches_finite_differences(act):
    rng = np.random.default_rng(0)
    net = MLP((3, 5, 2), rng, act)
    x = rng.normal(size=(4, 3))
    w = rng.normal(size=(4, 2))
    loss = lambda: float((net.forward(x) * w).sum())  # noqa: E731
    net.forward(x)
    analytic = net.backward(w)
    for a, n in zip(analytic, numeric_grad(loss, net.params)):
        np.testing.assert_allclose(a, n, rtol=1e-4, atol=1e-6)


def test_ppo_actor_gradient():
    """The hand-written gradient of the clipped surrogate (+ entropy) matches finite differences."""
    rng = np.random.default_rng(1)
    ag = PPOAgent(4, 2, seed=1, hidden=(6,), ent_coef=0.01)
    x = rng.normal(size=(8, 4))
    a = rng.integers(0, 2, size=8)
    logp_old = np.log(_softmax(ag.actor.forward(x))[np.arange(8), a]) + rng.normal(0, 0.1, size=8)
    adv = rng.normal(size=8)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

    def loss():
        p = _softmax(ag.actor.forward(x))
        logp = np.log(p[np.arange(8), a])
        r = np.exp(logp - logp_old)
        surr = np.minimum(r * adv_n, np.clip(r, 0.8, 1.2) * adv_n)
        H = -(p * np.log(p)).sum(axis=1)
        return float(-(surr.mean()) - 0.01 * H.mean())

    captured = {}
    ag.opt_actor.step = lambda g: captured.setdefault("g", g)
    ag.opt_critic.step = lambda g: None
    ag._minibatch(x, a, logp_old, adv, np.zeros(8))
    for g_a, g_n in zip(captured["g"], numeric_grad(loss, ag.actor.params)):
        np.testing.assert_allclose(g_a, g_n, rtol=1e-3, atol=1e-7)


def test_dqn_learns_a_two_state_problem():
    ag = DQNAgent(2, 2, gamma=0.0, epsilon_start=0.0, seed=0, hidden=(16,), lr=1e-2,
                  learning_starts=32, train_freq=1, target_update=50, batch_size=32)
    rng = np.random.default_rng(0)
    for _ in range(1500):
        s = int(rng.integers(2))
        x = np.eye(2)[s]
        a = int(rng.integers(2))
        r = 1.0 if a == s else 0.0
        ag.update(x, a, r, x, 0)
    assert ag.select_action(np.eye(2)[0]) == 0 and ag.select_action(np.eye(2)[1]) == 1


def test_holm_bonferroni():
    adj = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)      # monotone step-down


def test_convergence_episode():
    r = np.concatenate([np.linspace(-10, 5, 300), np.full(700, 5.0)])
    t = convergence_episode(r, window=10, fraction=0.9, hold=200)
    assert t is not None and 250 < t < 320
    assert convergence_episode(np.full(50, 1.0), window=5, hold=200) == 0
