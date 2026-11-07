"""Deep MARL baselines of Section 7.6.10: independent DQN and PPO agents, one per intersection.

Implemented with NumPy only (no GPU, no deep-learning framework), with the usual default
hyper-parameters (config.SCENARIO_SETTINGS["saiss"]["DQN"/"PPO"]). They receive the same
state (continuous, normalised: Table 7.2), the same binary action and the same reward
as the tabular agents, and their updates also use the EXECUTED action after the
safety constraint.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Minimal multilayer perceptron with manual back-propagation and Adam
# =============================================================================
class MLP:
    def __init__(self, sizes, rng: np.random.Generator, activation: str = "relu", out_gain: float = 1.0):
        self.activation = activation
        self.params: list[np.ndarray] = []
        for k, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if k == len(sizes) - 2 else np.sqrt(2.0)
            W = rng.normal(0.0, gain / np.sqrt(n_in), size=(n_in, n_out))
            self.params += [W, np.zeros(n_out)]
        self._cache = None

    def _act(self, z):
        return np.maximum(z, 0.0) if self.activation == "relu" else np.tanh(z)

    def _dact(self, z, h):
        return (z > 0).astype(z.dtype) if self.activation == "relu" else 1.0 - h ** 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(x)
        cache = [(None, h)]
        n_layers = len(self.params) // 2
        for k in range(n_layers):
            W, b = self.params[2 * k], self.params[2 * k + 1]
            z = h @ W + b
            h = z if k == n_layers - 1 else self._act(z)
            cache.append((z, h))
        self._cache = cache
        return h

    def backward(self, grad_out: np.ndarray) -> list[np.ndarray]:
        """Gradients of the loss w.r.t. the parameters, given dL/d(output) of the last forward()."""
        cache, grads = self._cache, [None] * len(self.params)
        g = grad_out
        n_layers = len(self.params) // 2
        for k in reversed(range(n_layers)):
            z, h = cache[k + 1]
            if k != n_layers - 1:
                g = g * self._dact(z, h)
            h_prev = cache[k][1]
            grads[2 * k] = h_prev.T @ g
            grads[2 * k + 1] = g.sum(axis=0)
            g = g @ self.params[2 * k].T
        return grads

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=None):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.max_grad_norm = max_grad_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        if self.max_grad_norm is not None:
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if norm > self.max_grad_norm:
                grads = [g * (self.max_grad_norm / (norm + 1e-12)) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# DQN (Mnih et al., 2015): replay memory + target network, epsilon-greedy
# =============================================================================
class DQNAgent:
    name = "dqn"
    continuous = True

    def __init__(self, obs_dim, n_actions, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.9993,
                 epsilon_min=0.005, seed=42, hidden=(64, 64), lr=1e-4, buffer_size=50_000,
                 batch_size=32, learning_starts=1_000, train_freq=4, target_update=1_000,
                 huber_delta=1.0, max_grad_norm=10.0, **_):
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.gamma = n_actions, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon_start, epsilon_decay, epsilon_min
        sizes = (obs_dim, *hidden, n_actions)
        self.q = MLP(sizes, self.rng, "relu")
        self.q_target = MLP(sizes, self.rng, "relu")
        self.q_target.copy_from(self.q)
        self.opt = Adam(self.q.params, lr=lr, max_grad_norm=max_grad_norm)
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.train_freq, self.target_update, self.huber_delta = train_freq, target_update, huber_delta
        self.size, self.ptr, self.cap = 0, 0, buffer_size
        self.X = np.zeros((buffer_size, obs_dim))
        self.A = np.zeros(buffer_size, dtype=np.int64)
        self.R = np.zeros(buffer_size)
        self.X2 = np.zeros((buffer_size, obs_dim))
        self.steps = 0

    def select_action(self, x):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q.forward(x)[0]
        return int(np.argmax(q))

    def start_episode(self):
        pass

    def update(self, x, a, r, x_next, a_next):
        i = self.ptr
        self.X[i], self.A[i], self.R[i], self.X2[i] = x, a, r, x_next
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.steps += 1
        if self.size >= self.learning_starts and self.steps % self.train_freq == 0:
            self._train()
        if self.steps % self.target_update == 0:
            self.q_target.copy_from(self.q)

    def _train(self):
        idx = self.rng.integers(0, self.size, size=self.batch_size)
        x, a, r, x2 = self.X[idx], self.A[idx], self.R[idx], self.X2[idx]
        target = r + self.gamma * self.q_target.forward(x2).max(axis=1)   # time-limit: bootstrap
        q = self.q.forward(x)
        err = q[np.arange(len(idx)), a] - target
        d = self.huber_delta
        g_err = np.where(np.abs(err) <= d, err, d * np.sign(err)) / len(idx)   # Huber loss
        grad_out = np.zeros_like(q)
        grad_out[np.arange(len(idx)), a] = g_err
        self.opt.step(self.q.backward(grad_out))

    def end_episode(self):
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def state_dict(self):
        return {"q": [p.copy() for p in self.q.params], "q_target": [p.copy() for p in self.q_target.params],
                "epsilon": self.epsilon, "steps": self.steps}

    def load_state_dict(self, s):
        for p, v in zip(self.q.params, s["q"]):
            p[...] = v
        for p, v in zip(self.q_target.params, s["q_target"]):
            p[...] = v
        self.epsilon, self.steps = s["epsilon"], s["steps"]


# =============================================================================
# PPO (Schulman et al., 2017): clipped objective, GAE, separate actor and critic
# =============================================================================
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class PPOAgent:
    name = "ppo"
    continuous = True

    def __init__(self, obs_dim, n_actions, gamma=0.95, seed=42, hidden=(64, 64), lr=3e-4, n_steps=2048,
                 batch_size=64, n_epochs=10, clip=0.2, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
                 max_grad_norm=0.5, **_):
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.gamma = n_actions, gamma
        self.actor = MLP((obs_dim, *hidden, n_actions), self.rng, "tanh", out_gain=0.01)
        self.critic = MLP((obs_dim, *hidden, 1), self.rng, "tanh", out_gain=1.0)
        self.opt_actor = Adam(self.actor.params, lr=lr, max_grad_norm=max_grad_norm)
        self.opt_critic = Adam(self.critic.params, lr=lr, max_grad_norm=max_grad_norm)
        self.n_steps, self.batch_size, self.n_epochs = n_steps, batch_size, n_epochs
        self.clip, self.lam, self.ent_coef, self.vf_coef = clip, gae_lambda, ent_coef, vf_coef
        self.epsilon = 0.0          # PPO explores through its stochastic policy
        self._reset_rollout()

    def _reset_rollout(self):
        self.buf = {"x": [], "a": [], "r": [], "logp": [], "v": [], "v_next": [], "end": []}

    def select_action(self, x):
        p = _softmax(self.actor.forward(x))[0]
        return int(self.rng.choice(self.n_actions, p=p))

    def start_episode(self):
        pass

    def update(self, x, a, r, x_next, a_next):
        p = _softmax(self.actor.forward(x))[0]
        b = self.buf
        b["x"].append(np.asarray(x, dtype=float))
        b["a"].append(int(a))
        b["r"].append(float(r))
        b["logp"].append(float(np.log(p[a] + 1e-12)))
        b["v"].append(float(self.critic.forward(x)[0, 0]))
        b["v_next"].append(float(self.critic.forward(x_next)[0, 0]))
        b["end"].append(False)
        if len(b["r"]) >= self.n_steps:
            self._train()

    def end_episode(self):
        if self.buf["end"]:
            self.buf["end"][-1] = True      # do not propagate the advantage across episodes

    def decay_epsilon(self):
        pass

    def _train(self):
        b = self.buf
        x = np.asarray(b["x"])
        a = np.asarray(b["a"])
        r = np.asarray(b["r"])
        logp_old = np.asarray(b["logp"])
        v = np.asarray(b["v"])
        v_next = np.asarray(b["v_next"])
        end = np.asarray(b["end"])
        n = len(r)
        adv = np.zeros(n)
        last = 0.0
        for t in reversed(range(n)):
            delta = r[t] + self.gamma * v_next[t] - v[t]
            last = delta + self.gamma * self.lam * (0.0 if end[t] else last)
            adv[t] = last
        returns = adv + v
        for _ in range(self.n_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                mb = order[start:start + self.batch_size]
                self._minibatch(x[mb], a[mb], logp_old[mb], adv[mb], returns[mb])
        self._reset_rollout()

    def _minibatch(self, x, a, logp_old, adv, ret):
        m = len(a)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if m > 1 else adv
        # ---- actor: clipped surrogate objective L^CLIP (Chapter 5) + entropy bonus
        logits = self.actor.forward(x)
        p = _softmax(logits)
        logp = np.log(p[np.arange(m), a] + 1e-12)
        ratio = np.exp(logp - logp_old)
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
        use_unclipped = unclipped <= clipped                  # min(...) picks the unclipped term
        dlogp = np.where(use_unclipped, -adv * ratio, 0.0) / m  # d(-L_clip)/d log pi(a)
        onehot = np.zeros_like(p)
        onehot[np.arange(m), a] = 1.0
        g_logits = dlogp[:, None] * (onehot - p)
        if self.ent_coef > 0:
            logp_all = np.log(p + 1e-12)
            H = -(p * logp_all).sum(axis=1, keepdims=True)
            dH = -p * (logp_all + H)                          # dH/dlogits
            g_logits -= self.ent_coef * dH / m
        self.opt_actor.step(self.actor.backward(g_logits))
        # ---- critic: value regression
        values = self.critic.forward(x)[:, 0]
        g_v = (self.vf_coef * 2.0 * (values - ret) / m)[:, None]
        self.opt_critic.step(self.critic.backward(g_v))

    def state_dict(self):
        return {"actor": [p.copy() for p in self.actor.params],
                "critic": [p.copy() for p in self.critic.params]}

    def load_state_dict(self, s):
        for p, v in zip(self.actor.params, s["actor"]):
            p[...] = v
        for p, v in zip(self.critic.params, s["critic"]):
            p[...] = v
