"""Multi-agent training loop for the Saiss network (Chapter 7, Algorithm 7).

At every decision step, for every agent i:
    requested action  a_t^i  = epsilon-greedy on its values (Q^H for H-QSMC-MARL)
    executed action   ~a_t^i = C(a_t^i, s_t^i)  (safety constraint, Eq. 7.15)
The effective joint action is applied in SUMO; Q-learning and SARSA tables are updated every
step with the EXECUTED actions (SARSA uses ~a_{t+1}^i), the Monte Carlo table at the end of the
episode. Checkpoints (Q-tables, epsilon, RNG states) are written every CHECKPOINT_EVERY episodes.
"""

from __future__ import annotations

import time

import numpy as np

from agents import TABULAR, HybridMARLAgent
from agents.deep import DQNAgent, PPOAgent
from config import run_dir
from environment.multi_environment import MultiIntersectionEnv

from .common import (EpisodeLogger, load_checkpoint, progress, save_checkpoint, save_tables,
                     write_metadata)

BASE_FIELDS = ["episode", "reward", "reward_sum", "queue", "wait", "speed", "co2", "fuel",
               "throughput", "epsilon", "steps", "duration_s"]


def make_agents(cfg, env: MultiIntersectionEnv) -> dict:
    agents = {}
    for i in env.agent_ids:
        seed = cfg.SEED * 100 + i          # one reproducible generator per agent
        common = dict(gamma=cfg.GAMMA, epsilon_start=cfg.EPSILON_START, epsilon_decay=cfg.EPSILON_DECAY,
                      epsilon_min=cfg.EPSILON_MIN, seed=seed)
        alg = cfg.ALGORITHM
        if alg == "hybrid":
            agents[i] = HybridMARLAgent(env.n_states, env.n_actions, alpha=cfg.ALPHA,
                                        weights=cfg.HYBRID_WEIGHTS, mc_visit=cfg.MC_VISIT, **common)
        elif alg in TABULAR:
            agents[i] = TABULAR[alg](env.n_states, env.n_actions, alpha=cfg.ALPHA,
                                     mc_visit=cfg.MC_VISIT, **common)
        elif alg == "dqn":
            agents[i] = DQNAgent(env.obs_dim, env.n_actions, **common, **cfg.DQN)
        elif alg == "ppo":
            agents[i] = PPOAgent(env.obs_dim, env.n_actions, gamma=cfg.GAMMA, seed=seed, **cfg.PPO)
        else:  # "fixed"
            return {}
    return agents


def _state(agent, o):
    return o["x"] if agent.continuous else o["s"]


def run_episode(env: MultiIntersectionEnv, agents: dict, episode: int) -> dict:
    t0 = time.time()
    obs = env.reset(episode)
    ids = env.agent_ids
    total = {i: 0.0 for i in ids}
    steps = 0
    if not agents:  # fixed-time / native program reference
        done = False
        while not done:
            obs, rewards, done, _ = env.step({})
            for i in ids:
                total[i] += rewards[i]
            steps += 1
    else:
        for a in agents.values():
            a.start_episode()
        act = {i: env.project(i, agents[i].select_action(_state(agents[i], obs[i]))) for i in ids}
        done = False
        while not done:
            obs_next, rewards, done, _ = env.step(act)
            act_next = {i: env.project(i, agents[i].select_action(_state(agents[i], obs_next[i])))
                        for i in ids}
            for i in ids:
                ag = agents[i]
                ag.update(_state(ag, obs[i]), act[i], rewards[i], _state(ag, obs_next[i]), act_next[i])
                total[i] += rewards[i]
            obs, act = obs_next, act_next
            steps += 1
        for a in agents.values():
            a.end_episode()
    summary = env.episode_summary()
    per_agent = summary.pop("per_agent")
    row = {**summary, "steps": steps, "duration_s": round(time.time() - t0, 2)}
    steps = max(steps, 1)
    row["reward"] = float(np.mean([total[i] / steps for i in ids]))   # network mean reward per step
    row["reward_sum"] = float(np.mean([total[i] for i in ids]))
    for i in ids:
        row[f"reward_I{i}"] = total[i] / steps
        row[f"queue_I{i}"] = per_agent[i]["queue"]
        row[f"wait_I{i}"] = per_agent[i]["wait"]
        row[f"speed_I{i}"] = per_agent[i]["speed"]
    return row


def train_multi(cfg) -> None:
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_metadata(out, cfg)
    env = MultiIntersectionEnv(cfg, label=f"marl_{cfg.ALGORITHM}_{cfg.SEED}")
    agents = make_agents(cfg, env)
    fields = BASE_FIELDS + [f"{k}_I{i}" for i in env.agent_ids for k in ("reward", "queue", "wait", "speed")]
    logger = EpisodeLogger(out / "episodes.csv", fields)
    ckpt = out / "checkpoint.pkl"
    start = 0
    if agents and cfg.RESUME:
        last = load_checkpoint(ckpt, {str(k): v for k, v in agents.items()})
        if last >= 0:
            logger.truncate(last)
            start = last + 1
            print(f"Resuming from episode {start}")
    n_episodes = cfg.FIXED_EPISODES if not agents else cfg.EPISODES
    print(f"Run directory: {out}\nAgents: {env.n_agents}, discrete states per agent: {env.n_states}, "
          f"neighbours in the state: {env.use_neighbors}")
    t0 = time.time()
    try:
        for ep in range(start, n_episodes):
            eps_used = float(np.mean([a.epsilon for a in agents.values()])) if agents else 0.0
            row = run_episode(env, agents, ep)
            if ep == start:
                print("Traffic lights:", env.tls)
            for a in agents.values():
                a.decay_epsilon()
            row.update(episode=ep, epsilon=eps_used)
            logger.log(row)
            progress(cfg, ep, row, t0)
            if agents and ((ep + 1) % cfg.CHECKPOINT_EVERY == 0 or ep + 1 == n_episodes):
                save_checkpoint(ckpt, ep, {str(k): v for k, v in agents.items()})
    finally:
        env.close()
    if agents:
        save_tables(out / "qtables.npz", {f"I{k}": v for k, v in agents.items()})
    print(f"Done. Results in {out}")
