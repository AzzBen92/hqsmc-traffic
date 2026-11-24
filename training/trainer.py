"""Single-agent training loop (Chapters 5 and 6, Algorithms 5 and 6).

At every step:   execute a_t -> observe r_{t+1}, s_{t+1} -> select a_{t+1} -> update(s_t, a_t, r, s_{t+1}, a_{t+1})
End of episode:  Monte Carlo pass (Monte Carlo and hybrid methods), then epsilon <- max(eps_min, d eps)
"""

from __future__ import annotations

import time

from agents import TABULAR, HybridAgent
from config import run_dir
from environment.environment import TrafficSignalEnv

from .common import (EpisodeLogger, load_checkpoint, progress, save_checkpoint, save_tables,
                     write_metadata)

FIELDS = ["episode", "reward", "reward_sum", "queue", "wait", "speed", "co2", "fuel", "throughput",
          "switches", "epsilon", "w_q", "w_s", "w_mc", "steps", "duration_s"]


def make_agent(cfg, n_states: int, n_actions: int):
    common = dict(alpha=cfg.ALPHA, gamma=cfg.GAMMA, epsilon_start=cfg.EPSILON_START,
                  epsilon_decay=cfg.EPSILON_DECAY, epsilon_min=cfg.EPSILON_MIN, seed=cfg.SEED,
                  mc_visit=cfg.MC_VISIT)
    if cfg.ALGORITHM == "hybrid":
        return HybridAgent(n_states, n_actions, weights=cfg.HYBRID_WEIGHTS,
                           weight_mode=cfg.HYBRID_WEIGHT_MODE, **common)
    if cfg.ALGORITHM in TABULAR:
        return TABULAR[cfg.ALGORITHM](n_states, n_actions, **common)
    return None   # "fixed": native SUMO program


def run_episode(env: TrafficSignalEnv, agent, episode: int) -> dict:
    t0 = time.time()
    s = env.reset(episode)
    total, steps = 0.0, 0
    if agent is None:
        done = False
        while not done:
            s, r, done, _ = env.step(0)
            total += r
            steps += 1
    else:
        agent.start_episode()
        a = agent.select_action(s)
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = agent.select_action(s_next)
            agent.update(s, a, r, s_next, a_next)
            s, a = s_next, a_next
            total += r
            steps += 1
        agent.end_episode()
    summary = env.episode_summary()
    summary.update(reward=total / max(steps, 1), reward_sum=total, steps=steps,
                   duration_s=round(time.time() - t0, 2))
    return summary


def train(cfg) -> None:
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_metadata(out, cfg)
    env = TrafficSignalEnv(cfg, label=f"{cfg.ALGORITHM}_{cfg.SEED}")
    agent = make_agent(cfg, env.n_states, env.n_actions)
    agents = {"agent": agent} if agent is not None else {}
    logger = EpisodeLogger(out / "episodes.csv", FIELDS)
    ckpt = out / "checkpoint.pkl"
    start = 0
    if agents and cfg.RESUME:
        last = load_checkpoint(ckpt, agents)
        if last >= 0:
            logger.truncate(last)
            start = last + 1
            print(f"Resuming from episode {start}")
    n_episodes = cfg.FIXED_EPISODES if agent is None else cfg.EPISODES
    print(f"Run directory: {out}\nStates: {env.n_states}, actions: {env.n_actions}")
    t0 = time.time()
    try:
        for ep in range(start, n_episodes):
            if isinstance(agent, HybridAgent):
                agent.set_progress((ep + 1) / cfg.EPISODES)      # xi (dynamic weights only)
            eps_used = agent.epsilon if agent else 0.0
            summary = run_episode(env, agent, ep)
            if agent:
                agent.decay_epsilon()
            w = agent.weights if isinstance(agent, HybridAgent) else ("", "", "")
            row = {"episode": ep, "epsilon": eps_used, "w_q": w[0], "w_s": w[1], "w_mc": w[2], **summary}
            logger.log(row)
            progress(cfg, ep, row, t0)
            if agents and ((ep + 1) % cfg.CHECKPOINT_EVERY == 0 or ep + 1 == n_episodes):
                save_checkpoint(ckpt, ep, agents)
    finally:
        env.close()
    if agents:
        save_tables(out / "qtables.npz", agents)
    print(f"Done. Results in {out}")
