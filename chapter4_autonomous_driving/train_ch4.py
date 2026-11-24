"""Chapter 4: Q-learning, SARSA, Monte Carlo and hybrid (V1) agents driving a car on a track.

    python chapter4_autonomous_driving/train_ch4.py               # 1000 episodes x 100 steps
    python chapter4_autonomous_driving/train_ch4.py --render      # watch the hybrid agent (pygame)

Settings of Section 4.5.1: alpha = 0.1, gamma = 0.95, epsilon_0 = 1.0 with an exponential decay
per episode that differs by method (d = 0.9949 for Q-learning and hybrid, 0.9969 for SARSA,
0.9966 for Monte Carlo). Writes results/chapter4/<method>.csv, summary.csv and the figures of
Section 4.5.3 (figures/chapter4/).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import MonteCarloAgent, QLearningAgent, SarsaAgent  # noqa: E402
from hybrid_v1 import HybridV1Agent  # noqa: E402
from track_env import TrackEnv  # noqa: E402

DECAY = {"qlearning": 0.9949, "sarsa": 0.9969, "montecarlo": 0.9966, "hybrid": 0.9949}
CLASSES = {"qlearning": QLearningAgent, "sarsa": SarsaAgent, "montecarlo": MonteCarloAgent,
           "hybrid": HybridV1Agent}
LABELS = {"qlearning": "Q-learning", "sarsa": "SARSA", "montecarlo": "Monte Carlo", "hybrid": "Hybride"}


def run(method: str, episodes: int, seed: int, render: bool = False) -> pd.DataFrame:
    env = TrackEnv(seed=seed, render_mode="human" if render else None)
    agent = CLASSES[method](env.n_states, env.n_actions, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                            epsilon_decay=DECAY[method], epsilon_min=0.0, seed=seed)
    rows = []
    for ep in range(episodes):
        if hasattr(agent, "set_progress"):
            agent.set_progress((ep + 1) / episodes)
        s, _ = env.reset()
        agent.start_episode()
        a = agent.select_action(s)
        total, done = 0.0, False
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            a2 = agent.select_action(s2)
            agent.update(s, a, r, s2, a2)
            s, a, total, done = s2, a2, total + r, term or trunc
        agent.end_episode()
        rows.append({"episode": ep, "reward": total, "crashes": env.crashes, "epsilon": agent.epsilon})
        agent.decay_epsilon()
    env.close()
    return pd.DataFrame(rows)


def convergence(rewards: np.ndarray, window: int = 50):
    """First episode where the smoothed reward exceeds the mean reward of the whole run (Section 4.5.3)."""
    sm = pd.Series(rewards).rolling(window, min_periods=1).mean().to_numpy()
    idx = np.flatnonzero(sm > rewards.mean())
    return int(idx[0]) if len(idx) else None


def figures(results: dict, out: Path, window: int = 50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT))
    from evaluation.plots import STYLE
    out.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    means = [results[m]["reward"].mean() for m in methods]
    stds = [results[m]["reward"].std() for m in methods]
    axes[0].bar([LABELS[m] for m in methods], means, yerr=stds, capsize=4,
                color=[STYLE[m]["color"] for m in methods], edgecolor="#fcfcfb", linewidth=2)
    axes[0].set_ylabel("Récompense moyenne ± écart-type")
    axes[0].set_title("(a)")
    for m in methods:
        sm = results[m]["reward"].rolling(window, min_periods=1).mean()
        axes[1].plot(sm, color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"], label=LABELS[m])
    axes[1].set_xlabel("Épisode")
    axes[1].set_ylabel("Récompense lissée")
    axes[1].set_title("(b)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_4_4_recompenses.pdf")
    fig.savefig(out / "fig_4_4_recompenses.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        ax.plot(results[m]["epsilon"], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"],
                label=LABELS[m])
    ax.set_xlabel("Épisode")
    ax.set_ylabel(r"$\varepsilon$")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_4_5_exploration.pdf")
    fig.savefig(out / "fig_4_5_exploration.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        ax.plot(results[m]["reward"].rolling(window, min_periods=2).std(), color=STYLE[m]["color"],
                ls=STYLE[m]["ls"], lw=STYLE[m]["lw"], label=LABELS[m])
    ax.set_xlabel("Épisode")
    ax.set_ylabel(f"Écart-type glissant ({window} épisodes)")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_4_6_stabilite.pdf")
    fig.savefig(out / "fig_4_6_stabilite.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="*", default=["qlearning", "sarsa", "montecarlo", "hybrid"])
    ap.add_argument("--render", action="store_true", help="watch the last method (needs pygame)")
    args = ap.parse_args()
    out = ROOT / "results" / "chapter4"
    out.mkdir(parents=True, exist_ok=True)
    results, summary = {}, []
    for m in args.methods:
        t0 = time.time()
        df = run(m, args.episodes, args.seed, render=args.render and m == args.methods[-1])
        dt = time.time() - t0
        df.to_csv(out / f"{m}.csv", index=False)
        results[m] = df
        summary.append({"method": LABELS[m], "time_s": round(dt, 2), "mean_reward": round(df.reward.mean(), 2),
                        "std_reward": round(df.reward.std(), 2),
                        "convergence_episode": convergence(df.reward.to_numpy())})
        print(summary[-1])
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    figures(results, ROOT / "figures" / "chapter4")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
