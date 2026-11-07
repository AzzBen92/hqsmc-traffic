"""Indicators computed from the per-episode CSV files (Sections 6.4 and 7.5.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["reward", "queue", "wait", "speed", "co2", "fuel", "throughput"]
METHOD_LABELS = {"hybrid": "Hybride", "qlearning": "Q-learning", "sarsa": "SARSA",
                 "montecarlo": "Monte Carlo", "dqn": "DQN", "ppo": "PPO", "fixed": "Référence"}
MARL_LABELS = {**METHOD_LABELS, "hybrid": "H-QSMC-MARL"}


def run_path(results_dir: Path, scenario: str, method: str, seed: int, tag: str = "") -> Path:
    name = method + (f"_{tag}" if tag else "")
    return Path(results_dir) / scenario / name / f"seed_{seed}" / "episodes.csv"


def load_runs(results_dir: Path, scenario: str, methods, seeds, tag: str = "") -> dict:
    """{method: {seed: DataFrame}} for the runs that exist on disk."""
    runs = {}
    for m in methods:
        for s in seeds:
            p = run_path(results_dir, scenario, m, s, tag)
            if p.exists():
                runs.setdefault(m, {})[s] = pd.read_csv(p)
    return runs


def moving_average(x, window: int) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def convergence_episode(rewards, window: int = 200, fraction: float = 0.9, hold: int = 200):
    """First episode from which the smoothed reward reaches `fraction` of its final value and stays
    there for at least `hold` consecutive episodes (Section 6.4). Returns None if never reached.

    This is the thesis definition when the final reward is positive (as in Chapters 6 and 7).
    When it is negative or zero (e.g. Chapter 5, reward = - waiting time), "90 % of the final value"
    is ill-defined, and the criterion becomes 90 % of the progress from the initial to the final
    smoothed value: smoothed >= initial + fraction * (final - initial).
    """
    sm = moving_average(rewards, window)
    if len(sm) == 0:
        return None
    final = sm[-1]
    threshold = fraction * final if final > 0 else sm[0] + fraction * (final - sm[0])
    ok = sm >= threshold
    hold = min(hold, len(ok))
    # first t such that ok[t : t + hold] is all True
    csum = np.concatenate([[0], np.cumsum(ok)])
    for t in range(0, len(ok) - hold + 1):
        if csum[t + hold] - csum[t] == hold:
            return int(t)
    return None


def final_values(df: pd.DataFrame, last: int, metrics=METRICS) -> dict:
    tail = df.tail(last)
    return {m: float(tail[m].mean()) for m in metrics if m in tail}


def seed_table(runs: dict, last: int, metric: str = "reward") -> pd.DataFrame:
    """Rows = methods, columns = seeds: mean of `metric` over the last `last` episodes."""
    data = {m: {s: final_values(df, last, [metric])[metric] for s, df in by_seed.items()}
            for m, by_seed in runs.items()}
    return pd.DataFrame(data).T.sort_index(axis=1)


def mean_curve(by_seed: dict, column: str) -> np.ndarray:
    """Seed-averaged curve (episodes truncated to the shortest run)."""
    arrays = [df[column].to_numpy(dtype=float) for df in by_seed.values() if column in df]
    n = min(len(a) for a in arrays)
    return np.mean([a[:n] for a in arrays], axis=0)


def holm_bonferroni(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        adjusted[name] = running
    return adjusted
