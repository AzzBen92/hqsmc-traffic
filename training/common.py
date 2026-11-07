"""Utilities shared by the training loops: CSV logging, checkpoints, run metadata."""

from __future__ import annotations

import csv
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np


class EpisodeLogger:
    """Appends one row per episode to <run_dir>/episodes.csv."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path, self.fieldnames = Path(path), fieldnames
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore").writerow(row)

    def truncate(self, last_episode: int) -> None:
        """Keep only the rows with episode <= last_episode (used when resuming)."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            rows = [r for r in csv.DictReader(f) if int(r["episode"]) <= last_episode]
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def save_checkpoint(path: Path, episode: int, agents: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"episode": episode,
                     "agents": {k: a.state_dict() for k, a in agents.items()}}, f)
    tmp.replace(path)


def load_checkpoint(path: Path, agents: dict) -> int:
    """Restore the agents; returns the last completed episode (-1 if there is no checkpoint)."""
    path = Path(path)
    if not path.exists():
        return -1
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    for k, a in agents.items():
        a.load_state_dict(ckpt["agents"][k])
    return int(ckpt["episode"])


def save_tables(path: Path, agents: dict) -> None:
    """Final Q-tables (.npz), one array per agent and per table."""
    arrays = {}
    for k, a in agents.items():
        if hasattr(a, "tables"):
            for name, t in a.tables().items():
                arrays[f"{k}_{name}"] = t
    if arrays:
        np.savez_compressed(path, **arrays)


def write_metadata(run_dir: Path, cfg) -> None:
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()
            if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None), Path))}
    meta["_python"] = platform.python_version()
    meta["_platform"] = platform.platform()
    meta["_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import traci  # type: ignore
        meta["_traci"] = getattr(traci, "__version__", "unknown")
    except ImportError:
        pass
    (Path(run_dir) / "config.json").write_text(json.dumps(meta, indent=2, default=str))


def progress(cfg, episode: int, row: dict, t0: float) -> None:
    if (episode + 1) % cfg.PRINT_EVERY == 0 or episode == 0:
        elapsed = time.time() - t0
        print(f"[{cfg.SCENARIO}/{cfg.ALGORITHM} seed={cfg.SEED}] ep {episode + 1}/{cfg.EPISODES}  "
              f"reward={row['reward']:.3f}  queue={row['queue']:.2f}  wait={row['wait']:.2f}  "
              f"speed={row['speed']:.2f}  eps={row['epsilon']:.4f}  ({elapsed:.0f}s)", flush=True)
