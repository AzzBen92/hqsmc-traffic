"""Run every method x seed of a scenario (the full protocol of Chapters 6 and 7).

    python scripts/run_experiments.py --scenario sefrou                  # 4 methods x 5 seeds + reference
    python scripts/run_experiments.py --scenario saiss --jobs 4          # 4 tabular methods x 3 seeds
    python scripts/run_experiments.py --scenario saiss --methods dqn ppo # deep MARL baselines
    python scripts/run_experiments.py --scenario saiss --demand high     # demand robustness (Table 7.10)

Each run is independent (its own SUMO process), so --jobs N runs N of them in parallel.
Runs resume from their last checkpoint if interrupted. Settings not given on the command line
are those of config.py.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402


def run_one(overrides: dict) -> str:
    sys.path.insert(0, str(ROOT))
    cfg = load_config(**overrides)
    if cfg.SCENARIO == "saiss":
        from training.multi_trainer import train_multi
        train_multi(cfg)
    else:
        from training.trainer import train
        train(cfg)
    return f"{cfg.SCENARIO}/{cfg.ALGORITHM}{'_' + cfg.RUN_TAG if cfg.RUN_TAG else ''} seed={cfg.SEED}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["isolated", "sefrou", "saiss"])
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: hybrid qlearning sarsa montecarlo fixed")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: SEEDS of the scenario")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--demand", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--libsumo", action="store_true", help="use libsumo (faster, no GUI)")
    args = ap.parse_args()

    base = {"SCENARIO": args.scenario}
    if args.episodes:
        base["EPISODES"] = args.episodes
    if args.demand:
        base["DEMAND"] = args.demand
    if args.tag:
        base["RUN_TAG"] = args.tag
    if args.libsumo:
        base["USE_LIBSUMO"] = True
    cfg = load_config(**base)
    methods = args.methods or ["hybrid", "qlearning", "sarsa", "montecarlo", "fixed"]
    seeds = args.seeds or cfg.SEEDS
    jobs = [{**base, "ALGORITHM": m, "SEED": s} for m in methods for s in seeds]
    print(f"{len(jobs)} runs: methods={methods}, seeds={seeds}")
    if args.jobs <= 1:
        for j in jobs:
            print("finished:", run_one(j))
        return
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for f in as_completed(futures):
            print("finished:", f.result())


if __name__ == "__main__":
    main()
