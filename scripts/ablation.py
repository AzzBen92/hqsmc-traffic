"""Weight sensitivity and ablation studies (seed 42).

    python scripts/ablation.py --scenario sefrou --part weights        # Table 6.5
    python scripts/ablation.py --scenario saiss  --part weights        # Table 7.11
    python scripts/ablation.py --scenario saiss  --part components     # Table 7.12
    python scripts/ablation.py --scenario saiss  --part components --summarize

Weights: nine configurations (w_Q, w_S, w_MC) of the simplex, including the three degenerate
cases (each algorithm alone). Components (Chapter 7):
    A  H-QSMC-MARL, with neighbours      B  hybrid, without neighbours
    C  Q-learning, with neighbours       D  Q-learning, without neighbours
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import load_config  # noqa: E402
from evaluation.metrics import final_values, run_path  # noqa: E402
from scripts.run_experiments import run_one  # noqa: E402

WEIGHTS = [
    (1.00, 0.00, 0.00, "Q-learning seul"),
    (0.00, 1.00, 0.00, "SARSA seul"),
    (0.00, 0.00, 1.00, "Monte Carlo seul"),
    (0.60, 0.30, 0.10, ""),
    (0.50, 0.30, 0.20, "retenue"),
    (0.40, 0.30, 0.30, ""),
    (0.50, 0.40, 0.10, ""),
    (0.50, 0.20, 0.30, ""),
    (0.33, 0.33, 0.34, "poids uniformes"),
]
COMPONENTS = {
    "A": ("hybrid", True, "H-QSMC-MARL (modèle complet)"),
    "B": ("hybrid", False, "Hybride, sans voisins"),
    "C": ("qlearning", True, "Q-learning, avec voisins"),
    "D": ("qlearning", False, "Q-learning, sans voisins"),
}


def weight_tag(w) -> str:
    return "w_" + "_".join(f"{int(round(100 * x)):03d}" for x in w[:3])


def jobs_for(args) -> list[dict]:
    base = {"SCENARIO": args.scenario, "SEED": args.seed}
    if args.episodes:
        base["EPISODES"] = args.episodes
    if args.libsumo:
        base["USE_LIBSUMO"] = True
    if args.part == "weights":
        return [{**base, "ALGORITHM": "hybrid", "HYBRID_WEIGHTS": w[:3], "RUN_TAG": weight_tag(w)}
                for w in WEIGHTS]
    return [{**base, "ALGORITHM": alg, "USE_NEIGHBORS": nb, "RUN_TAG": f"abl_{k}"}
            for k, (alg, nb, _) in COMPONENTS.items()]


def summarize(args):
    cfg = load_config(SCENARIO=args.scenario)
    rows = []
    for j in jobs_for(args):
        p = run_path(cfg.RESULTS_DIR, args.scenario, j["ALGORITHM"], args.seed, j["RUN_TAG"])
        if not p.exists():
            continue
        r = final_values(pd.read_csv(p), cfg.EVAL_LAST, ["reward"])["reward"]
        if args.part == "weights":
            w = j["HYBRID_WEIGHTS"]
            note = next(x[3] for x in WEIGHTS if tuple(x[:3]) == tuple(w))
            rows.append({"w_Q": w[0], "w_S": w[1], "w_MC": w[2], "reward": round(r, 3), "note": note})
        else:
            k = j["RUN_TAG"][-1]
            alg, nb, desc = COMPONENTS[k]
            rows.append({"config": k, "description": desc, "hybrid": alg == "hybrid",
                         "neighbours": nb, "reward": round(r, 3)})
    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / args.scenario / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ablation_{args.part}.csv", index=False)
    print(df.to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["sefrou", "saiss"])
    ap.add_argument("--part", required=True, choices=["weights", "components"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--libsumo", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="only build the table from existing runs")
    args = ap.parse_args()
    if args.part == "components" and args.scenario != "saiss":
        raise SystemExit("The component ablation (neighbour information) concerns the Saiss network only.")
    if not args.summarize:
        jobs = jobs_for(args)
        if args.jobs <= 1:
            for j in jobs:
                run_one(j)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for f in as_completed([pool.submit(run_one, j) for j in jobs]):
                    print("finished:", f.result())
    summarize(args)


if __name__ == "__main__":
    main()
