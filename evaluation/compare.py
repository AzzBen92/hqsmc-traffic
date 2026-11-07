"""Result tables and statistical tests from the per-episode CSV files.

    python -m evaluation.compare --scenario sefrou          # Tables 6.2 to 6.4
    python -m evaluation.compare --scenario saiss           # Tables 7.8, 7.9 and 7.13
    python -m evaluation.compare --scenario saiss --demand high

Final metrics: mean over the last EVAL_LAST episodes of each run, then mean +/- std over the
seeds (the seed is the statistical unit). Rewards of the hybrid method vs each baseline:
two-sided Welch t-test, with Holm-Bonferroni correction over the comparisons.
Outputs are written to results/<scenario>/summary/ (CSV + Markdown).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import load_config

from .metrics import (MARL_LABELS, METHOD_LABELS, METRICS, convergence_episode, final_values,
                      holm_bonferroni, load_runs, mean_curve, seed_table)


def summarize(cfg, methods, tag: str = "", reference_key: str = "hybrid") -> dict:
    seeds = cfg.SEEDS
    runs = load_runs(cfg.RESULTS_DIR, cfg.SCENARIO, methods, seeds, tag)
    if not runs:
        raise SystemExit(f"No results found under {cfg.RESULTS_DIR / cfg.SCENARIO}.")
    labels = MARL_LABELS if cfg.SCENARIO == "saiss" else METHOD_LABELS
    last = cfg.EVAL_LAST

    # ---- final metrics, mean +/- std over seeds
    rows = []
    for m, by_seed in runs.items():
        vals = pd.DataFrame([final_values(df, last) for df in by_seed.values()])
        row = {"method": labels.get(m, m), "n_seeds": len(by_seed)}
        for k in METRICS:
            if k in vals:
                row[f"{k}_mean"] = vals[k].mean()
                row[f"{k}_std"] = vals[k].std(ddof=1) if len(vals) > 1 else 0.0
        if m != "fixed":
            curve = mean_curve(by_seed, "reward")
            row["convergence_episode"] = convergence_episode(curve, cfg.SMOOTH_WINDOW, cfg.CONV_FRACTION,
                                                             cfg.CONV_HOLD)
        rows.append(row)
    final = pd.DataFrame(rows).set_index("method")

    # ---- relative change w.r.t. the reference (fixed-time / native SUMO program)
    ref = getattr(cfg, "REFERENCE", None) or {}
    if "fixed" in runs:   # measured reference has priority over the indicative values
        fx = final.loc[labels["fixed"]]
        ref = {"speed": fx["speed_mean"], "queue": fx["queue_mean"], "co2": fx["co2_mean"],
               "fuel": fx.get("fuel_mean", np.nan)}
    for k in ("speed", "queue", "co2", "fuel"):
        if k in ref and f"{k}_mean" in final:
            final[f"delta_{k}_%"] = 100.0 * (final[f"{k}_mean"] - ref[k]) / ref[k]

    # ---- per-seed rewards and Welch tests
    learned = {m: r for m, r in runs.items() if m != "fixed"}
    per_seed = seed_table(learned, last, "reward")
    per_seed.index = [labels.get(m, m) for m in per_seed.index]
    tests = {}
    if reference_key in learned:
        ref_vals = per_seed.loc[labels[reference_key]].dropna().to_numpy()
        raw = {}
        for m in learned:
            if m == reference_key:
                continue
            other = per_seed.loc[labels.get(m, m)].dropna().to_numpy()
            if len(ref_vals) >= 2 and len(other) >= 2:
                raw[labels.get(m, m)] = float(stats.ttest_ind(ref_vals, other, equal_var=False).pvalue)
        adj = holm_bonferroni(raw) if raw else {}
        tests = {k: {"p_welch": raw[k], "p_holm": adj[k]} for k in raw}
    return {"final": final, "per_seed": per_seed, "tests": pd.DataFrame(tests).T, "runs": runs}


def _fmt(final: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=final.index)
    for k in METRICS:
        if f"{k}_mean" in final:
            digits = 3 if k == "reward" else (1 if k in ("co2", "fuel", "throughput") else 2)
            out[k] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(final[f"{k}_mean"], final[f"{k}_std"])]
    for c in final.columns:
        if c.startswith("delta_"):
            out[c] = final[c].map(lambda v: f"{v:+.1f} %")
    if "convergence_episode" in final:
        out["convergence"] = final["convergence_episode"]
    return out


def write_outputs(cfg, res: dict, name: str) -> Path:
    out = cfg.RESULTS_DIR / cfg.SCENARIO / "summary"
    out.mkdir(parents=True, exist_ok=True)
    res["final"].to_csv(out / f"{name}_final.csv")
    res["per_seed"].to_csv(out / f"{name}_per_seed.csv")
    if len(res["tests"]):
        res["tests"].to_csv(out / f"{name}_tests.csv")
    md = [f"# {cfg.SCENARIO} - {name}", "",
          f"Mean ± std over seeds, each run averaged over its last {cfg.EVAL_LAST} episodes.", "",
          _fmt(res["final"]).to_markdown(), "", "## Reward per seed", "",
          res["per_seed"].round(3).to_markdown()]
    if len(res["tests"]):
        md += ["", "## Welch t-test (hybrid vs baseline), Holm-Bonferroni correction", "",
               res["tests"].to_markdown(floatfmt=".4g")]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None, help="isolated | sefrou | saiss (default: config.py)")
    ap.add_argument("--demand", default=None, help="saiss only: low | medium | high")
    ap.add_argument("--results", default=None, help="results directory (default: config.RESULTS_DIR)")
    ap.add_argument("--tag", default="", help="RUN_TAG of the runs to compare")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    over = {}
    if args.scenario:
        over["SCENARIO"] = args.scenario
    if args.demand:
        over["DEMAND"] = args.demand
    if args.results:
        over["RESULTS_DIR"] = Path(args.results)
    cfg = load_config(**over)
    methods = args.methods or ["hybrid", "qlearning", "sarsa", "montecarlo", "fixed"]
    if cfg.SCENARIO == "saiss" and args.methods is None:
        methods += ["dqn", "ppo"]
    tag = args.tag
    if cfg.SCENARIO == "saiss" and cfg.DEMAND != "medium":
        tag = (tag + "_" if tag else "") + f"demand_{cfg.DEMAND}"
    res = summarize(cfg, methods, tag)
    write_outputs(cfg, res, f"comparison{'_' + tag if tag else ''}")


if __name__ == "__main__":
    main()
