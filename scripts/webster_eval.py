"""Webster fixed-time plan vs native SUMO program (perspective of Section 7.7 / jury question).

Builds a Webster plan with SUMO's official tool (tlsCycleAdaptation.py, Webster's formula),
then evaluates BOTH the native program and the Webster plan with exactly the same monitor,
reward (Eq. 6.2) and seeds as the thesis experiments, and prints a comparison table with a
paired t-test (same seeds => paired comparison) and a Welch t-test.

Usage, from the project root (SUMO 1.27+ installed, e.g. `pip install eclipse-sumo`):

    python scripts/webster_eval.py --scenario sefrou                 # native vs Webster
    python scripts/webster_eval.py --scenario sefrou --min-green 10  # g_min = 10 s (thesis constraint)
    python scripts/webster_eval.py --scenario saiss                  # all eight intersections

Results are appended to results/webster_<scenario>.csv (one row per controller x seed).
The RL methods are NOT retrained here: compare the table printed by this script with the
values obtained for H-QSMC on the same machine and the same demand files.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from environment.sumo_utils import find_sumo_home  # noqa: E402

FIELDS = ["controller", "seed", "reward", "queue", "wait", "speed", "co2", "fuel", "vehicles"]


def build_webster(scenario: str, min_green: int | None) -> Path:
    """Run tlsCycleAdaptation.py (Webster) and force the resulting programs to static."""
    folder = ROOT / "map" / scenario
    net = next(p for p in (folder / "osm.net.xml.gz", folder / "isolated.net.xml") if p.exists())
    routes = next((folder / n for n in ("osm.passenger.rou.xml", "osm.passenger.trips.xml", "isolated.rou.xml")
                   if (folder / n).exists()), None)
    if routes is None:
        raise SystemExit(f"No route/trip file found in {folder}.")
    tag = f"_g{min_green}" if min_green else ""
    out = folder / f"webster{tag}.add.xml"

    def adapt(route_file: Path) -> None:
        cmd = [sys.executable, str(find_sumo_home() / "tools" / "tlsCycleAdaptation.py"),
               "-n", str(net), "-r", str(route_file), "-o", str(out), "--program", "W"]
        if min_green:
            cmd += ["--min-green", str(min_green)]
        subprocess.run(cmd, check=True)

    adapt(routes)
    if "<tlLogic" not in out.read_text():
        # tlsCycleAdaptation ignores <flow> demand (e.g. probability flows of the
        # isolated scenario): expand the flows into routed vehicles first.
        import re
        txt = re.sub(r'probability="([0-9.]+)"',
                     lambda m: f'vehsPerHour="{float(m.group(1)) * 3600:.0f}"',
                     routes.read_text())
        conv = folder / "webster_input.rou.xml"
        conv.write_text(txt)
        routed = folder / "webster_routed.rou.xml"
        subprocess.run([str(find_sumo_home() / "bin" / "duarouter"), "-n", str(net),
                        "-r", str(conv), "-o", str(routed), "--seed", "42"], check=True)
        adapt(routed)
    # Webster is a fixed-time method: make the generated programs static.
    out.write_text(out.read_text().replace('type="actuated"', 'type="static"'))
    return out


def run_episode_fixed(scenario: str, seed: int, extra: list[str], program: str | None,
                      label: str) -> dict:
    """One deterministic episode without agent, native program or Webster program."""
    cfg = load_config(SCENARIO=scenario, ALGORITHM="fixed", SEED=seed, USE_LIBSUMO=True,
                      SUMO_EXTRA_ARGS=extra)
    if scenario == "saiss":
        from environment.multi_environment import MultiTrafficSignalEnv as Env
    else:
        from environment.environment import TrafficSignalEnv as Env
    env = Env(cfg, label=f"{label}{seed}")
    env.reset(0)
    tls_ids = list(env.tls.values()) if scenario == "saiss" else [env.tls_id]
    if program:
        for tid in tls_ids:
            env.traci.trafficlight.setProgram(tid, program)
    total, steps, done = 0.0, 0, False
    while not done:
        _, r, done, _ = env.step({} if scenario == "saiss" else 0)
        total += sum(r.values()) / len(r) if isinstance(r, dict) else r   # saiss: network reward
        steps += 1
    s = env.episode_summary()
    per = s.pop("per_agent", None)
    if per:                                     # saiss: summary is already the network mean
        s["vehicles"] = sum(a.get("vehicles", 0) for a in per.values())
    s["reward"] = total / max(steps, 1)
    env.close()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="sefrou", choices=["isolated", "sefrou", "saiss"])
    ap.add_argument("--min-green", type=int, default=None,
                    help="minimum green (s) passed to Webster, e.g. 10 = thesis g_min")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="default: the seeds of the scenario (config.py)")
    args = ap.parse_args()

    cfg = load_config(SCENARIO=args.scenario, ALGORITHM="fixed")
    seeds = args.seeds or cfg.SEEDS
    add = build_webster(args.scenario, args.min_green)
    controllers = [("native", [], None), ("webster", ["-a", str(add)], "W")]

    rows: dict[str, list[dict]] = {}
    for name, extra, prog in controllers:
        rows[name] = []
        for seed in seeds:
            s = run_episode_fixed(args.scenario, seed, extra, prog, label=name[:3])
            rows[name].append({"controller": name, "seed": seed, **{k: s.get(k) for k in FIELDS[2:]}})
            print(f"{name:8s} seed {seed:3d}  reward {s['reward']:9.3f}  queue {s['queue']:6.2f}  "
                  f"wait {s['wait']:7.2f}  speed {s['speed']:5.2f}  co2 {s['co2']:9.1f}  fuel {s['fuel']:8.1f}")

    out = cfg.RESULTS_DIR / f"webster_{args.scenario}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for rs in rows.values():
            w.writerows(rs)

    print("\n=== Mean over seeds", seeds, "===")
    for name, rs in rows.items():
        print(f"{name:8s} " + "  ".join(f"{k} {st.mean(r[k] for r in rs):9.2f}" for k in FIELDS[2:8]))
    a = [r["reward"] for r in rows["native"]]
    b = [r["reward"] for r in rows["webster"]]
    d = [y - x for x, y in zip(a, b)]
    if len(seeds) >= 3:
        from scipy import stats
        print(f"\nWebster - native (reward): mean diff {st.mean(d):+.3f}, "
              f"better on {sum(x > 0 for x in d)}/{len(d)} seeds")
        print(f"paired t-test p = {stats.ttest_rel(b, a).pvalue:.4f}   "
              f"Welch t-test p = {stats.ttest_ind(b, a, equal_var=False).pvalue:.4f}")
    print(f"\nRows appended to {out}")


if __name__ == "__main__":
    main()
