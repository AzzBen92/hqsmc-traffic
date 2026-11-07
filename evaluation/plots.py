"""Figures of Chapters 5, 6 and 7 from the per-episode CSV files.

    python -m evaluation.plots --scenario sefrou     # Figures 6.4 to 6.13
    python -m evaluation.plots --scenario saiss      # Figures 7.7 to 7.13
    python -m evaluation.plots --scenario isolated   # Figure 5.3

Curves are averaged over the available seeds and smoothed by a moving average
(SMOOTH_WINDOW episodes). Figures are written as PDF and PNG to figures/<scenario>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import load_config  # noqa: E402

from .metrics import (MARL_LABELS, METHOD_LABELS, convergence_episode, final_values,  # noqa: E402
                      load_runs, mean_curve, moving_average)

# Categorical palette (validated for colour-vision deficiencies) + one line style per method,
# so that the methods remain distinguishable in grey-scale print.
STYLE = {
    "hybrid":     {"color": "#2a78d6", "ls": "-",  "lw": 2.0, "z": 5},
    "qlearning":  {"color": "#eb6834", "ls": "--", "lw": 1.4, "z": 4},
    "sarsa":      {"color": "#1baf7a", "ls": "-.", "lw": 1.4, "z": 3},
    "montecarlo": {"color": "#eda100", "ls": ":",  "lw": 1.6, "z": 2},
    "dqn":        {"color": "#e87ba4", "ls": "--", "lw": 1.4, "z": 2},
    "ppo":        {"color": "#008300", "ls": "-.", "lw": 1.4, "z": 2},
}
REF_COLOR = "#5f5e5a"
ORDER = ["hybrid", "qlearning", "sarsa", "montecarlo", "dqn", "ppo"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3de", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})

YLABEL = {"reward": "Récompense moyenne", "queue": "File d'attente (véh.)", "wait": "Temps d'attente (s)",
          "speed": "Vitesse moyenne (m/s)", "co2": "Émissions de CO$_2$ (mg/s)",
          "fuel": "Consommation de carburant", "throughput": "Débit (véh./h)", "epsilon": r"$\varepsilon$"}


class Plotter:
    def __init__(self, cfg, runs: dict, labels: dict):
        self.cfg, self.runs, self.labels = cfg, runs, labels
        self.methods = [m for m in ORDER if m in runs]
        self.out = Path(cfg.FIGURES_DIR) / cfg.SCENARIO
        self.out.mkdir(parents=True, exist_ok=True)
        self.w = cfg.SMOOTH_WINDOW
        self.ref = self._reference()

    def _reference(self) -> dict:
        if "fixed" in self.runs:
            vals = [final_values(df, len(df)) for df in self.runs["fixed"].values()]
            return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        return dict(getattr(self.cfg, "REFERENCE", None) or {})

    # ------------------------------------------------------------------ helpers
    def curve(self, m, col, smooth=True):
        c = mean_curve(self.runs[m], col)
        return moving_average(c, self.w) if smooth else c

    def line(self, ax, m, col, x0=0, smooth=True):
        y = self.curve(m, col, smooth)
        st = STYLE[m]
        ax.plot(np.arange(x0, x0 + len(y)), y, color=st["color"], ls=st["ls"], lw=st["lw"],
                zorder=st["z"], label=self.labels.get(m, m))
        return y

    def reference_line(self, ax, key):
        if key in self.ref:
            ax.axhline(self.ref[key], color=REF_COLOR, ls=(0, (4, 3)), lw=1.1,
                       label=f"Référence ({self.ref[key]:.2f})")

    def conv(self, m):
        return convergence_episode(mean_curve(self.runs[m], "reward"), self.w,
                                   self.cfg.CONV_FRACTION, self.cfg.CONV_HOLD)

    def save(self, fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}")
        plt.close(fig)
        print("  ", self.out / f"{name}.pdf")

    def metric_figure(self, col, name, ref_key=None, title=None):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in self.methods:
            self.line(ax, m, col)
        if ref_key:
            self.reference_line(ax, ref_key)
        ax.set_xlabel("Épisode")
        ax.set_ylabel(YLABEL[col])
        if title:
            ax.set_title(title)
        ax.legend(frameon=False)
        self.save(fig, name)

    # ------------------------------------------------------------------ Chapter 6 figures
    def reward_individual(self):
        n = len(self.methods)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), squeeze=False, sharex=True)
        for ax, m in zip(axes.flat, self.methods):
            raw = self.curve(m, "reward", smooth=False)
            st = STYLE[m]
            ax.plot(raw, color=st["color"], alpha=0.18, lw=0.6)
            y = self.line(ax, m, "reward")
            t = self.conv(m)
            if t is not None:
                ax.plot(t, y[t], marker="D", ms=8, color="#0b0b0b", zorder=10)
                ax.annotate(f"t* = {t}", (t, y[t]), xytext=(8, -14), textcoords="offset points", fontsize=9)
            ax.set_title(self.labels.get(m, m))
            ax.set_ylabel(YLABEL["reward"])
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Épisode")
        fig.tight_layout()
        self.save(fig, "fig_reward_learning")

    def reward_zoom(self, start=2000, stop=9500):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in self.methods:
            y = self.curve(m, "reward")
            a, b = min(start, len(y)), min(stop, len(y))
            st = STYLE[m]
            ax.plot(np.arange(a, b), y[a:b], color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=self.labels.get(m, m))
            t = self.conv(m)
            if t is not None and a <= t < b:
                ax.plot(t, y[t], marker="D", ms=8, color=st["color"], markeredgecolor="#fcfcfb", zorder=10)
        ax.set_xlabel("Épisode")
        ax.set_ylabel(YLABEL["reward"])
        ax.legend(frameon=False)
        self.save(fig, "fig_zoom_convergence")

    def emissions(self):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col in zip(axes, ("co2", "fuel")):
            for m in self.methods:
                self.line(ax, m, col)
            self.reference_line(ax, col)
            ax.set_xlabel("Épisode")
            ax.set_ylabel(YLABEL[col])
        axes[0].legend(frameon=False)
        fig.tight_layout()
        self.save(fig, "fig_emissions")

    def epsilon(self):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for m in self.methods:
            if "epsilon" in next(iter(self.runs[m].values())) and m != "ppo":
                self.line(ax, m, "epsilon", smooth=False)
        ax.set_yscale("log")
        ax.set_xlabel("Épisode")
        ax.set_ylabel(YLABEL["epsilon"])
        ax.legend(frameon=False)
        self.save(fig, "fig_epsilon")

    def boxplot(self):
        n = min(len(mean_curve(self.runs[m], "reward")) for m in self.methods)
        mid, half = n // 2, min(500, n // 10)
        spans = [("Début", 0, min(100, n)), ("Milieu", max(mid - half, 0), min(mid + half, n)),
                 ("Fin", max(n - self.cfg.EVAL_LAST, 0), n)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        for ax, (name, a, b) in zip(axes, spans):
            title = f"{name} (épisodes {a + 1}-{b})"
            data = [self.curve(m, "reward", smooth=False)[a:b] for m in self.methods]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "#0b0b0b"})
            for patch, m in zip(bp["boxes"], self.methods):
                patch.set_facecolor(STYLE[m]["color"])
                patch.set_alpha(0.55)
            ax.set_xticks(range(1, len(self.methods) + 1),
                          [self.labels.get(m, m) for m in self.methods], rotation=20)
            ax.set_title(title)
        axes[0].set_ylabel(YLABEL["reward"])
        fig.tight_layout()
        self.save(fig, "fig_boxplot")

    def dashboard(self):
        cols = ["reward", "speed", "queue", "wait", "co2", "fuel"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
        for ax, col in zip(axes.flat, cols):
            for m in self.methods:
                y = self.line(ax, m, col)
                t = self.conv(m)
                if col == "reward" and t is not None:
                    ax.plot(t, y[t], marker="D", ms=7, color=STYLE[m]["color"], markeredgecolor="#fcfcfb")
            if col != "reward" and col in self.ref:   # the reference has no comparable reward
                self.reference_line(ax, col)
            ax.set_title(YLABEL[col])
        for ax in axes[-1]:
            ax.set_xlabel("Épisode")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.save(fig, "fig_dashboard")

    def heatmap(self):
        """Normalised performance (1 = best of the methods, 0 = worst) on each criterion."""
        crit = [("reward", True), ("speed", True), ("queue", False), ("wait", False),
                ("co2", False), ("fuel", False)]
        vals = {m: {k: np.mean([final_values(df, self.cfg.EVAL_LAST)[k] for df in self.runs[m].values()])
                    for k, _ in crit} for m in self.methods}
        conv = {m: self.conv(m) for m in self.methods}
        names = [YLABEL[k].split(" (")[0] for k, _ in crit]
        M = np.zeros((len(self.methods), len(crit) + 1))
        for j, (k, higher) in enumerate(crit):
            col = np.array([vals[m][k] for m in self.methods])
            rng = col.max() - col.min()
            M[:, j] = 0.5 if rng == 0 else ((col - col.min()) / rng if higher else (col.max() - col) / rng)
        c = np.array([conv[m] if conv[m] is not None else np.nan for m in self.methods], dtype=float)
        if np.isfinite(c).sum() >= 2:
            rng = np.nanmax(c) - np.nanmin(c)
            M[:, -1] = np.where(np.isfinite(c), (np.nanmax(c) - c) / rng if rng else 0.5, 0.0)
        names.append("Convergence↓")
        fig, ax = plt.subplots(figsize=(9, 0.7 * len(self.methods) + 1.5))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
        ax.set_yticks(range(len(self.methods)), [self.labels.get(m, m) for m in self.methods])
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="#0b0b0b" if 0.25 < M[i, j] < 0.8 else "#fcfcfb")
        fig.colorbar(im, ax=ax, fraction=0.03, label="Score normalisé")
        self.save(fig, "fig_heatmap")

    # ------------------------------------------------------------------ Chapter 7 figures
    def radar(self, other="qlearning"):
        if other not in self.methods or "hybrid" not in self.methods:
            return
        crit = [("reward", True), ("queue", False), ("wait", False), ("speed", True),
                ("co2", False), ("fuel", False)]
        tab = {m: {k: np.mean([final_values(df, self.cfg.EVAL_LAST)[k] for df in self.runs[m].values()])
                   for k, _ in crit} for m in self.methods if m in ("hybrid", "qlearning", "sarsa", "montecarlo")}
        scores = {}
        for m in ("hybrid", other):
            s = []
            for k, higher in crit:
                col = np.array([tab[x][k] for x in tab])
                rng = col.max() - col.min()
                v = tab[m][k]
                s.append(0.5 if rng == 0 else ((v - col.min()) / rng if higher else (col.max() - v) / rng))
            scores[m] = s
        ang = np.linspace(0, 2 * np.pi, len(crit), endpoint=False).tolist()
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(polar=True)
        for m, s in scores.items():
            st = STYLE[m]
            ax.plot(ang + ang[:1], s + s[:1], color=st["color"], ls=st["ls"], lw=2, label=self.labels.get(m, m))
            ax.fill(ang + ang[:1], s + s[:1], color=st["color"], alpha=0.12)
        ax.set_xticks(ang, [YLABEL[k].split(" (")[0] for k, _ in crit])
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0,25", "0,5", "0,75", "1"], fontsize=7, color="#5f5e5a")
        ax.set_rlabel_position(30)          # between two axes, away from the axis labels
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
        self.save(fig, "fig_radar_summary")

    def deep_convergence(self):
        ms = [m for m in ("hybrid", "dqn", "ppo") if m in self.methods]
        if len(ms) < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in ms:
            y = self.curve(m, "reward")
            lo, hi = y[0], np.mean(y[-self.cfg.EVAL_LAST:])
            yn = (y - lo) / (hi - lo) if hi != lo else np.zeros_like(y)
            st = STYLE[m]
            ax.plot(yn, color=st["color"], ls=st["ls"], lw=st["lw"], label=self.labels.get(m, m))
        ax.axhline(1.0, color=REF_COLOR, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Récompense normalisée (niveau convergé = 1)")
        ax.legend(frameon=False)
        self.save(fig, "fig_dqn_ppo_convergence")

    # ------------------------------------------------------------------ Chapter 5 figure
    def chapter5_panels(self):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        panels = [("wait", "(a) Délai moyen (s)"), ("queue", "(b) Longueur moyenne de file (véh.)"),
                  ("reward", "(c) Temps d'attente cumulé (s)"), ("throughput", "(d) Débit (véh./h)")]
        for ax, (col, title) in zip(axes.flat, panels):
            for m in self.methods:
                y = self.curve(m, col)
                if col == "reward":
                    y = -y      # reward = - waiting time
                st = STYLE[m]
                ax.plot(y, color=st["color"], ls=st["ls"], lw=st["lw"], label=self.labels.get(m, m))
            ax.set_title(title)
        for ax in axes[-1]:
            ax.set_xlabel("Épisode")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        self.save(fig, "fig_5_3_comparaison")


def make_figures(cfg, tag: str = ""):
    methods = ["hybrid", "qlearning", "sarsa", "montecarlo", "dqn", "ppo", "fixed"]
    runs = load_runs(cfg.RESULTS_DIR, cfg.SCENARIO, methods, cfg.SEEDS, tag)
    if not [m for m in runs if m != "fixed"]:
        raise SystemExit(f"No learning runs found under {cfg.RESULTS_DIR / cfg.SCENARIO}.")
    labels = MARL_LABELS if cfg.SCENARIO == "saiss" else METHOD_LABELS
    p = Plotter(cfg, runs, labels)
    print(f"Figures -> {p.out}")
    if cfg.SCENARIO == "isolated":
        p.chapter5_panels()
        return
    if cfg.SCENARIO == "sefrou":
        p.reward_individual()
        p.metric_figure("reward", "fig_reward_comparison")
        p.reward_zoom()
        p.metric_figure("speed", "fig_speed", ref_key="speed")
        p.metric_figure("queue", "fig_queue", ref_key="queue")
        p.emissions()
        p.epsilon()
        p.boxplot()
        p.dashboard()
        p.heatmap()
    else:
        tab = [m for m in p.methods if m in ("hybrid", "qlearning", "sarsa", "montecarlo")]
        deep = p.methods
        p.methods = tab
        p.metric_figure("reward", "fig_training_reward")
        p.metric_figure("queue", "fig_queue_length", ref_key="queue")
        p.metric_figure("wait", "fig_waiting_time")
        p.metric_figure("speed", "fig_vehicle_speed_curve", ref_key="speed")
        p.emissions()
        p.radar()
        p.methods = deep
        p.deep_convergence()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    over = {}
    if args.scenario:
        over["SCENARIO"] = args.scenario
    if args.results:
        over["RESULTS_DIR"] = Path(args.results)
    if args.figures:
        over["FIGURES_DIR"] = Path(args.figures)
    make_figures(load_config(**over), args.tag)


if __name__ == "__main__":
    main()
