"""
Central configuration of the experiments (thesis, Appendix A).

As described in the thesis, an experiment is fully defined by editing this file:
    SCENARIO   -> "isolated" (Chapter 5), "sefrou" (Chapter 6) or "saiss" (Chapter 7)
    ALGORITHM  -> "qlearning", "sarsa", "montecarlo", "hybrid", "fixed"
                  (+ "dqn" and "ppo" for the Saiss network, Chapter 7)
    SEED       -> random seed (Python epsilon-greedy draws AND SUMO --seed)
    EPISODES   -> number of training episodes

Then, from the project root:
    python -m training.train      # one intersection (Chapters 5 and 6)
    python train_multi.py         # Saiss network, eight intersections (Chapter 7)

Scripts that run several seeds or methods (scripts/run_experiments.py) call
`load_config(**overrides)` instead of editing this file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent

# =============================================================================
# 1. Experiment selection (edit these four lines)
# =============================================================================
SCENARIO = "sefrou"
ALGORITHM = "hybrid"
SEED = 42
EPISODES = 10_000

# Optional tag appended to the results folder (e.g. "w_060_030_010", "no_neighbors")
RUN_TAG = ""

# The "fixed" baseline (native SUMO signal program) is deterministic for a given seed:
# a few episodes are enough to measure it.
FIXED_EPISODES = 1
PRINT_EVERY = 10             # progress line every N episodes

# =============================================================================
# 2. SUMO settings (common)
# =============================================================================
SUMO_HOME = os.environ.get("SUMO_HOME", "")   # auto-detected if empty (see environment/sumo_utils.py)
SUMO_GUI = False             # True to watch the simulation in sumo-gui
USE_LIBSUMO = False          # True = in-process libsumo (2-5x faster, no GUI)
EPISODE_LENGTH = 3600        # s, one simulated hour (Tables 6.1 and 7.7)
STEP_LENGTH = 1.0            # s, simulation AND decision step (Delta t = 1 s)
SUMO_SEED_PER_EPISODE = False  # False: SUMO --seed = SEED for every episode (thesis setting)
TIME_TO_TELEPORT = 300       # SUMO default
SUMO_EXTRA_ARGS: list[str] = []

# =============================================================================
# 3. Learning hyper-parameters (common to Chapters 5, 6 and 7)
# =============================================================================
ALPHA = 0.10                 # learning rate
GAMMA = 0.95                 # discount factor
EPSILON_START = 1.0
EPSILON_DECAY = 0.9993       # multiplicative decay per episode
EPSILON_MIN = 0.005
HYBRID_WEIGHTS = (0.5, 0.3, 0.2)   # (w_Q, w_S, w_MC), Chapters 6 and 7
MC_VISIT = "last"            # "last": each (s,a) updated once per episode, with the return of its LAST visit

# Multi-objective reward (Eq. 6.2 and 7.6):
#   r = -2.0 q - 0.05 w + 1.5 v - 1e-3 CO2 - 1e-3 fuel
REWARD_COEFS = {"queue": -2.0, "wait": -0.05, "speed": 1.5, "co2": -1e-3, "fuel": -1e-3}

# =============================================================================
# 4. Evaluation
# =============================================================================
EVAL_LAST = 500              # final metrics = mean over the last 500 episodes
SMOOTH_WINDOW = 200          # moving average used for curves and convergence
CONV_FRACTION = 0.90         # convergence: smoothed reward reaches 90 % of its final value ...
CONV_HOLD = 200              # ... and stays there for at least 200 consecutive episodes
CHECKPOINT_EVERY = 100       # save Q-tables + RNG state every N episodes
RESUME = True                # resume from the last checkpoint if it exists
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# =============================================================================
# 5. Scenario-specific settings
# =============================================================================
SCENARIO_SETTINGS = {
    # ---------------------------------------------------------------- Chapter 5
    # Isolated intersection, hybrid method with DYNAMIC weights (variant V2).
    # Preliminary study: the detailed configuration (state variables, demand, ...)
    # is documented in docs/IMPLEMENTATION_NOTES.md.
    "isolated": {
        "SUMO_CFG": ROOT / "map" / "isolated" / "isolated.sumocfg",
        "TLS_ID": None,
        "TLS_COORD": None,
        "REWARD_MODE": "waiting",           # r = - cumulative waiting time
        "HYBRID_WEIGHT_MODE": "dynamic",    # w = f(xi), xi = episode / EPISODES
        "MIN_GREEN": 10, "MAX_GREEN": 60, "YELLOW": 3, "ALL_RED": 2,
        "QUEUE_BINS": 5, "QUEUE_MAX": 20.0,
        "WAIT_BINS": 5, "WAIT_MAX": 120.0,
        "SPEED_BINS": 5, "SPEED_MAX": 20.0,
        "EVAL_LAST": 100,
        "SMOOTH_WINDOW": 100,
        "SEEDS": [42],
    },
    # ---------------------------------------------------------------- Chapter 6
    # Route de Sefrou - Boulevard Moulay Rachid, Fes (Table 6.1).
    "sefrou": {
        "SUMO_CFG": ROOT / "map" / "sefrou" / "osm.sumocfg",
        "TLS_ID": None,          # SUMO id of the studied traffic light (None = auto)
        "TLS_COORD": None,       # (x, y) in network coordinates, used if TLS_ID is None
        "REWARD_MODE": "multi",
        "HYBRID_WEIGHT_MODE": "fixed",
        "MIN_GREEN": 10, "MAX_GREEN": 60, "YELLOW": 3, "ALL_RED": 2,
        # State discretisation (Section 6.2.2): 5 uniform bins, out-of-range -> nearest bin
        "QUEUE_BINS": 5, "QUEUE_MAX": 20.0,      # vehicles, per approach (L = 4)
        "WAIT_BINS": 5, "WAIT_MAX": 120.0,       # s
        "SPEED_BINS": 5, "SPEED_MAX": 20.0,      # m/s
        "SEEDS": [42, 10, 20, 30, 7],
        # Fixed-time / native SUMO reference (Table 6.3)
        "REFERENCE": {"speed": 10.47, "queue": 6.81, "co2": 5000.0},
    },
    # ---------------------------------------------------------------- Chapter 7
    # Saiss district, Fes: eight signalised intersections (Tables 7.3 to 7.7).
    "saiss": {
        "SUMO_CFG": ROOT / "map" / "saiss" / "osm.sumocfg",
        "DEMAND": "medium",       # "low" (~3000 trips), "medium" (4313), "high" (~6200)
        "SUMO_CFG_BY_DEMAND": {
            "low": ROOT / "map" / "saiss" / "osm_low.sumocfg",
            "medium": ROOT / "map" / "saiss" / "osm.sumocfg",
            "high": ROOT / "map" / "saiss" / "osm_high.sumocfg",
        },
        # SUMO ids of the eight traffic lights I1..I8. If None, each agent is attached
        # to the traffic light closest to the coordinates of Table 7.3 (network x, y).
        "TLS_IDS": None,
        "TLS_COORDS": {
            1: (2372.0, 2998.0), 2: (934.0, 3650.0), 3: (1726.0, 3511.0), 4: (1488.0, 2891.0),
            5: (1884.0, 3177.0), 6: (2308.0, 2499.0), 7: (961.0, 2503.0), 8: (1261.0, 1624.0),
        },
        # Direct neighbours, fixed offline (Table 7.3)
        "NEIGHBORS": {
            1: [5, 6], 2: [3], 3: [2, 4, 5], 4: [3, 5, 7],
            5: [1, 3, 4], 6: [1], 7: [4, 8], 8: [7],
        },
        "USE_NEIGHBORS": True,    # False = ablation configurations B and D (Table 7.12)
        "REWARD_MODE": "multi",
        "MIN_GREEN": 10,          # g_min of the safety constraint (Eq. 7.14)
        "MAX_GREEN": None,        # no maximum green in Chapter 7
        "YELLOW": 3, "ALL_RED": 2,
        # Normalisation and discretisation of the local state (Table 7.2)
        "QUEUE_MAX": 20.0,        # min-max scaling of q on [0, QUEUE_MAX]
        "WAIT_MAX": 120.0,        # min-max scaling of w on [0, WAIT_MAX]
        "QUEUE_BINS": 5, "WAIT_BINS": 5, "SPEED_BINS": 5,
        "DURATION_BINS": 2,       # d / g_min in [0,1) -> 0, >= 1 -> 1 (switch admissible)
        "NEIGHBOR_BINS": 5,       # bins for each of the two neighbour summaries
        "SEEDS": [42, 10, 20],
        # Indicative fixed-time reference values (Section 7.5.4)
        "REFERENCE": {"speed": 10.47, "queue": 6.90, "co2": 5000.0, "fuel": 2000.0},
        # Deep MARL baselines (Section 7.6.10): default hyper-parameters, NumPy implementation
        "DQN": {"hidden": (64, 64), "lr": 1e-4, "buffer_size": 50_000, "batch_size": 32,
                "learning_starts": 1_000, "train_freq": 4, "target_update": 1_000,
                "huber_delta": 1.0, "max_grad_norm": 10.0},
        "PPO": {"hidden": (64, 64), "lr": 3e-4, "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "clip": 0.2, "gae_lambda": 0.95, "ent_coef": 0.0,
                "vf_coef": 0.5, "max_grad_norm": 0.5},
    },
}

VALID_ALGORITHMS = {
    "isolated": {"qlearning", "sarsa", "montecarlo", "hybrid", "fixed"},
    "sefrou": {"qlearning", "sarsa", "montecarlo", "hybrid", "fixed"},
    "saiss": {"qlearning", "sarsa", "montecarlo", "hybrid", "fixed", "dqn", "ppo"},
}


def load_config(**overrides) -> SimpleNamespace:
    """Return the configuration as a namespace.

    Order of precedence: overrides > scenario settings > common settings.
    Example: load_config(SCENARIO="saiss", ALGORITHM="sarsa", SEED=10)
    """
    g = globals()
    common = {k: copy.deepcopy(v) for k, v in g.items()
              if k.isupper() and k not in ("SCENARIO_SETTINGS", "VALID_ALGORITHMS")}
    scenario = overrides.get("SCENARIO", common["SCENARIO"])
    if scenario not in SCENARIO_SETTINGS:
        raise ValueError(f"Unknown SCENARIO '{scenario}'. Choose among {list(SCENARIO_SETTINGS)}.")
    cfg = {**common, **copy.deepcopy(SCENARIO_SETTINGS[scenario]), **overrides}
    cfg["SCENARIO"] = scenario
    if cfg["ALGORITHM"] not in VALID_ALGORITHMS[scenario]:
        raise ValueError(f"ALGORITHM '{cfg['ALGORITHM']}' is not available for scenario '{scenario}'.")
    if scenario == "saiss" and "SUMO_CFG" not in overrides:
        cfg["SUMO_CFG"] = cfg["SUMO_CFG_BY_DEMAND"][cfg["DEMAND"]]
    w = cfg["HYBRID_WEIGHTS"]
    if abs(sum(w) - 1.0) > 1e-9 or min(w) < 0:
        raise ValueError(f"HYBRID_WEIGHTS must be non-negative and sum to 1, got {w}.")
    cfg["SUMO_CFG"] = Path(cfg["SUMO_CFG"])
    cfg["RESULTS_DIR"] = Path(cfg["RESULTS_DIR"])
    cfg["FIGURES_DIR"] = Path(cfg["FIGURES_DIR"])
    return SimpleNamespace(**cfg)


def run_dir(cfg: SimpleNamespace) -> Path:
    """results/<scenario>/<algorithm>[_<tag>]/seed_<seed>/"""
    name = cfg.ALGORITHM + (f"_{cfg.RUN_TAG}" if cfg.RUN_TAG else "")
    if cfg.SCENARIO == "saiss" and getattr(cfg, "DEMAND", "medium") != "medium":
        name += f"_demand_{cfg.DEMAND}"
    return cfg.RESULTS_DIR / cfg.SCENARIO / name / f"seed_{cfg.SEED}"
