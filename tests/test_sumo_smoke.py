"""End-to-end smoke tests with SUMO on the synthetic networks (skipped if SUMO is not installed).

    python scripts/make_test_networks.py   # once
    pytest -q tests/test_sumo_smoke.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from config import ROOT, load_config

try:
    from environment.sumo_utils import find_sumo_home
    find_sumo_home()
    HAS_SUMO = True
except Exception:  # noqa: BLE001
    HAS_SUMO = False

pytestmark = pytest.mark.skipif(not HAS_SUMO, reason="SUMO not installed")
GRID_IDS = dict(zip(range(1, 9), ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]))


@pytest.fixture(scope="module", autouse=True)
def networks():
    if not (ROOT / "map" / "demo_grid" / "osm.sumocfg").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_networks.py")], check=True)


@pytest.mark.parametrize("scenario,alg,cfgfile", [
    ("isolated", "hybrid", None),
    ("sefrou", "hybrid", "map/demo_single/osm.sumocfg"),
    ("sefrou", "montecarlo", "map/demo_single/osm.sumocfg"),
    ("sefrou", "fixed", "map/demo_single/osm.sumocfg"),
])
def test_single_agent(tmp_path, scenario, alg, cfgfile):
    from training.trainer import train
    over = dict(SCENARIO=scenario, ALGORITHM=alg, EPISODES=2, EPISODE_LENGTH=120, RESULTS_DIR=tmp_path,
                USE_LIBSUMO=True, RESUME=False)
    if cfgfile:
        over["SUMO_CFG"] = ROOT / cfgfile
    train(load_config(**over))
    df = pd.read_csv(next(Path(tmp_path).rglob("episodes.csv")))
    assert len(df) == (1 if alg == "fixed" else 2)
    assert (df["steps"] == 120).all()


@pytest.mark.parametrize("alg", ["hybrid", "qlearning", "sarsa", "dqn", "ppo", "fixed"])
def test_multi_agent(tmp_path, alg):
    from training.multi_trainer import train_multi
    cfg = load_config(SCENARIO="saiss", ALGORITHM=alg, EPISODES=2, EPISODE_LENGTH=120, RESULTS_DIR=tmp_path,
                      SUMO_CFG=ROOT / "map/demo_grid/osm.sumocfg", TLS_IDS=GRID_IDS, USE_LIBSUMO=True,
                      RESUME=False)
    train_multi(cfg)
    df = pd.read_csv(next(Path(tmp_path).rglob("episodes.csv")))
    assert (df["steps"] == 120).all()
    assert {"reward_I1", "queue_I8"} <= set(df.columns)


def test_resume_from_checkpoint(tmp_path):
    from training.multi_trainer import train_multi
    base = dict(SCENARIO="saiss", ALGORITHM="hybrid", EPISODE_LENGTH=60, RESULTS_DIR=tmp_path,
                SUMO_CFG=ROOT / "map/demo_grid/osm.sumocfg", TLS_IDS=GRID_IDS, USE_LIBSUMO=True,
                CHECKPOINT_EVERY=1)
    train_multi(load_config(EPISODES=2, **base))
    train_multi(load_config(EPISODES=4, **base))          # resumes at episode 2
    df = pd.read_csv(next(Path(tmp_path).rglob("episodes.csv")))
    assert list(df["episode"]) == [0, 1, 2, 3]
    assert df["epsilon"].is_monotonic_decreasing
