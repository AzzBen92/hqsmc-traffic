# H-QSMC: hybrid Q-learning – SARSA – Monte Carlo for adaptive traffic signal control

Code of the PhD thesis *Reinforcement learning and optimal control* (Azzeddine Ben Moussa,
L2MASI, Faculty of Sciences Dhar El Mahraz, Sidi Mohamed Ben Abdellah University, Fez) and
of the associated articles:

* A. Ben Moussa, A. Khazari, *Optimizing Traffic Signal Control Using Reinforcement Learning
  Methods: Hybrid Approach*, **Mathematical and Computational Applications** 31, 151 (2026) - Chapter 6.
* A. Ben Moussa, A. Khazari, *A Multi-Agent Reinforcement Learning Framework for Adaptive Traffic
  Signal Control in Real Urban Networks: Simulation, Performance Analysis, and Statistical
  Evaluation*, **Future Transportation** 6(5), 208 (2026),
  [doi:10.3390/futuretransp6050208](https://doi.org/10.3390/futuretransp6050208) - Chapter 7.
* A. Ben Moussa, A. Khazari, *Reinforcement Learning for Autonomous Driving: Optimization
  Strategies and Methodologies*, Springer Proc. Math. Stat. 538, pp. 369-385 (2026) - Chapter 4.

> **En bref (FR)** - Ce dépôt contient les simulations de la thèse : voiture autonome sur piste
> (chap. 4), intersection isolée (chap. 5), intersection de la Route de Sefrou à Fès avec
> H-QSMC (chap. 6) et réseau de huit intersections du quartier Saïss avec H-QSMC-MARL, DQN et
> PPO (chap. 7). Tout se règle dans `config.py` ; voir « Reproducing the thesis » ci-dessous.

## The methods

| Variant | Chapter | Q-tables | Weights (w_Q, w_S, w_MC) | Code |
|---|---|---|---|---|
| V1 | 4 | mix of three estimates | dynamic f(ξ) | `chapter4_autonomous_driving/hybrid_v1.py` |
| V2 | 5 | 1 | dynamic f(ξ) | `agents/hybrid.py` → `HybridAgent(weight_mode="dynamic")` |
| V3 **H-QSMC** | 6 | 1 | fixed (0.5, 0.3, 0.2) | `agents/hybrid.py` → `HybridAgent` |
| V4 **H-QSMC-MARL** | 7 | 3 per agent | fixed (0.5, 0.3, 0.2) | `agents/hybrid.py` → `HybridMARLAgent` |

**H-QSMC** (single table): at every step the Q-learning and SARSA TD errors, computed from the
same table, are applied together, `Q(s,a) += α [w_Q δ_QL + w_S δ_SA]`; at the end of the episode
a backward Monte Carlo pass applies `Q(s,a) += α w_MC [G − Q(s,a)]` once per state-action pair
(return of its last visit).

**H-QSMC-MARL**: each intersection is an independent agent with three tables updated by their
own rules (Q-learning and SARSA every step, Monte Carlo at the end of the episode) and combined at
decision time, `Q^H = 0.5 Q^Q + 0.3 Q^S + 0.2 Q^MC`. The local state includes the mean queue and
waiting time of the neighbouring intersections; a minimum-green safety constraint projects the
requested action, and the **executed** action is the one used in all updates.

## Repository layout

```
config.py                 scenario, method, seed, episodes, all hyper-parameters
train_multi.py            entry point, multi-agent Saiss network (Chapter 7)
environment/
  environment.py          one intersection: state, 4 actions, reward (Chapters 5 and 6)
  multi_environment.py    Saiss network: 8 agents, neighbour summary, safety constraint (Chapter 7)
  intersection.py         signal controller (min/max green, yellow, all-red) and measurements
  sumo_utils.py           SUMO / TraCI / libsumo start-up
agents/
  base_agent.py qlearning.py sarsa.py montecarlo.py hybrid.py   tabular methods
  deep.py                 DQN and PPO baselines (NumPy, CPU)
training/
  train.py trainer.py     single-agent training (python -m training.train)
  multi_trainer.py        multi-agent training, checkpoints and resume
evaluation/
  metrics.py compare.py plots.py   final metrics, Welch tests + Holm-Bonferroni, figures
scripts/
  build_network.py        OpenStreetMap -> SUMO network
  generate_demand.py      randomTrips demand (Sefrou, Saiss low/medium/high) + osm.sumocfg
  turning_ratios.py       turning movements realised at the 8 intersections (Table 7.6)
  run_experiments.py      all methods x seeds of a scenario (parallel)
  ablation.py             weight sensitivity (Tables 6.5, 7.11) and component ablation (Table 7.12)
  make_test_networks.py   synthetic networks to test the code without the Fes data
chapter4_autonomous_driving/   car-on-a-track environment (Gym API, optional Pygame) and agents
map/                      SUMO files (see map/README.md)
tests/                    unit tests and end-to-end SUMO tests (pytest)
docs/IMPLEMENTATION_NOTES.md   modelling choices, units, implementation details
```

## Installation

Python ≥ 3.10 and SUMO ≥ 1.27 (the thesis used SUMO 1.27.1).

```bash
git clone https://github.com/AzzBen92/hqsmc-traffic.git
cd hqsmc-traffic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs SUMO itself from PyPI (`eclipse-sumo`, with `traci`, `sumolib`,
`libsumo`). If you prefer the official SUMO installer, remove those lines and set the
`SUMO_HOME` environment variable.

Check the installation (builds small synthetic networks, then runs 27 tests, a few seconds):

```bash
python scripts/make_test_networks.py
pytest -q
```

## Reproducing the thesis

An experiment is defined in `config.py` by `SCENARIO`, `ALGORITHM`, `SEED` and `EPISODES`
(Appendix A of the thesis). All the other parameters (Tables 6.1 and 7.7) are in the same file.

### Chapter 4 - autonomous driving on a track
```bash
python chapter4_autonomous_driving/train_ch4.py            # 4 methods, 1000 episodes x 100 steps
python chapter4_autonomous_driving/train_ch4.py --render   # watch the agent (pip install pygame)
```

### Chapter 5 - isolated intersection (dynamic weights)
`map/isolated/` is created by `scripts/make_test_networks.py`. In `config.py`:
`SCENARIO = "isolated"`, `ALGORITHM = "hybrid"` (or `qlearning`, `sarsa`, `montecarlo`), then
```bash
python -m training.train
python -m evaluation.plots --scenario isolated
```

### Chapter 6 - Route de Sefrou, Fes (H-QSMC)
Put the SUMO files of the intersection in `map/sefrou/` (see `map/README.md`), set
`SCENARIO = "sefrou"` and run `python -m training.train`, or run the full protocol
(4 methods x 5 seeds {42, 10, 20, 30, 7} + native SUMO program):
```bash
python scripts/run_experiments.py --scenario sefrou --jobs 4 --libsumo
python scripts/ablation.py --scenario sefrou --part weights --jobs 4 --libsumo   # Table 6.5
python -m evaluation.compare --scenario sefrou     # Tables 6.2-6.4 (+ Welch tests)
python -m evaluation.plots   --scenario sefrou     # Figures 6.4-6.13
```

### Chapter 7 - Saiss network, eight intersections (H-QSMC-MARL)
Put the network and demand files in `map/saiss/`, then
```bash
python train_multi.py                                                   # one run from config.py
python scripts/run_experiments.py --scenario saiss --jobs 4 --libsumo   # 4 methods x 3 seeds
python scripts/run_experiments.py --scenario saiss --methods dqn ppo --jobs 4 --libsumo
python scripts/run_experiments.py --scenario saiss --demand low  --methods hybrid qlearning sarsa montecarlo
python scripts/run_experiments.py --scenario saiss --demand high --methods hybrid qlearning sarsa montecarlo
python scripts/ablation.py --scenario saiss --part weights    --jobs 4 --libsumo   # Table 7.11
python scripts/ablation.py --scenario saiss --part components --jobs 4 --libsumo   # Table 7.12
python -m evaluation.compare --scenario saiss                  # Tables 7.8, 7.9, 7.13
python -m evaluation.compare --scenario saiss --demand high    # Table 7.10 (one column)
python -m evaluation.plots   --scenario saiss                  # Figures 7.7-7.13
```
Each agent is attached to the traffic light closest to the coordinates of Table 7.3; the SUMO ids
can also be given explicitly in `config.py` (`TLS_IDS`). The ids found are printed at the start
of each run: check them once.

### Outputs
* `results/<scenario>/<method>[_<tag>]/seed_<seed>/episodes.csv` - one row per episode: mean
  reward per step, queue, waiting time, speed, CO2, fuel, throughput, epsilon (+ per intersection
  for Chapter 7); `qtables.npz`; `checkpoint.pkl` (an interrupted run resumes automatically);
  `config.json` (all the parameters of the run).
* `results/<scenario>/summary/` - tables (CSV + Markdown); `figures/<scenario>/` - PDF and PNG.

### Running time
10 000 episodes of 3 600 simulated seconds are long. On a laptop CPU, one episode of a single
four-arm intersection takes about 1-2 s with TraCI, and one episode of an eight-intersection
network about 5-12 s; real OSM networks with more traffic are slower. Use `--libsumo` (about
2x faster, identical results), `--jobs N` to run seeds in parallel, and the automatic resume.

## Reproducibility notes
* Each run is fully determined by its seed: the seed initialises the Python generator of the
  epsilon-greedy draws and SUMO (`--seed`). TraCI and libsumo give identical results.
* Results depend on the SUMO version, on the network and demand files, and on the operating
  system; re-runs give results of the same nature as the published ones, not necessarily the
  same digits.
* Implementation details not reported in the thesis (Chapter 5 set-up, g_min and state bins of
  Chapter 7, units of the SUMO fuel output, ...) are given in `docs/IMPLEMENTATION_NOTES.md`.

## Citation
See `CITATION.cff`. If you use this code, please cite the articles above.

## License
MIT - see `LICENSE`.
