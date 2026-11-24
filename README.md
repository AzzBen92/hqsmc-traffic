# H-QSMC: hybrid Q-learning – SARSA – Monte Carlo for adaptive traffic signal control

Simulation code of the hybrid reinforcement-learning methods H-QSMC (one intersection) and
H-QSMC-MARL (network of intersections), with SUMO, and of the associated articles:

* A. Ben Moussa, A. Khazari, *Optimizing Traffic Signal Control Using Reinforcement Learning
  Methods: Hybrid Approach*, **Mathematical and Computational Applications** 31, 151 (2026).
* A. Ben Moussa, A. Khazari, *A Multi-Agent Reinforcement Learning Framework for Adaptive Traffic
  Signal Control in Real Urban Networks: Simulation, Performance Analysis, and Statistical
  Evaluation*, **Future Transportation** 6(5), 208 (2026),
  [doi:10.3390/futuretransp6050208](https://doi.org/10.3390/futuretransp6050208).

> **En bref (FR)** - Ce dépôt contient les simulations de l'intersection de la Route de Sefrou à
> Fès avec H-QSMC et du réseau de huit intersections du quartier Saïss avec H-QSMC-MARL, DQN et
> PPO. Tout se règle dans `config.py` ; voir « Running the experiments » ci-dessous.

## The methods

| Method | Q-tables | Weights (w_Q, w_S, w_MC) | Code |
|---|---|---|---|
| **H-QSMC** (one intersection) | 1 | fixed (0.5, 0.3, 0.2) | `agents/hybrid.py` → `HybridAgent` |
| **H-QSMC-MARL** (network) | 3 per agent | fixed (0.5, 0.3, 0.2) | `agents/hybrid.py` → `HybridMARLAgent` |

**H-QSMC** (single table): at every step the Q-learning and SARSA TD errors, computed from the
same table, are applied together, `Q(s,a) += α [w_Q δ_QL + w_S δ_SA]`; at the end of the episode
a backward Monte Carlo pass applies `Q(s,a) += α w_MC [G − Q(s,a)]` once per state-action pair
(return of its last visit).

**H-QSMC-MARL**: each intersection is an independent agent with three tables updated by their
own rules (Q-learning and SARSA every step, Monte Carlo at the end of the episode) and combined at
decision time, `Q^H = 0.5 Q^Q + 0.3 Q^S + 0.2 Q^MC`. The local state includes the mean queue and
waiting time of the neighbouring intersections; a minimum-green safety constraint projects the
requested action, and the **executed** action is the one used in all updates.

Baselines: tabular Q-learning, SARSA and Monte Carlo, the native SUMO signal program, and (network
only) independent DQN and PPO agents.

## Repository layout

```
config.py                 scenario, method, seed, episodes, all hyper-parameters
train_multi.py            entry point, multi-agent Saiss network
environment/
  environment.py          one intersection: state, 4 actions, reward
  multi_environment.py    Saiss network: 8 agents, neighbour summary, safety constraint
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
  turning_ratios.py       turning movements realised at the 8 intersections
  run_experiments.py      all methods x seeds of a scenario (parallel)
  ablation.py             weight sensitivity and component ablation
  make_test_networks.py   synthetic networks to test the code without the Fes data
map/                      SUMO files (see map/README.md)
tests/                    unit tests and end-to-end SUMO tests (pytest)
docs/IMPLEMENTATION_NOTES.md   modelling choices, units, implementation details
```

## Installation

Python ≥ 3.10 and SUMO ≥ 1.27.

```bash
git clone https://github.com/AzzBen92/hqsmc-traffic.git
cd hqsmc-traffic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs SUMO itself from PyPI (`eclipse-sumo`, with `traci`, `sumolib`,
`libsumo`). If you prefer the official SUMO installer, remove those lines and set the
`SUMO_HOME` environment variable.

Check the installation (builds small synthetic networks, then runs the tests, a few seconds):

```bash
python scripts/make_test_networks.py
pytest -q
```

## Running the experiments

An experiment is defined in `config.py` by `SCENARIO`, `ALGORITHM`, `SEED` and `EPISODES`.
All the other parameters are in the same file.

### One intersection - Route de Sefrou, Fes (H-QSMC)
The SUMO files are in `map/sefrou/` (see `map/README.md`). Set `SCENARIO = "sefrou"` and run
`python -m training.train`, or run the full protocol (4 methods x 5 seeds {42, 10, 20, 30, 7}
+ native SUMO program):
```bash
python scripts/run_experiments.py --scenario sefrou --jobs 4 --libsumo
python scripts/ablation.py --scenario sefrou --part weights --jobs 4 --libsumo   # weight sensitivity
python -m evaluation.compare --scenario sefrou     # result tables + Welch tests
python -m evaluation.plots   --scenario sefrou     # figures
```

### Network of eight intersections - Saiss district, Fes (H-QSMC-MARL)
The SUMO files are in `map/saiss/`, then
```bash
python train_multi.py                                                   # one run from config.py
python scripts/run_experiments.py --scenario saiss --jobs 4 --libsumo   # 4 methods x 3 seeds
python scripts/run_experiments.py --scenario saiss --methods dqn ppo --jobs 4 --libsumo
python scripts/run_experiments.py --scenario saiss --demand low  --methods hybrid qlearning sarsa montecarlo
python scripts/run_experiments.py --scenario saiss --demand high --methods hybrid qlearning sarsa montecarlo
python scripts/ablation.py --scenario saiss --part weights    --jobs 4 --libsumo   # weight sensitivity
python scripts/ablation.py --scenario saiss --part components --jobs 4 --libsumo   # ablation
python -m evaluation.compare --scenario saiss                  # result tables + Welch tests
python -m evaluation.compare --scenario saiss --demand high    # demand robustness
python -m evaluation.plots   --scenario saiss                  # figures
```
Each agent is attached to the traffic light closest to the coordinates given in `config.py`
(`TLS_COORDS`); the SUMO ids can also be given explicitly (`TLS_IDS`). The ids found are printed at
the start of each run: check them once.

### Outputs
* `results/<scenario>/<method>[_<tag>]/seed_<seed>/episodes.csv` - one row per episode: mean
  reward per step, queue, waiting time, speed, CO2, fuel, throughput, epsilon (+ per intersection
  for the network); `qtables.npz`; `checkpoint.pkl` (an interrupted run resumes automatically);
  `config.json` (all the parameters of the run).
* `results/<scenario>/summary/` - tables (CSV + Markdown); `figures/<scenario>/` - PDF and PNG.

### Running time
10 000 episodes of 3 600 simulated seconds are long. On a laptop CPU, one episode of a single
four-arm intersection takes a few seconds, and one episode of the Saiss network about one minute.
Use `--libsumo` (faster, identical results), `--jobs N` to run seeds in parallel, and the
automatic resume.

## Reproducibility notes
* Each run is fully determined by its seed: the seed initialises the Python generator of the
  epsilon-greedy draws and SUMO (`--seed`). TraCI and libsumo give identical results.
* Results depend on the SUMO version, on the network and demand files, and on the operating
  system; re-runs give results of the same nature as the published ones, not necessarily the
  same digits.
* Implementation details (state bins, minimum green, units of the SUMO fuel output, ...) are given
  in `docs/IMPLEMENTATION_NOTES.md`.

## Citation
See `CITATION.cff`. If you use this code, please cite the articles above.

## License
MIT - see `LICENSE`.
