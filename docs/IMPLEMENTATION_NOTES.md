# Implementation notes

This file describes how each element of the thesis is implemented, including the details that
are not reported in the manuscript. Each of them is a parameter of `config.py` or is isolated in
one function, so that it can be changed in one place.

## Common to Chapters 5-7

| Element | Thesis | Implementation |
|---|---|---|
| Decision step | Δt = 1 s, T = 3600 s | one decision per simulated second, exactly `EPISODE_LENGTH / STEP_LENGTH` decisions (the initial observation uses one extra SUMO step) |
| Seeds | Python ε-greedy generator + SUMO `--seed` | `random.Random(seed)` per agent (`seed*100 + i` in the multi-agent case), SUMO `--seed SEED` at every episode (`SUMO_SEED_PER_EPISODE = True` to use `SEED + episode`) |
| Episode end | fixed horizon | time-limit truncation: the last transition still bootstraps on `Q(s_T, ·)`; the episode also stops if no vehicle remains |
| Monte Carlo | backward pass, each (s, a) once, last-visit return, constant α | `agents/base_agent.monte_carlo_pass(visit="last")`; `MC_VISIT = "every"` for the every-visit variant |
| Q-table init | zeros | zeros; ties of the greedy choice are broken at random |
| ε schedule | ε ← max(ε_min, d ε) once per episode | `decay_epsilon()`; the ε logged for an episode is the one used during it |
| Reward | r = −2 q − 0.05 w + 1.5 v − 10⁻³ e − 10⁻³ f | `environment/intersection.multi_objective_reward` |
| Convergence episode | smoothed reward reaches 90 % of its final value and stays ≥ 200 episodes | `evaluation/metrics.convergence_episode`; when the final smoothed reward is ≤ 0 (e.g. Chapter 5) the criterion becomes 90 % of the progress from the initial to the final value |
| Statistics | Welch t-test on the per-seed means of the last 500 episodes, Holm-Bonferroni | `evaluation/compare.py` (the seed is the statistical unit) |

### Measurements (incoming lanes of the controlled traffic light, TraCI lane subscriptions)
* `q` - number of halting vehicles (speed < 0.1 m/s); per approach for the Chapter 6 state, the
  approach (N, E, S, W) being given by the direction of the last segment of each incoming lane.
* `w` - mean waiting time of the vehicles present (sum of SUMO lane waiting times / number of
  vehicles). Episode indicator: halting vehicle-seconds divided by the number of distinct vehicles
  that used the intersection, i.e. the mean over vehicles of W_i (Section 6.2.5).
* `v` - vehicle-weighted mean speed; free-flow speed of the lanes when no vehicle is present.
* `e` - SUMO CO2 emission (mg/s), summed over the incoming lanes.
* `f` - SUMO fuel consumption, summed over the incoming lanes. **Unit:** recent SUMO versions
  (≥ 1.14) report fuel in **mg/s** (earlier versions: ml/s). The reward coefficient 10⁻³ is
  applied to the raw SUMO value, as for CO2; the thesis tables label fuel in mL/s.
* Throughput - vehicles arrived in the whole network, per hour.

### Signal control
The green phases are those of the **native SUMO program** of the traffic light (computed by
netconvert, hence free of conflicting priority movements), in the order of the plan; a green whose
links are a strict subset of another one (e.g. a short protected-left phase) is not kept separately.
Each green is labelled North-South or East-West by the majority of the approaches it serves (the
approach being given by the direction of the last segment of each incoming lane). Only if the
native program has fewer than two greens is one green per axis built from the geometry. A phase
change goes through 3 s of yellow (links that stay green in the next phase stay green) and 2 s of
all-red.
* Chapter 6 phase index φ ∈ {0, 1, 2, 3}: index of the active green (0, 1, 2) or 3 during a
  transition. Actions: keep / green NS / green EW / next phase of the plan; "green NS" selects the
  current green if it serves NS, otherwise the next NS green of the plan. A change is refused before
  10 s of green; after 60 s of green the next phase of the plan is forced. Route de Sefrou: three
  greens (Route de Sefrou + Bd Moulay Rachid [NS], the two other arms [EW], branch 288920323#10 [NS]).
* Chapter 7: `p` = axis of the current (or target) green, 0 = horizontal (EW), 1 = vertical (NS);
  `d` = time since the onset of the current green (0 during a transition). No maximum green. An
  intersection with several greens on the same axis enters them in the order of the plan.

## Chapter 5 (isolated intersection)
Configuration of this preliminary study (not detailed in Section 5.6): a synthetic four-arm intersection (2 lanes per arm, 250 m, 50 km/h),
Bernoulli insertions on the four approaches (≈ 360 veh/h in total, 60 % straight / 20 % left /
20 % right), episodes of 3600 s, the same discretised state as in Chapter 6, and the reward
`−(sum of the waiting times of the vehicles on the incoming lanes)`. The scheduling functions of
the dynamic weights (SARSA dominant at the beginning, Q-learning increasing, Monte Carlo
increasing towards the end) are
`f_Q(ξ) = 0.2 + 0.6 ξ`, `f_S(ξ) = 0.7 − 0.6 ξ`, `f_MC(ξ) = 0.1 + 0.3 ξ²`, renormalised.

## Chapter 6 (Route de Sefrou)
As in Table 6.1. The studied traffic light is `TLS_ID`, or the one closest to `TLS_COORD`, or the
traffic light with the most controlled links (the network of `map/sefrou/` has a single one).
The demand file of `map/sefrou/` was produced by the OSM Web Wizard (randomTrips, insertion density
12, fringe factor 5, minimum distance 300 m): 630 trips in 3600 s. Demand: `scripts/generate_demand.py --scenario sefrou`
(`--insertion-rate 3600`); with `--jtr`, `jtrrouter` imposes 60 / 20 / 20 turning ratios at every
junction. The reference is the native SUMO program (`ALGORITHM = "fixed"`).

## Chapter 7 (Saiss network)
* Minimum green g_min = 10 s (`MIN_GREEN`), as in Chapter 6.
* State discretisation of the normalised variables: 5 uniform bins
  for q, w, v and for each neighbour summary, p binary, d binary (d < g_min, i.e. switch not
  admissible, or d ≥ g_min) → 5³ × 2 × 2 × 5² = 12 500 states per agent (500 without neighbours).
* q and w are min-max scaled on [0, 20 veh] and [0, 120 s] (Chapter 6 ranges), v divided by the
  free-flow speed.
* SARSA uses the next **executed** action ã_{t+1}; Q-learning and SARSA are updated once ã_{t+1}
  is known (standard SARSA order, as in Algorithm 6), before the next joint action is applied.
* The Monte Carlo table uses the same last-visit convention as Chapter 6.
* The fixed-time values used in the thesis (10.47 m/s, 5000 mg/s, 2000) are indicative values
  taken from Chapter 6 (Section 7.5.4). `evaluation/compare.py` uses a measured fixed-time run
  (`ALGORITHM = "fixed"`) when it exists, otherwise these indicative values.
* Weight sensitivity: in the three-table architecture, the weights (1, 0, 0), (0, 1, 0), (0, 0, 1)
  make the decision depend on one table only, i.e. Q-learning, SARSA or Monte Carlo alone.

### DQN and PPO (Section 7.6.10)
Independent agents per intersection, same continuous normalised state (7 features), same actions,
reward and safety constraint; NumPy implementation (no GPU), usual default hyper-parameters:
* DQN: MLP 64-64 ReLU, Adam 1e-4, replay 50 000 (smaller than the 10⁶ of common libraries, to fit
  in memory for eight agents), batch 32, learning starts 1000, update every 4 steps, target network
  copied every 1000 steps, Huber loss, γ = 0.95, same ε schedule as the tabular methods.
* PPO: separate actor and critic MLP 64-64 tanh, Adam 3e-4, 2048 steps per update, 10 epochs,
  minibatch 64, clip 0.2, GAE λ = 0.95, γ = 0.95, value coefficient 0.5, gradient clipping 0.5;
  the log-probability stored is that of the executed action.
* The replay buffer is not stored in the checkpoints (a resumed DQN run refills it).

## Chapter 4 (track)
Environment of Section 4.5.1: a rectangular ring
track (600 × 400 px, width 110 px); the car has a heading and a speed (accelerate/brake ±2 px/step,
max 12, turn ±15°) but the state is only its position, discretised in 20 px cells; −1 per step,
−10 for a crash, after which the car is put back at the start; 100 steps per episode; ε decay
0.9949 (Q-learning, hybrid), 0.9969 (SARSA), 0.9966 (Monte Carlo), no floor. Hybrid V1 (Eq. 4.1):
three tables updated by their own rules, and the decision table updated as
`Q ← (1 − α) Q + α (w_Q Q_Q + w_S Q_S + w_MC Q_MC)` for the visited pairs.
