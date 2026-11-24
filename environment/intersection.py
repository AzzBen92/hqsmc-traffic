"""Traffic-light control and traffic measurements for one signalised intersection.

* `build_green_states`  takes the green phases of the native SUMO program (conflict-free)
  and labels each of them North-South or East-West from the network geometry, so that the
  same code works for any intersection extracted from OpenStreetMap.
* `SignalController`     applies green requests with minimum / maximum green times and
  the yellow (3 s) + all-red (2 s) transition (Table 6.1).
* `IntersectionMonitor`  measures queues, waiting time, speed, CO2 and fuel on the
  incoming lanes through TraCI subscriptions (one call per lane per step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import traci.constants as tc  # constants are shared by traci and libsumo

COMPASS_ORDER = ("N", "E", "S", "W")


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def lane_origin(traci, lane_id: str) -> str:
    """Compass side the lane comes FROM (N, E, S or W), from its last shape segment."""
    shape = traci.lane.getShape(lane_id)
    (x1, y1), (x2, y2) = shape[-2], shape[-1]
    dx, dy = x2 - x1, y2 - y1
    if abs(dy) >= abs(dx):
        return "S" if dy > 0 else "N"      # travelling north -> comes from the south
    return "W" if dx > 0 else "E"          # travelling east  -> comes from the west


def lane_axis(traci, lane_id: str) -> str:
    return "NS" if lane_origin(traci, lane_id) in ("N", "S") else "EW"


def link_direction(traci, in_lane: str, out_lane: str) -> str:
    """SUMO direction of a connection: s (straight), l, r, L, R, t (turn-around)."""
    for link in traci.lane.getLinks(in_lane):   # extended tuples (default in traci and libsumo)
        if link[0] == out_lane:
            return link[6]
    return "s"


def tls_position(traci, tls_id: str) -> tuple[float, float]:
    """Position of a traffic light: its junction if it exists, else the mean end of its incoming lanes."""
    try:
        return traci.junction.getPosition(tls_id)
    except Exception:  # noqa: BLE001 - joined clusters have no junction with the TLS id
        pts = [traci.lane.getShape(l)[-1] for l in incoming_lanes(traci, tls_id)]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def incoming_lanes(traci, tls_id: str) -> list[str]:
    lanes = []
    for links in traci.trafficlight.getControlledLinks(tls_id):
        for in_lane, _out, _via in links:
            if not in_lane.startswith(":") and in_lane not in lanes:
                lanes.append(in_lane)
    return lanes


def nearest_tls(traci, xy: tuple[float, float], exclude=()) -> str:
    best, best_d = None, math.inf
    for tls in traci.trafficlight.getIDList():
        if tls in exclude:
            continue
        x, y = tls_position(traci, tls)
        d = math.hypot(x - xy[0], y - xy[1])
        if d < best_d:
            best, best_d = tls, d
    if best is None:
        raise RuntimeError("No traffic light found in the network.")
    return best


def main_tls(traci) -> str:
    """Traffic light with the largest number of controlled links (used when TLS_ID is not set)."""
    ids = traci.trafficlight.getIDList()
    if not ids:
        raise RuntimeError("The network contains no traffic light.")
    return max(ids, key=lambda t: sum(len(l) for l in traci.trafficlight.getControlledLinks(t)))


def _green_links(state: str) -> frozenset:
    return frozenset(i for i, c in enumerate(state) if c in "Gg")


def _geometric_greens(traci, tls_id: str, links, n: int) -> list[str]:
    """Fallback: one green per axis built from the geometry ('G' straight/right, 'g' left/U-turn)."""
    states = {ax: ["r"] * n for ax in ("NS", "EW")}
    for idx in range(min(n, len(links))):
        if not links[idx]:
            continue
        in_lane, out_lane, _via = links[idx][0]
        direction = link_direction(traci, in_lane, out_lane)
        states[lane_axis(traci, in_lane)][idx] = "g" if direction in ("l", "L", "t") else "G"
    return ["".join(v) for v in states.values() if any(c != "r" for c in v)]


def green_axis(traci, state: str, links) -> str:
    """Axis (NS / EW) of a green state: majority of the approaches of its green links."""
    votes = {"NS": 0, "EW": 0}
    for i in _green_links(state):
        if i < len(links) and links[i]:
            votes[lane_axis(traci, links[i][0][0])] += 1
    return "NS" if votes["NS"] >= votes["EW"] else "EW"


def build_green_states(traci, tls_id: str) -> tuple[list[str], list[str]]:
    """Green phases of a traffic light and the axis (NS / EW) of each of them.

    The green phases are those of the native SUMO program (computed by netconvert, so free of
    conflicting priority movements), in the order of the program; a phase whose green links are
    a subset of another one (e.g. a short protected-left phase) is not kept as a separate green.
    If the native program has fewer than two greens, one green per axis is built from the
    geometry instead. Returns (greens, axes).
    """
    links = traci.trafficlight.getControlledLinks(tls_id)
    n = len(traci.trafficlight.getRedYellowGreenState(tls_id))
    native: list[str] = []
    for logic in traci.trafficlight.getAllProgramLogics(tls_id)[:1]:
        for ph in logic.phases:
            st = ph.state
            if "y" in st or not _green_links(st) or st in native:
                continue
            native.append(st)
    sets = [_green_links(st) for st in native]
    greens = [st for st, g in zip(native, sets)
              if not any(g < other for other in sets)]          # drop strict subsets
    if len(greens) < 2:
        greens = _geometric_greens(traci, tls_id, links, n) or greens
    if len(greens) == 1:
        greens = greens * 2
    return greens, [green_axis(traci, st, links) for st in greens]


# -----------------------------------------------------------------------------
# Signal controller
# -----------------------------------------------------------------------------
class SignalController:
    """Green requests -> SUMO signal states, with min/max green and yellow + all-red.

    `greens` are the green phases in the order of the plan, `axes` their axis (NS / EW).
    """

    def __init__(self, traci, tls_id: str, greens: list[str], axes: list[str], min_green: float,
                 max_green: float | None, yellow: float, all_red: float, step_length: float = 1.0):
        self.traci, self.tls_id, self.greens, self.axes = traci, tls_id, greens, axes
        self.min_green, self.max_green = min_green, max_green
        self.yellow_time, self.all_red_time, self.dt = yellow, all_red, step_length
        self.n_greens = len(greens)
        self.current = 0          # active green, or target green during a transition
        self.kind = "green"       # "green" | "yellow" | "allred"
        self.timer = 0.0
        self.elapsed = 0.0        # time since the onset of the current green
        self.switches = 0
        self._previous = 0
        self._set(self.greens[0])

    # --- state seen by the agents -------------------------------------------------
    @property
    def in_transition(self) -> bool:
        return self.kind != "green"

    @property
    def phase_index(self) -> int:
        """phi in {0, 1, 2, 3} (Chapter 6): index of the active green (0, 1, 2; capped at 2),
        3 during a yellow / all-red transition."""
        return 3 if self.kind != "green" else min(self.current, 2)

    @property
    def axis(self) -> str:
        """Axis of the active (or target) green."""
        return self.axes[self.current]

    @property
    def can_switch(self) -> bool:
        return self.kind == "green" and self.elapsed >= self.min_green

    def green_for_axis(self, axis: str) -> int:
        """Current green if it serves `axis`, otherwise the next green of the plan serving it."""
        if self.axes[self.current] == axis:
            return self.current
        for step in range(1, self.n_greens + 1):
            k = (self.current + step) % self.n_greens
            if self.axes[k] == axis:
                return k
        return self.current          # no green serves this axis

    # --- actions --------------------------------------------------------------------
    def request(self, k: int) -> bool:
        """Request green k. Returns True if a phase change actually starts."""
        if self.kind != "green" or k == self.current or self.elapsed < self.min_green:
            return False
        self._start_transition(k)
        return True

    def request_axis(self, axis: str) -> bool:
        return self.request(self.green_for_axis(axis))

    def request_next(self) -> bool:
        return self.request((self.current + 1) % self.n_greens)

    def tick(self) -> None:
        """Advance the internal clocks by one step (call AFTER traci.simulationStep())."""
        if self.kind == "green":
            self.elapsed += self.dt
            if self.max_green is not None and self.elapsed >= self.max_green:
                self._start_transition((self.current + 1) % self.n_greens)
        elif self.kind == "yellow":
            self.timer -= self.dt
            if self.timer <= 1e-9:
                if self.all_red_time > 0:
                    self.kind, self.timer = "allred", self.all_red_time
                    self._set("r" * len(self.greens[0]))
                else:
                    self._green_onset()
        else:  # all red
            self.timer -= self.dt
            if self.timer <= 1e-9:
                self._green_onset()

    # --- internals ------------------------------------------------------------------
    def _start_transition(self, k: int) -> None:
        if self.n_greens < 2 or self.greens[k] == self.greens[self.current]:
            return
        self._previous, self.current = self.current, k
        self.switches += 1
        nxt = self.greens[k]
        if self.yellow_time > 0:
            # links green now and still green in the next phase stay green; the others turn yellow
            prev = self.greens[self._previous]
            yellow = "".join(c if (c in "Gg" and nxt[i] in "Gg") else ("y" if c in "Gg" else c)
                             for i, c in enumerate(prev))
            self.kind, self.timer = "yellow", self.yellow_time
            self._set(yellow)
        elif self.all_red_time > 0:
            self.kind, self.timer = "allred", self.all_red_time
            self._set("r" * len(self.greens[0]))
        else:
            self._green_onset()

    def _green_onset(self) -> None:
        self.kind, self.timer, self.elapsed = "green", 0.0, 0.0
        self._set(self.greens[self.current])

    def _set(self, state: str) -> None:
        self.traci.trafficlight.setRedYellowGreenState(self.tls_id, state)


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------
LANE_VARS = (
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    tc.LAST_STEP_MEAN_SPEED,
    tc.VAR_WAITING_TIME,
    tc.VAR_CO2EMISSION,
    tc.VAR_FUELCONSUMPTION,
    tc.LAST_STEP_VEHICLE_ID_LIST,
)


@dataclass
class EpisodeAccumulator:
    steps: int = 0
    queue: float = 0.0
    wait_state: float = 0.0
    speed_weighted: float = 0.0
    vehicle_steps: float = 0.0
    co2: float = 0.0
    fuel: float = 0.0
    halting_seconds: float = 0.0
    vehicles: set = field(default_factory=set)


class IntersectionMonitor:
    """Traffic indicators on the incoming lanes of one traffic light."""

    def __init__(self, traci, tls_id: str, step_length: float = 1.0):
        self.traci, self.tls_id, self.dt = traci, tls_id, step_length
        self.lanes = incoming_lanes(traci, tls_id)
        self.approach_of = {l: lane_origin(traci, l) for l in self.lanes}
        self.free_speed = (sum(traci.lane.getMaxSpeed(l) for l in self.lanes) / len(self.lanes)
                           if self.lanes else 13.89)
        for lane in self.lanes:
            traci.lane.subscribe(lane, LANE_VARS)
        self.acc = EpisodeAccumulator()
        self.last: dict = {}

    def measure(self) -> dict:
        """Indicators of the current step; also updates the episode accumulators."""
        queues = {c: 0.0 for c in COMPASS_ORDER}
        n_tot = halt_tot = wait_tot = speed_sum = co2 = fuel = 0.0
        ids: list[str] = []
        for lane in self.lanes:
            r = self.traci.lane.getSubscriptionResults(lane)
            if not r:
                continue
            n = r[tc.LAST_STEP_VEHICLE_NUMBER]
            halt = r[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
            queues[self.approach_of[lane]] += halt
            n_tot += n
            halt_tot += halt
            wait_tot += r[tc.VAR_WAITING_TIME]
            if n > 0:
                speed_sum += max(r[tc.LAST_STEP_MEAN_SPEED], 0.0) * n
            co2 += r[tc.VAR_CO2EMISSION]
            fuel += r[tc.VAR_FUELCONSUMPTION]
            ids.extend(r[tc.LAST_STEP_VEHICLE_ID_LIST])
        speed = speed_sum / n_tot if n_tot > 0 else self.free_speed
        m = {
            "queues": [queues[c] for c in COMPASS_ORDER],   # q^(l), l = N, E, S, W
            "queue": halt_tot,                               # total queue length (veh)
            "wait": wait_tot / n_tot if n_tot > 0 else 0.0,  # mean waiting time (s)
            "wait_total": wait_tot,                          # cumulative waiting time (s)
            "speed": speed,                                  # mean speed (m/s)
            "co2": co2,                                      # mg/s
            "fuel": fuel,                                    # SUMO fuel output (see notes)
            "n_veh": n_tot,
        }
        a = self.acc
        a.steps += 1
        a.queue += halt_tot
        a.wait_state += m["wait"]
        a.speed_weighted += speed_sum
        a.vehicle_steps += n_tot
        a.co2 += co2
        a.fuel += fuel
        a.halting_seconds += halt_tot * self.dt
        a.vehicles.update(ids)
        self.last = m
        return m

    def episode_summary(self) -> dict:
        """Episode indicators (Section 6.2.5): mean queue, per-vehicle waiting, mean speed, CO2, fuel."""
        a = self.acc
        steps = max(a.steps, 1)
        n_vehicles = len(a.vehicles)
        return {
            "queue": a.queue / steps,
            # mean over vehicles of the waiting time accumulated at this intersection
            "wait": a.halting_seconds / n_vehicles if n_vehicles else 0.0,
            "speed": a.speed_weighted / a.vehicle_steps if a.vehicle_steps else self.free_speed,
            "co2": a.co2 / steps,
            "fuel": a.fuel / steps,
            "vehicles": n_vehicles,
        }


def multi_objective_reward(m: dict, coefs: dict) -> float:
    """r = -2.0 q - 0.05 w + 1.5 v - 1e-3 CO2 - 1e-3 fuel (Eq. 6.2 / 7.6)."""
    return (coefs["queue"] * m["queue"] + coefs["wait"] * m["wait"] + coefs["speed"] * m["speed"]
            + coefs["co2"] * m["co2"] + coefs["fuel"] * m["fuel"])


def bin_index(x: float, n_bins: int, x_max: float) -> int:
    """Uniform bins on [0, x_max]; out-of-range values go to the nearest bin (Section 6.2.2)."""
    if x_max <= 0:
        return 0
    return int(min(max(math.floor(n_bins * x / x_max), 0), n_bins - 1))
