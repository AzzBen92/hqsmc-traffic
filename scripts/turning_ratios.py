"""Turning movements realised at the eight Saiss intersections (Table 7.6).

    python scripts/turning_ratios.py --routes map/saiss/osm.passenger.rou.xml

For every route, each pair of consecutive edges that meets at one of the studied junctions is
matched against the connections of that junction (straight / left / right / U-turn).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sumolib  # noqa: E402

from config import load_config  # noqa: E402

DIR = {"s": "straight", "l": "left", "L": "left", "r": "right", "R": "right", "t": "uturn"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default=str(ROOT / "map" / "saiss" / "osm.net.xml.gz"))
    ap.add_argument("--routes", default=str(ROOT / "map" / "saiss" / "osm.passenger.rou.xml"))
    args = ap.parse_args()
    cfg = load_config(SCENARIO="saiss")
    net = sumolib.net.readNet(args.net, withPrograms=True)
    # traffic light of each agent: given ids, or the one closest to the coordinates of Table 7.3
    tls_nodes = {}
    for tls in net.getTrafficLights():
        nodes = {c[0].getEdge().getToNode() for c in tls.getConnections()}
        tls_nodes[tls.getID()] = nodes
    def position(tls_id):
        xs = [n.getCoord() for n in tls_nodes[tls_id]]
        return sum(x for x, _ in xs) / len(xs), sum(y for _, y in xs) / len(xs)
    mapping = dict(cfg.TLS_IDS) if cfg.TLS_IDS else {}
    used = set(mapping.values())
    for i, xy in cfg.TLS_COORDS.items():
        if i in mapping:
            continue
        best = min((t for t in tls_nodes if t not in used),
                   key=lambda t: (position(t)[0] - xy[0]) ** 2 + (position(t)[1] - xy[1]) ** 2)
        mapping[i] = best
        used.add(best)
    node_to_agent = {n.getID(): i for i, t in mapping.items() for n in tls_nodes[t]}

    counts = defaultdict(Counter)
    vehicles = defaultdict(set)
    total, crossing = 0, set()
    for veh in sumolib.xml.parse(args.routes, "vehicle"):
        total += 1
        route = veh.route[0].edges.split() if veh.route else []
        for e1, e2 in zip(route[:-1], route[1:]):
            edge = net.getEdge(e1)
            agent = node_to_agent.get(edge.getToNode().getID())
            if agent is None:
                continue
            conns = edge.getOutgoing().get(net.getEdge(e2), [])
            if not conns:
                continue
            counts[agent][DIR.get(conns[0].getDirection(), "straight")] += 1
            vehicles[agent].add(veh.id)
            crossing.add(veh.id)
    print(f"{total} trips, {len(crossing)} ({100 * len(crossing) / max(total, 1):.1f} %) cross at least one studied intersection")
    print(f"{'Int.':<5}{'N':>6}{'straight %':>12}{'left %':>9}{'right %':>9}{'U-turn %':>10}")
    agg = Counter()
    for i in sorted(counts):
        c = counts[i]
        n = sum(c.values())
        agg.update(c)
        print(f"I{i:<4}{len(vehicles[i]):>6}" + "".join(f"{100 * c[k] / n:>{w}.1f}" for k, w in
              (("straight", 12), ("left", 9), ("right", 9), ("uturn", 10))))
    n = sum(agg.values())
    print(f"{'All':<5}{n:>6}" + "".join(f"{100 * agg[k] / n:>{w}.1f}" for k, w in
          (("straight", 12), ("left", 9), ("right", 9), ("uturn", 10))))


if __name__ == "__main__":
    main()
