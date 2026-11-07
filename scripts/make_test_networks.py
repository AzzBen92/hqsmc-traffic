"""Build small synthetic SUMO networks to test the whole pipeline without the Fes OSM data.

    python scripts/make_test_networks.py

Creates
    map/isolated/      isolated four-arm intersection with stochastic arrivals (Chapter 5)
    map/demo_single/   the same four-arm intersection with randomTrips demand (pipeline of Chapter 6)
    map/demo_grid/     4 x 2 grid of eight signalised intersections (pipeline of Chapter 7)

The isolated intersection IS the Chapter 5 set-up. The two demo networks are only meant to check
that the code runs: the thesis results were obtained on the real Route de Sefrou and Saiss networks
(map/sefrou/ and map/saiss/, see map/README.md).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.sumo_utils import find_sumo_home  # noqa: E402

SUMO_HOME = find_sumo_home()
BIN = SUMO_HOME / "bin"
TOOLS = SUMO_HOME / "tools"


def run(cmd, cwd=None):
    print(" ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def sumocfg(path: Path, net: str, routes: str, additional: str | None = None, end: int = 3600):
    add = f'\n        <additional-files value="{additional}"/>' if additional else ""
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net}"/>
        <route-files value="{routes}"/>{add}
    </input>
    <time>
        <begin value="0"/>
        <end value="{end}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
""")


def four_arm(folder: Path, name: str, arm: float = 250.0, lanes: int = 2, speed: float = 13.89):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.nod.xml").write_text(f"""<nodes>
    <node id="C" x="0" y="0" type="traffic_light"/>
    <node id="N" x="0" y="{arm}" type="priority"/>
    <node id="S" x="0" y="-{arm}" type="priority"/>
    <node id="E" x="{arm}" y="0" type="priority"/>
    <node id="W" x="-{arm}" y="0" type="priority"/>
</nodes>
""")
    edges = []
    for o in "NSEW":
        edges.append(f'    <edge id="{o}2C" from="{o}" to="C" numLanes="{lanes}" speed="{speed}"/>')
        edges.append(f'    <edge id="C2{o}" from="C" to="{o}" numLanes="{lanes}" speed="{speed}"/>')
    (folder / f"{name}.edg.xml").write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n")
    # relative paths, so that the generated network does not record machine-specific paths
    run([BIN / "netconvert", "-n", f"{name}.nod.xml", "-e", f"{name}.edg.xml",
         "-o", f"{name}.net.xml", "--no-turnarounds", "true", "--tls.default-type", "static"], cwd=folder)


def isolated():
    """Chapter 5: horizontal and vertical stochastic arrivals (Bernoulli insertion every second)."""
    folder = ROOT / "map" / "isolated"
    four_arm(folder, "isolated")
    flows = []
    # straight 60 %, left 20 %, right 20 % from each approach; total about 360 veh/h
    turns = {"N": ("S", "E", "W"), "S": ("N", "W", "E"), "E": ("W", "S", "N"), "W": ("E", "N", "S")}
    p_total = 0.025
    for o, (straight, left, right) in turns.items():
        for dest, share in ((straight, 0.6), (left, 0.2), (right, 0.2)):
            flows.append(f'    <flow id="{o}_{dest}" begin="0" end="3600" probability="{p_total * share:.4f}" '
                         f'from="{o}2C" to="C2{dest}" departLane="best" departSpeed="max"/>')
    (folder / "isolated.rou.xml").write_text("<routes>\n" + "\n".join(flows) + "\n</routes>\n")
    sumocfg(folder / "isolated.sumocfg", "isolated.net.xml", "isolated.rou.xml")


def demo_single():
    folder = ROOT / "map" / "demo_single"
    four_arm(folder, "osm")
    run([sys.executable, TOOLS / "randomTrips.py", "-n", folder / "osm.net.xml",
         "-o", folder / "osm.passenger.trips.xml", "-r", folder / "osm.passenger.rou.xml",
         "-b", "0", "-e", "3600", "--insertion-rate", "1800", "--seed", "42", "--fringe-factor", "10",
         "--validate", "--remove-loops"])
    sumocfg(folder / "osm.sumocfg", "osm.net.xml", "osm.passenger.trips.xml")


def demo_grid():
    folder = ROOT / "map" / "demo_grid"
    folder.mkdir(parents=True, exist_ok=True)
    run([BIN / "netgenerate", "--grid", "--grid.x-number", "4", "--grid.y-number", "2",
         "--grid.length", "300", "--grid.attach-length", "200", "--default.lanenumber", "2",
         "--default.speed", "13.89", "--default-junction-type", "traffic_light",
         "--tls.default-type", "static", "--no-turnarounds", "true",
         "-o", folder / "osm.net.xml"])
    for level, period in (("medium", 0.8347), ("low", 1.2), ("high", 0.58)):
        trips = f"osm.passenger.trips{'' if level == 'medium' else '_' + level}.xml"
        run([sys.executable, TOOLS / "randomTrips.py", "-n", folder / "osm.net.xml",
             "-o", folder / trips, "-r", folder / trips.replace("trips", "rou"),
             "-b", "0", "-e", "3600", "-p", str(period),
             "--fringe-factor", "5", "--min-distance", "300", "--seed", "42",
             "--validate", "--remove-loops"])
        cfg_name = "osm.sumocfg" if level == "medium" else f"osm_{level}.sumocfg"
        sumocfg(folder / cfg_name, "osm.net.xml", trips)


if __name__ == "__main__":
    isolated()
    demo_single()
    demo_grid()
    print("Test networks written under map/isolated, map/demo_single and map/demo_grid.")
