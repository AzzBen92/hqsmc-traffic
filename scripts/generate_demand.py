"""Traffic demand with SUMO's randomTrips.py and the matching osm.sumocfg (Sections 6.2.1 and 7.5.3).

    python scripts/generate_demand.py --scenario sefrou                  # 3600 veh/h (--insertion-rate)
    python scripts/generate_demand.py --scenario saiss --demand medium   # 4313 trips / 3600 s
    python scripts/generate_demand.py --scenario saiss --demand low      # ~3000 trips (70 %)
    python scripts/generate_demand.py --scenario saiss --demand high     # ~6200 trips (144 %)

Saiss: origins/destinations drawn among fringe edges (fringe factor 5), minimum trip distance
300 m, departures evenly spaced (period = 3600 / number of trips), routes by duarouter (fastest
path). Sefrou: insertion rate 3600 veh/h over the whole network; with --jtr the turning
movements at every junction follow 60 % straight / 20 % left / 20 % right (jtrrouter).
Only passenger cars (default SUMO vehicle type) are generated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.sumo_utils import find_sumo_home  # noqa: E402

SAISS_PERIOD = {"low": 3600 / 3000, "medium": 3600 / 4313, "high": 3600 / 6200}


def write_sumocfg(folder: Path, cfg_name: str, net: str, routes: str):
    extra = [f for f in ("osm.poly.xml.gz", "output.add.xml") if (folder / f).exists()]
    add = f'\n        <additional-files value="{",".join(extra)}"/>' if extra else ""
    (folder / cfg_name).write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net}"/>
        <route-files value="{routes}"/>{add}
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
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
    print("wrote", folder / cfg_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["sefrou", "saiss"])
    ap.add_argument("--demand", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=42, help="seed of randomTrips (demand realisation)")
    ap.add_argument("--net", default="osm.net.xml.gz")
    ap.add_argument("--jtr", action="store_true", help="sefrou: impose 60/20/20 turning ratios with jtrrouter")
    args = ap.parse_args()

    home = find_sumo_home()
    tools = home / "tools"
    folder = ROOT / "map" / args.scenario
    net = folder / args.net
    if not net.exists():
        raise SystemExit(f"{net} not found: build the network first (scripts/build_network.py).")
    suffix = "" if args.scenario == "sefrou" or args.demand == "medium" else f"_{args.demand}"
    trips = f"osm.passenger.trips{suffix}.xml"
    routes = f"osm.passenger{suffix}.rou.xml"
    common = [sys.executable, str(tools / "randomTrips.py"), "-n", str(net), "-b", "0", "-e", "3600",
              "--seed", str(args.seed), "--validate", "--remove-loops", "--vehicle-class", "passenger",
              "--prefix", "veh"]
    if args.scenario == "saiss":
        cmd = common + ["-o", str(folder / trips), "-r", str(folder / routes),
                        "-p", f"{SAISS_PERIOD[args.demand]:.4f}", "--fringe-factor", "5",
                        "--min-distance", "300"]
        subprocess.run(cmd, check=True)
        cfg_name = "osm.sumocfg" if args.demand == "medium" else f"osm_{args.demand}.sumocfg"
        write_sumocfg(folder, cfg_name, args.net, routes)
        return

    if not args.jtr:
        cmd = common + ["-o", str(folder / trips), "-r", str(folder / routes), "--insertion-rate", "3600"]
        subprocess.run(cmd, check=True)
        write_sumocfg(folder, "osm.sumocfg", args.net, trips)
        return
    # jtrrouter: flows entering from the fringe, turning ratios right / straight / left = 20 / 60 / 20
    jtr_common = [c for c in common if c != "--validate"]
    subprocess.run(jtr_common + ["-o", str(folder / "osm.fringe.flows.xml"), "--insertion-rate", "3600",
                                 "--fringe-factor", "max", "--jtrrouter", "--flows", "40"], check=True)
    subprocess.run([str(home / "bin" / "jtrrouter"), "-n", str(net), "--route-files",
                    str(folder / "osm.fringe.flows.xml"), "--turn-defaults", "20,60,20",
                    "--accept-all-destinations", "true", "--remove-loops", "true",
                    "-o", str(folder / "osm.passenger.rou.xml"), "--seed", str(args.seed)], check=True)
    write_sumocfg(folder, "osm.sumocfg", args.net, "osm.passenger.rou.xml")


if __name__ == "__main__":
    main()
