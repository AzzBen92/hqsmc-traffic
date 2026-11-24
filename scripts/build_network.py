"""Build a SUMO network from OpenStreetMap (Sections 6.1 and 7.5.1).

    # 1) download the OSM extract (needs internet), bbox = west,south,east,north (WGS84)
    python scripts/build_network.py --name sefrou --bbox <W>,<S>,<E>,<N>
    # 2) or convert an .osm file exported from openstreetmap.org
    python scripts/build_network.py --name saiss --osm path/to/map.osm

Writes map/<name>/osm.net.xml.gz (roads + traffic lights) and map/<name>/osm.poly.xml.gz
(buildings, for display only), with the options of SUMO's OSM Web Wizard.
Check the result in netedit/sumo-gui, then generate the demand with generate_demand.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.sumo_utils import find_sumo_home  # noqa: E402

NETCONVERT_OPTIONS = [
    "--geometry.remove", "true",
    "--roundabouts.guess", "true",
    "--ramps.guess", "true",
    "--junctions.join", "true",
    "--tls.guess-signals", "true",
    "--tls.discard-simple", "true",
    "--tls.join", "true",
    "--tls.default-type", "static",
    "--edges.join", "true",
    "--keep-edges.by-vclass", "passenger",
    "--remove-edges.isolated", "true",
    "--output.street-names", "true",
    "--output.original-names", "true",
    "--no-turnarounds.except-deadend", "true",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="folder under map/ (sefrou, saiss, ...)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--osm", help="existing .osm file")
    g.add_argument("--bbox", help="west,south,east,north (WGS84) to download with SUMO's osmGet.py")
    args = ap.parse_args()

    home = find_sumo_home()
    out = ROOT / "map" / args.name
    out.mkdir(parents=True, exist_ok=True)
    osm = Path(args.osm) if args.osm else out / "osm_bbox.osm.xml.gz"
    if args.bbox:
        subprocess.run([sys.executable, str(home / "tools" / "osmGet.py"), "--bbox", args.bbox,
                        "--prefix", "osm", "--output-dir", str(out), "--gzip"], check=True)
    typemap = home / "data" / "typemap" / "osmNetconvert.typ.xml"
    cmd = [str(home / "bin" / "netconvert"), "--osm-files", str(osm), "-o", str(out / "osm.net.xml.gz")]
    if typemap.exists():
        cmd += ["--type-files", str(typemap)]
    subprocess.run(cmd + NETCONVERT_OPTIONS, check=True)
    poly_types = home / "data" / "typemap" / "osmPolyconvert.typ.xml"
    pcmd = [str(home / "bin" / "polyconvert"), "--osm-files", str(osm), "--net-file", str(out / "osm.net.xml.gz"),
            "-o", str(out / "osm.poly.xml.gz"), "--ignore-errors", "true"]
    if poly_types.exists():
        pcmd += ["--type-file", str(poly_types)]
    subprocess.run(pcmd, check=False)
    print(f"Network written to {out / 'osm.net.xml.gz'}")


if __name__ == "__main__":
    main()
