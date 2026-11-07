# SUMO files

Each study has its own folder. A simulation is entirely defined by the files of its folder:

| File | Content |
|---|---|
| `osm.net.xml.gz` | road network and traffic lights (netconvert, from OpenStreetMap) |
| `osm.passenger.trips.xml` / `osm.passenger.rou.xml` | demand (randomTrips.py, duarouter) |
| `osm.sumocfg` | configuration linking network, demand and outputs |
| `osm.poly.xml.gz`, `output.add.xml` | optional additional files (buildings, detectors, outputs) |

## Folders

* `sefrou/` - Route de Sefrou - Boulevard Moulay Rachid intersection, Fes (Chapter 6). Included:
  network, demand (630 trips, OSM Web Wizard), buildings, the OSM extract (`osm_bbox.osm.xml.gz`,
  bounds 34.0032-34.0203 N, 4.9930-4.9705 W) and `osm.sumocfg`.
* `saiss/` - Saiss district, eight signalised intersections (Chapter 7). Three demand levels:
  `osm.sumocfg` (medium, 4313 trips), `osm_low.sumocfg` (~3000), `osm_high.sumocfg` (~6200).
* `isolated/` - synthetic isolated intersection (Chapter 5), created by
  `python scripts/make_test_networks.py` (with the `demo_single/` and `demo_grid/` test networks).

## Using your own files

Copy the network and demand files used for the experiments into `sefrou/` and `saiss/` with the
names above (the `.sumocfg` must reference them with relative paths). To rebuild them from
OpenStreetMap:

```bash
python scripts/build_network.py --name sefrou --osm path/to/sefrou.osm
python scripts/generate_demand.py --scenario sefrou            # add --jtr for 60/20/20 turns
python scripts/build_network.py --name saiss --osm path/to/saiss.osm
python scripts/generate_demand.py --scenario saiss --demand medium
python scripts/generate_demand.py --scenario saiss --demand low
python scripts/generate_demand.py --scenario saiss --demand high
python scripts/turning_ratios.py                               # Table 7.6
```

Open the network once in `sumo-gui` or `netedit` to check the traffic lights; in `config.py`,
`TLS_ID` (Sefrou) and `TLS_IDS` (Saiss) can fix the SUMO ids of the controlled traffic lights.
