# Transit RL – Kilkenny

    pip install numpy scipy networkx gymnasium
    python demo.py kilkenny_pois.geojson kilkenny_roads.graphml   # heuristic network -> viewer_data.json
    python build_viewer.py                                        # -> viewer_standalone.html (or open viewer.html and load the JSON)

## Use in RL
    from transit_rl import *
    world = load_world(POIS, ROADS); demand = DemandModel(world.zones, day='weekday')
    env = TransitEnv(world, demand, SimConfig(max_routes=20, max_buses=60), RewardWeights())
    obs, info = env.reset()
    obs, r, done, trunc, info = env.step([op, route, site])   # ops: see transit_rl/env.py OPS

Log an episode for the viewer: call `transit_rl.export.snapshot(env.sim, label)` after any step, collect the dicts
in a list and `export.write('viewer_data.json', world, demand, env.cfg, frames)`.

## Where to tune
* `demand.py`  – purposes, hourly profiles (WEEKDAY / WEEKEND), gravity decay `d0`, trip rates
* `data.py`    – building -> residents / jobs / pupils / attraction tables
* `sim.py`     – `SimConfig` (route cap, fleet, catchment, mode-choice, induced-demand strength) and `RewardWeights`
