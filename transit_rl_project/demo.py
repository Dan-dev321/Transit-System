"""Build a heuristic network step by step, log every frame, and write viewer_data.json (+ standalone HTML)."""
import sys, numpy as np
from transit_rl import *
from transit_rl.export import snapshot, write

POIS, ROADS = sys.argv[1:3] if len(sys.argv) > 2 else ('kilkenny_pois.geojson', 'kilkenny_roads.graphml')
world = load_world(POIS, ROADS); dem = DemandModel(world.zones); cfg = SimConfig()
sim = TransitSim(world.roads, world.zones, dem, cfg); z, rd = world.zones, world.roads
site_of = lambda i: int(np.argmin(sim.zs[i]))
frames = [snapshot(sim, 'empty network')]; sim.evaluate()
centre = site_of(int(np.argmax(z.act))); targets = []
for i in np.argsort(-z.act)[:120]:                       # farthest-first over dense zones
    p = np.array([z.x[i], z.y[i]])
    if np.linalg.norm(p - [z.x[np.argmax(z.act)], z.y[np.argmax(z.act)]]) > 1000 and \
       all(np.linalg.norm(p - q) > 1100 for q in targets): targets.append(p)
    if len(targets) == 8: break
for r, tp in enumerate(targets):
    tgt = int(np.argmin(np.linalg.norm(rd.xy - tp, axis=1)))
    path, last = rd.path(centre, tgt), None
    for n in path:
        if last is None or np.linalg.norm(rd.xy[n] - rd.xy[last]) > 380 or n == path[-1]:
            if last is None or n != last: sim.append(r, n); last = n
    for _ in range(2): sim.add_bus(r)
    sim.evaluate(); sim.grow(); frames.append(snapshot(sim, f'route {r + 1} built'))
for rep in range(3):                                      # let induced demand build up, add buses where loaded
    m = sim.evaluate(); best = max(m['route_info'], key=lambda i: i['boardings'] / i['buses'])
    sim.add_bus(best['route']); sim.evaluate(); sim.grow(); frames.append(snapshot(sim, f'+bus on route {best["route"] + 1}'))
for f in frames: print(f['label'].ljust(22), f['m'])
write('viewer_data.json', world, dem, cfg, frames)
