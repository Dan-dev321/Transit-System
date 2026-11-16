"""Export map + episode snapshots to a JSON file the browser viewer (viewer.html) can play."""
import json
import numpy as np


def map_payload(world, demand, cfg):
    r, z, P = world.roads, world.zones, world.poi
    pts = np.column_stack([P['x'], P['y'], P['cls']]).round().astype(int).tolist()
    roads = [[int(mj), np.asarray(p).round().astype(int).ravel().tolist()] for _, _, p, mj in r.edges]
    zh = demand.zone_hourly()
    zones = [[int(zx), int(zy), [int(round(v)) for v in zh[:, i]]] for i, (zx, zy) in enumerate(zip(z.x, z.y))]
    return dict(roads=roads, bld=pts, zones=zones, cell=z.cell, R=cfg.catchment_m, service=list(cfg.service_hours),
                bounds=[float(r.xy[:, 0].min()), float(r.xy[:, 1].min()), float(r.xy[:, 0].max()), float(r.xy[:, 1].max())])


def snapshot(sim, label=''):
    m = sim.last or sim.evaluate(); r = sim.roads; info = {i['route']: i for i in m['route_info']}
    routes = []
    for i, rt in enumerate(sim.routes):
        if len(rt['stops']) < 2 or rt['buses'] < 1: continue
        inf = info.get(i, {})
        routes.append(dict(i=i, b=rt['buses'], h=inf.get('headway', 0), n=inf.get('boardings', 0),
                           s=[r.xy[x].round().astype(int).tolist() for x in rt['stops']],
                           p=r.polyline(rt['stops']).round().astype(int).ravel().tolist()))
    keys = ['score', 'riders', 'coverage', 'connect', 'load_factor', 'avg_wait', 'avg_time', 'unmet', 'overlap',
            'empty_stops', 'idle_routes', 'stops', 'routes', 'buses']
    return dict(label=label, stops=[r.xy[x].round().astype(int).tolist() for x in sim.stops], routes=routes,
                m={k: round(float(m[k]), 3) for k in keys})


def write(path, world, demand, cfg, frames):
    json.dump(dict(map=map_payload(world, demand, cfg), frames=frames), open(path, 'w', encoding='utf-8'), separators=(',', ':'))
