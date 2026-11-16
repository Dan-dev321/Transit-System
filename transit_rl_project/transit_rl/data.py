"""Load the POI GeoJSON + road GraphML into simulation-ready arrays.

Everything is projected to local metres (x east, y north). Roads are densified so
bus stops can be placed every <= max_edge metres ("sites" = candidate stop locations).
"""
import json, math
from dataclasses import dataclass
import numpy as np, networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

M_PER_DEG = 111194.9


def _t(res=0, jobs=0, pupils=0, other=0, cls=6):
    # res: residents | jobs: workers | pupils: school/college places | other: shopping/leisure pull
    # cls (viewer colour): 0 home 1 retail/commercial 2 office 3 industrial 4 education 5 park/leisure 6 other
    return dict(res=res, jobs=jobs, pupils=pupils, other=other, cls=cls)

BUILDING = {'house': _t(res=2.7), 'detached': _t(res=2.7), 'residential': _t(res=6), 'apartments': _t(res=18),
            'commercial': _t(jobs=18, other=4, cls=1), 'retail': _t(jobs=12, other=10, cls=1),
            'office': _t(jobs=30, cls=2), 'industrial': _t(jobs=15, cls=3), 'warehouse': _t(jobs=6, cls=3)}
AMENITY = {'school': _t(jobs=35, pupils=350, cls=4), 'college': _t(jobs=60, pupils=500, cls=4),
           'library': _t(jobs=6, other=15, cls=1), 'cafe': _t(jobs=5, other=12, cls=1), 'pub': _t(jobs=5, other=15, cls=1),
           'restaurant': _t(jobs=8, other=15, cls=1), 'fast_food': _t(jobs=5, other=10, cls=1),
           'clinic': _t(jobs=12, other=20, cls=6)}
LEISURE = {'park': _t(other=30, cls=5), 'sports_centre': _t(jobs=8, other=25, cls=5)}
SHOP = {'supermarket': _t(jobs=25, other=60, cls=1)}; SHOP_DEFAULT = _t(jobs=3, other=6, cls=1)
OFFICE = {'government': _t(jobs=30, other=8, cls=2)}; OFFICE_DEFAULT = _t(jobs=8, cls=2)


def classify(p):
    a, s, o, l, b = (p.get(k) for k in ('amenity', 'shop', 'office', 'leisure', 'building'))
    if a in AMENITY: return AMENITY[a]
    if l in LEISURE: return LEISURE[l]
    if s: return None if s == 'vacant' else SHOP.get(s, SHOP_DEFAULT)
    if o: return None if o == 'vacant' else OFFICE.get(o, OFFICE_DEFAULT)
    if b in BUILDING: return BUILDING[b]
    if a: return _t(jobs=3, other=3)
    if l: return _t(other=8, cls=5)
    return None


class Projector:
    def __init__(s, lon0, lat0): s.lon0, s.lat0, s.k = lon0, lat0, math.cos(math.radians(lat0))
    def __call__(s, lon, lat):
        return (np.asarray(lon) - s.lon0) * s.k * M_PER_DEG, (np.asarray(lat) - s.lat0) * M_PER_DEG


def load_pois(path, proj):
    keep, lon, lat = [], [], []
    for f in json.load(open(path, encoding='utf-8'))['features']:
        t = classify(f['properties'])
        if t is None or f['geometry']['type'] != 'Point': continue
        keep.append(t); lon.append(f['geometry']['coordinates'][0]); lat.append(f['geometry']['coordinates'][1])
    x, y = proj(lon, lat)
    P = {k: np.array([t[k] for t in keep], float) for k in ('res', 'jobs', 'pupils', 'other')}
    P['cls'] = np.array([t['cls'] for t in keep], int); P['x'], P['y'] = x, y
    return P


@dataclass
class Zones:
    x: np.ndarray; y: np.ndarray; res: np.ndarray; jobs: np.ndarray; pupils: np.ndarray; other: np.ndarray
    act: np.ndarray; gx: np.ndarray; gy: np.ndarray; H: int; W: int; cell: float; x0: float; y0: float


def build_zones(P, cell=200.0):
    x0, y0 = P['x'].min(), P['y'].min()
    ix, iy = ((P['x'] - x0) // cell).astype(int), ((P['y'] - y0) // cell).astype(int)
    W, H = ix.max() + 1, iy.max() + 1
    key = iy * W + ix
    uk, inv = np.unique(key, return_inverse=True)
    agg = lambda v: np.bincount(inv, weights=v, minlength=len(uk))
    res, jobs, pup, oth = (agg(P[k]) for k in ('res', 'jobs', 'pupils', 'other'))
    w = P['res'] + P['jobs'] + P['pupils'] * .5 + P['other'] * .3 + 1e-6
    act = res + jobs + .5 * pup + .3 * oth
    return Zones(agg(P['x'] * w) / agg(w), agg(P['y'] * w) / agg(w), res, jobs, pup, oth, act,
                 uk % W, uk // W, H, W, cell, x0, y0)


def _cut(P, cum, a, b):
    def pt(d):
        k = min(max(np.searchsorted(cum, d, side='right') - 1, 0), len(P) - 2)
        seg = cum[k + 1] - cum[k]
        return P[k] + (0 if seg == 0 else (d - cum[k]) / seg) * (P[k + 1] - P[k])
    return np.array([pt(a)] + [P[k] for k in range(len(P)) if a < cum[k] < b] + [pt(b)])


class Roads:
    """Undirected, densified road graph. Edge weights = bus travel seconds."""
    def __init__(s, path, proj, max_edge=150.0, bus_factor=1.8, bus_max_kph=50.0):
        G = nx.read_graphml(path)
        pos = {n: np.array(proj(float(d['x']), float(d['y']))) for n, d in G.nodes(data=True)}
        best = {}
        for u, v, d in G.edges(data=True):
            if u == v: continue
            hw = d.get('highway', '')
            major = int(hw in ('trunk', 'primary', 'secondary', 'tertiary'))
            if 'geometry' in d:
                pts = d['geometry'][d['geometry'].index('(') + 1:-1].split(',')
                ll = np.array([[float(c) for c in p.split()] for p in pts])
                poly = np.column_stack(proj(ll[:, 0], ll[:, 1]))
            else: poly = np.array([pos[u], pos[v]])
            L = float(d.get('length', 0)) or float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
            t = max(float(d.get('travel_time', 0)), L / (bus_max_kph / 3.6)) * bus_factor
            a, b = (u, v) if u < v else (v, u)
            if u > v: poly = poly[::-1]
            if (a, b) not in best or t < best[(a, b)][0]: best[(a, b)] = (t, poly, major)
        ids = {n: i for i, n in enumerate(pos)}; xy = [pos[n] for n in pos]
        A, B, T, MJ, POLY = [], [], [], [], []
        for (a, b), (t, poly, major) in best.items():
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
            L = cum[-1]; n = max(1, math.ceil(L / max_edge)); prev = ids[a]
            for k in range(1, n + 1):
                nxt = ids[b] if k == n else len(xy)
                piece = _cut(poly, cum, L * (k - 1) / n, L * k / n)
                if k < n: xy.append(piece[-1])
                A.append(prev); B.append(nxt); T.append(t / n); MJ.append(major); POLY.append(piece); prev = nxt
        N = len(xy); A, B, T = np.array(A), np.array(B), np.array(T)
        _, lab = connected_components(coo_matrix((T, (A, B)), shape=(N, N)), directed=False)
        keep = lab == np.bincount(lab).argmax(); remap = -np.ones(N, int); remap[keep] = np.arange(keep.sum())
        em = keep[A] & keep[B]
        s.xy = np.array(xy)[keep]; s.N = len(s.xy)
        A, B, T = remap[A[em]], remap[B[em]], T[em]
        s.edges = [(a, b, POLY[i], MJ[i]) for a, b, i in zip(A, B, np.nonzero(em)[0])]
        s.poly = {}
        for a, b, p, _ in s.edges: s.poly[(a, b)] = p; s.poly[(b, a)] = p[::-1]
        s.csr = coo_matrix((np.r_[T, T], (np.r_[A, B], np.r_[B, A])), shape=(s.N, s.N)).tocsr()
        s._cache = {}

    def sssp(s, i):
        if i not in s._cache:
            d, p = dijkstra(s.csr, directed=False, indices=i, return_predecessors=True)
            s._cache[i] = (d.astype(np.float32), p.astype(np.int32))
        return s._cache[i]

    def path(s, i, j):
        _, pred = s.sssp(i); out = [j]
        while out[-1] != i: out.append(int(pred[out[-1]]))
        return out[::-1]

    def polyline(s, sites):
        pts = []
        for a, b in zip(sites[:-1], sites[1:]):
            pth = s.path(a, b)
            for q in zip(pth[:-1], pth[1:]):
                pts.extend(s.poly[q][1:] if pts else s.poly[q])
        return np.array(pts) if pts else np.zeros((0, 2))


@dataclass
class World:
    proj: Projector; poi: dict; roads: Roads; zones: Zones


def load_world(pois_path, roads_path, cell=200.0, max_edge=150.0):
    import xml.etree.ElementTree as ET
    lons, lats = [], []
    for ev, el in ET.iterparse(roads_path):
        if el.tag.endswith('node'):
            d = {c.get('key'): c.text for c in el}
            lats.append(float(d['d4'])); lons.append(float(d['d5'])); el.clear()
    proj = Projector(np.mean(lons), np.mean(lats))
    P = load_pois(pois_path, proj)
    return World(proj, P, Roads(roads_path, proj, max_edge), build_zones(P, cell))
