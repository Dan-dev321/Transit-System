"""Transit network + evaluator (catchments, routes, buses, mode choice, load, reward terms)."""
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class SimConfig:
    max_routes: int = 20            # customisable route cap
    max_buses: int = 60             # fleet size
    max_route_stops: int = 40
    bus_capacity: int = 60
    catchment_m: float = 400.       # stop catchment radius (fades to 0 at this distance)
    min_spacing_m: float = 250.     # stops closer than this overlap wastefully
    walk_mps: float = 1.25
    circuity: float = 1.3
    dwell_s: float = 20.
    layover_min: float = 4.
    board_pen_min: float = 2.       # extra cost per boarding (=> transfers hurt)
    max_wait_min: float = 20.
    car_kph: float = 32.
    car_overhead_min: float = 6.
    bias_min: float = 8.            # transit disadvantage vs car/walk
    scale_min: float = 8.           # logit scale
    max_share: float = 0.3
    service_hours: tuple = (6, 23)
    induced_gamma: float = 0.6      # demand uplift where transit access is good
    growth: float = 0.25            # how fast access feeds back into demand
    idle_pax_per_bus: float = 40.   # below this daily boardings/bus a route is "idle"
    min_stop_activity: float = 150. # catchment activity below this = wasted stop
    ref_share: float = 0.04         # riders / all trips that counts as score 1.0


@dataclass
class RewardWeights:
    riders: float = 1.0; coverage: float = 0.5; connect: float = 0.4; load: float = 0.3
    wait: float = 0.3; time: float = 0.3; unmet: float = 0.2
    overlap: float = 0.05; idle: float = 0.15; empty: float = 0.05; unrouted: float = 0.03


class TransitSim:
    def __init__(s, roads, zones, demand, cfg=None, weights=None):
        s.roads, s.z, s.d = roads, zones, demand
        s.cfg, s.w = cfg or SimConfig(), weights or RewardWeights()
        dx = zones.x[:, None] - roads.xy[None, :, 0]; dy = zones.y[:, None] - roads.xy[None, :, 1]
        s.zs = np.sqrt(dx * dx + dy * dy).astype(np.float32)          # zone -> site distance
        c = s.cfg; sh = np.arange(*c.service_hours)
        s.Cs = demand.C[sh]                                           # (service hours, K)
        s.Tst = np.stack(demand.T)
        s.reset()

    # ------------------------------------------------------------------ editing API (all return bool = valid)
    def reset(s):
        s.stops, s.routes = [], [dict(stops=[], buses=0) for _ in range(s.cfg.max_routes)]
        s.access = np.zeros(len(s.z.x)); s._acc_now = np.zeros(len(s.z.x)); s.last = None
        s.zone_pa = np.zeros(len(s.z.x)); s.zone_share = np.zeros(len(s.z.x))

    def fleet_used(s): return sum(r['buses'] for r in s.routes)
    def place_stop(s, site):
        if site in s.stops: return False
        s.stops.append(int(site)); return True
    def remove_stop(s, site):
        if site not in s.stops: return False
        s.stops.remove(site)
        for r in s.routes:
            if site in r['stops']: r['stops'].remove(site)
            if len(r['stops']) == 0: r['buses'] = 0
        return True
    def append(s, r, site):
        rt = s.routes[r]
        if site in rt['stops'] or len(rt['stops']) >= s.cfg.max_route_stops: return False
        if not rt['stops'] and s.fleet_used() >= s.cfg.max_buses: return False
        s.place_stop(site); rt['stops'].append(int(site)); rt['buses'] = max(rt['buses'], 1); return True
    def remove_from_route(s, r, site):
        rt = s.routes[r]
        if site not in rt['stops']: return False
        rt['stops'].remove(site)
        if not rt['stops']: rt['buses'] = 0
        return True
    def delete_route(s, r):
        if not s.routes[r]['stops']: return False
        s.routes[r] = dict(stops=[], buses=0); return True
    def add_bus(s, r):
        if not s.routes[r]['stops'] or s.fleet_used() >= s.cfg.max_buses: return False
        s.routes[r]['buses'] += 1; return True
    def remove_bus(s, r):
        if s.routes[r]['buses'] <= 1: return False
        s.routes[r]['buses'] -= 1; return True
    def grow(s): s.access = (1 - s.cfg.growth) * s.access + s.cfg.growth * s._acc_now

    # ------------------------------------------------------------------ evaluation
    def _ride_min(s, a, b): return float(s.roads.sssp(a)[0][b]) / 60 + s.cfg.dwell_s / 60

    def evaluate(s):
        c, z, d = s.cfg, s.z, s.d
        Z = len(z.x); R = c.catchment_m
        # ---- stop-level waste terms (all placed stops)
        stops = np.array(s.stops, int); overlap = empty = 0.; stop_act = np.zeros(0)
        if len(stops):
            wz = np.clip(1 - (s.zs[:, stops] / R) ** 2, 0, 1)
            stop_act = (z.act[:, None] * wz).sum(0)
            empty = float(np.sum(1 - np.minimum(1, stop_act / c.min_stop_activity)))
            xy = s.roads.xy[stops]; dd = np.linalg.norm(xy[:, None] - xy[None], axis=2)
            overlap = float(np.sum(np.triu(np.clip(1 - dd / c.min_spacing_m, 0, 1), 1)))
        # ---- active routes
        act = []
        for r, rt in enumerate(s.routes):
            if len(rt['stops']) >= 2 and rt['buses'] > 0:
                st = rt['stops']; tt = [s._ride_min(a, b) for a, b in zip(st[:-1], st[1:])]
                hw = (2 * sum(tt) + c.layover_min) / rt['buses']
                seg = [1.25 * np.linalg.norm(s.roads.xy[a] - s.roads.xy[b]) / 1000 for a, b in zip(st[:-1], st[1:])]
                act.append((r, st, tt, hw, seg))
        m = dict(stops=len(s.stops), routes=len(act), buses=s.fleet_used(), overlap=overlap, empty_stops=empty,
                 riders=0., coverage=0., connect=0., load_factor=0., avg_wait=0., avg_time=0., unmet=0.,
                 idle_routes=0., unrouted=0., route_info=[], total_trips=d.total)
        A = sorted({x for a in act for x in a[1]})
        m['unrouted'] = float(len([x for x in s.stops if x not in A]))
        if not act:
            s._acc_now[:] = 0; s.zone_pa[:] = 0; s.zone_share[:] = 0
            return s._finish(m)
        aidx = {x: i for i, x in enumerate(A)}; S = len(A)
        # ---- expanded graph: hub per stop + one node per (route, stop)
        rows, cols, wts, off, rinfo = [], [], [], S, []
        for r, st, tt, hw, seg in act:
            base, n = off, len(st); off += n
            for k, site in enumerate(st):
                h, node = aidx[site], base + k
                rows += [h, node]; cols += [node, h]; wts += [min(hw / 2, c.max_wait_min) + c.board_pen_min, 1e-3]
                if k < n - 1: rows += [node, node + 1]; cols += [node + 1, node]; wts += [tt[k], tt[k]]
            rinfo.append((r, base, n, hw, seg))
        Nn = off
        dist, pred = dijkstra(csr_matrix((wts, (rows, cols)), shape=(Nn, Nn)), directed=True,
                              indices=np.arange(S), return_predecessors=True)
        TT = dist[:, :S]
        # ---- zone access + transit time + mode share
        dzA = s.zs[:, A]; near = dzA.argmin(1); dw = dzA[np.arange(Z), near]
        pa = np.clip(1 - (dw / R) ** 2, 0, 1); walk = dw * c.circuity / c.walk_mps / 60
        Tz = walk[:, None] + TT[near][:, near] + walk[None, :]
        Tz[near[:, None] == near[None, :]] = np.inf
        ok = np.isfinite(Tz) & (pa[:, None] * pa[None, :] > 0)
        alt = np.minimum(d.D * c.circuity / c.walk_mps / 60, c.car_overhead_min + d.D * c.circuity / (c.car_kph / 3.6) / 60)
        x = np.where(ok, (Tz - alt - c.bias_min) / c.scale_min, 50.)
        M = (c.max_share / (1 + np.exp(np.clip(x, -50, 50))) * pa[:, None] * pa[None, :] * ok).astype(np.float32)
        Tz0 = np.where(ok, Tz, 0)
        mult = (1 + c.induced_gamma * s.access).astype(np.float32); mm = mult[:, None] * mult[None, :]
        idx = (near[:, None] * S + near[None, :]).ravel()
        K = len(d.T); SV = np.zeros(K); ST = np.zeros(K); SD = np.zeros(K); SR = np.zeros(K)
        loads = np.zeros((3, K, Nn))
        for k in range(K):
            base_k = d.T[k] * mm; V = base_k * M
            SV[k], ST[k], SD[k], SR[k] = V.sum(), (V * Tz0).sum(), base_k.sum(), (base_k * (M > 0)).sum()
            F = np.bincount(idx, weights=V.ravel(), minlength=S * S).reshape(S, S)
            loads[:, k] = _trace(pred, S, Nn, F)
        hk = s.Cs.sum(0); raw = hk @ SV
        if raw <= 1e-6: return s._finish(m)
        # ---- per-hour capacity
        brd, fwd, bwd = (s.Cs @ loads[i] for i in range(3))             # (hours, Nn)
        cap = np.zeros(Nn); segF = np.zeros(Nn); segB = np.zeros(Nn)
        for r, base, n, hw, seg in rinfo:
            cap[base:base + n] = 60 / hw * c.bus_capacity
            segF[base + 1:base + n] = seg; segB[base:base + n - 1] = seg
        exc = np.maximum(0, fwd - cap).sum(1) + np.maximum(0, bwd - cap).sum(1)
        tot = (fwd + bwd).sum(1)
        lost = np.clip(np.divide(exc, tot, out=np.zeros_like(exc), where=tot > 0), 0, .9)
        keep = (1 - lost)[:, None]
        served = float(((s.Cs @ SV) * (1 - lost)).sum())
        brd_n, paxkm = (brd * keep).sum(0), ((fwd * segF + bwd * segB) * keep).sum()
        capkm, board_tot, wait_w = 0., 0., 0.; idle = 0.
        nh = len(s.Cs); route_info = []
        for r, base, n, hw, seg in rinfo:
            b = s.routes[r]['buses']; bd = float(brd_n[base:base + n].sum())
            capkm += 60 / hw * 2 * sum(seg) * c.bus_capacity * nh
            board_tot += bd; wait_w += bd * min(hw / 2, c.max_wait_min)
            idle += float(np.clip(1 - bd / b / c.idle_pax_per_bus, 0, 1))
            route_info.append(dict(route=r, buses=b, headway=round(hw, 1), boardings=round(bd), length_km=round(sum(seg), 1)))
        # ---- coverage / connectivity / induced-demand state
        cov = float((z.act * pa).sum() / z.act.sum())
        reach = float(hk @ SR / (hk @ SD))   # share of (demand-weighted) trips with a transit option
        lcc = _lcc_fraction([a[1] for a in act])
        bph = {}
        for r, st, tt, hw, seg in act:
            for site in st: bph[site] = bph.get(site, 0) + 60 / hw
        ff = np.array([min(1, bph[a] / 6) for a in A])
        s._acc_now = pa * ff[near]; s.zone_pa = pa; s.zone_share = M.mean(1) / c.max_share
        m.update(riders=served, coverage=cov, connect=.5 * reach + .5 * lcc,
                 load_factor=float(paxkm / capkm) if capkm else 0., avg_wait=wait_w / max(board_tot, 1e-9),
                 avg_time=float(hk @ ST / (hk @ SV)), unmet=float(1 - served / raw), idle_routes=idle, route_info=route_info)
        return s._finish(m)

    def _finish(s, m):
        c, w = s.cfg, s.w
        rn = m['riders'] / (c.ref_share * m['total_trips']); q = min(1, rn)
        m['score'] = float(w.riders * rn + w.coverage * m['coverage'] + w.connect * m['connect']
                           + w.load * min(m['load_factor'], .7) / .7
                           - q * (w.wait * m['avg_wait'] / 15 + w.time * m['avg_time'] / 40) - w.unmet * m['unmet']
                           - w.overlap * m['overlap'] - w.idle * m['idle_routes'] - w.empty * m['empty_stops']
                           - w.unrouted * m['unrouted'])
        s.last = m; return m


def _trace(pred, S, Nn, F):
    """Accumulate stop-pair flows F onto expanded-graph links -> (board, forward-ride, backward-ride) loads."""
    i, j = np.nonzero(F > 1e-9); f = F[i, j]; src, cur = i, j
    out = np.zeros((3, Nn))
    for _ in range(300):
        p = pred[src, cur]; m = (cur != src) & (p >= 0)
        if not m.any(): break
        src, cur, p, f = src[m], cur[m], p[m], f[m]
        rn = cur >= S
        for t, sel in enumerate((rn & (p < S), rn & (p == cur - 1) & (p >= S), rn & (p == cur + 1) & (p >= S))):
            if sel.any(): out[t] += np.bincount(cur[sel], weights=f[sel], minlength=Nn)
        cur = p
    return out


def _lcc_fraction(routes):
    par = list(range(len(routes)))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    owner = {}
    for r, st in enumerate(routes):
        for x in st:
            if x in owner: par[f(r)] = f(owner[x])
            else: owner[x] = r
    cnt = {}
    for r in range(len(routes)): cnt[f(r)] = cnt.get(f(r), 0) + 1
    return max(cnt.values()) / len(routes)
