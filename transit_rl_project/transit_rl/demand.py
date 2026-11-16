"""Synthetic travel demand: purposes x time-of-day x gravity distribution."""
import numpy as np

HRS = np.arange(24)

def _prof(parts):
    p = np.zeros(24)
    for mu, sd, w in parts:
        d = np.abs(HRS - mu); d = np.minimum(d, 24 - d); p += w * np.exp(-0.5 * (d / sd) ** 2)
    return p / p.sum()

# 'out' = home -> destination leg, 'ret' = destination -> home leg. Each profile sums to 1 over 24h.
WEEKDAY = {
 'work':   dict(rate=.40, d0=3000., dest='jobs',   out=_prof([(7.9, .9, 1), (13.2, .8, .15)]), ret=_prof([(17.2, 1.2, 1), (13.5, .8, .15)])),
 'school': dict(rate=.22, d0=2000., dest='pupils', out=_prof([(8.4, .5, 1)]), ret=_prof([(15.3, .8, 1), (13., .5, .3)])),
 'other':  dict(rate=1.0, d0=2500., dest='other',  out=_prof([(10.5, 2., 1), (18., 1.5, .6)]), ret=_prof([(13.5, 2.5, 1), (20., 1.5, .5)])),
}
WEEKEND = {
 'work':   dict(WEEKDAY['work'], rate=.06),
 'school': dict(WEEKDAY['school'], rate=0.),
 'other':  dict(rate=1.5, d0=3000., dest='other', out=_prof([(11., 2.2, 1), (17., 2., .6)]), ret=_prof([(15., 3., 1), (21., 1.5, .5)])),
}


class DemandModel:
    """Base (no transit feedback) demand. T[k] are (Z,Z) daily trip matrices; C[h,k] their hourly share.
    OD at hour h = sum_k C[h,k] * T[k]  (home->dest terms and their transposes for the return leg)."""
    def __init__(s, zones, day='weekday', trip_scale=1.0, min_trip_m=500.):
        prm = WEEKDAY if day == 'weekday' else WEEKEND
        dx, dy = zones.x[:, None] - zones.x[None], zones.y[:, None] - zones.y[None]
        s.D = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        s.T, cols = [], []
        for p in prm.values():
            if p['rate'] == 0: continue
            f = np.exp(-s.D / p['d0']) * (s.D >= min_trip_m)
            K = f * getattr(zones, p['dest'])[None, :]
            K = K / (K.sum(1, keepdims=True) + 1e-9)
            G = (trip_scale * p['rate'] * zones.res)[:, None] * K
            s.T += [G.astype(np.float32), G.T.copy().astype(np.float32)]; cols += [p['out'], p['ret']]
        s.C = np.stack(cols, 1)                                   # (24, K)
        s.orig = s.C @ np.stack([t.sum(1) for t in s.T])          # (24, Z) trips leaving each zone
        s.dest = s.C @ np.stack([t.sum(0) for t in s.T])          # (24, Z) trips arriving
        s.total = float(sum(t.sum() for t in s.T))

    def od(s, h): return np.tensordot(s.C[h], np.stack(s.T), 1)
    def zone_hourly(s): return s.orig + s.dest
