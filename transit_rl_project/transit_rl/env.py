"""Gymnasium environment: the agent edits a bus network; reward = change in network score."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sim import TransitSim

OPS = ['STOP', 'PLACE_STOP', 'REMOVE_STOP', 'APPEND_TO_ROUTE', 'REMOVE_FROM_ROUTE', 'DELETE_ROUTE', 'ADD_BUS', 'REMOVE_BUS']


class TransitEnv(gym.Env):
    """action = [op, route, site]   (route/site ignored by ops that don't use them)
       obs    = {'grid': (6,H,W) float32, 'vec': (6,) float32}
       reward = score(after) - score(before)  (dense, potential-based) - small penalty for invalid actions."""
    metadata = {'render_modes': []}

    def __init__(self, world, demand, cfg=None, weights=None, max_steps=150, invalid_penalty=0.01, seed=None):
        self.world, self.sim = world, TransitSim(world.roads, world.zones, demand, cfg, weights)
        self.cfg, self.max_steps, self.inv = self.sim.cfg, max_steps, invalid_penalty
        z = world.zones; H, W = z.H, z.W
        self.action_space = spaces.MultiDiscrete([len(OPS), self.cfg.max_routes, world.roads.N])
        self.observation_space = spaces.Dict({'grid': spaces.Box(0, 1, (6, H, W), np.float32),
                                              'vec': spaces.Box(-5, 5, (6,), np.float32)})
        pk = demand.zone_hourly()[[7, 8, 9, 16, 17, 18]].sum(0)
        self._static = (z.act / z.act.max(), pk / pk.max())
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.sim.reset(); self.t = 0
        self.m = self.sim.evaluate(); return self._obs(), {'metrics': self.m}

    def step(self, action):
        op, r, site = (int(a) for a in action); sim = self.sim; self.t += 1
        ok = {0: True, 1: sim.place_stop, 2: sim.remove_stop, 3: sim.append, 4: sim.remove_from_route,
              5: sim.delete_route, 6: sim.add_bus, 7: sim.remove_bus}[op]
        if op == 0: valid = True
        elif op in (1, 2): valid = ok(site)
        elif op in (3, 4): valid = ok(r, site)
        else: valid = ok(r)
        before = self.m['score']; self.m = sim.evaluate(); sim.grow()
        reward = self.m['score'] - before - (0 if valid else self.inv)
        done = op == 0; trunc = self.t >= self.max_steps
        return self._obs(), float(reward), done, trunc, {'metrics': self.m, 'valid': valid}

    def _obs(self):
        z, s = self.world.zones, self.sim; g = np.zeros((6, z.H, z.W), np.float32)
        a, pk = self._static
        g[0, z.gy, z.gx] = a; g[1, z.gy, z.gx] = pk; g[2, z.gy, z.gx] = s.zone_pa; g[3, z.gy, z.gx] = s.zone_share
        if s.stops:
            xy = s.roads.xy[s.stops]
            gx, gy = ((xy[:, 0] - z.x0) // z.cell).astype(int), ((xy[:, 1] - z.y0) // z.cell).astype(int)
            ok = (gx >= 0) & (gx < z.W) & (gy >= 0) & (gy < z.H); np.add.at(g[4], (gy[ok], gx[ok]), .25)
        g[5, z.gy, z.gx] = np.clip(s.access, 0, 1)
        m = self.m; c = self.cfg
        vec = [1 - self.t / self.max_steps, 1 - m['buses'] / c.max_buses, m['routes'] / c.max_routes,
               m['coverage'], m['riders'] / (c.ref_share * m['total_trips']), m['score']]
        return {'grid': np.clip(g, 0, 1), 'vec': np.array(vec, np.float32)}

    @property
    def roads(self): return self.world.roads
