"""Laser scan paths: constant speed along a polyline of waypoints (metres)."""
import numpy as np

class Path:
    def __init__(self, waypoints, v):
        self.p = np.asarray(waypoints, float)
        self.seg_len = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.v = v
        self.duration = self.cum[-1] / v          # s, laser switches off after this

    def pos(self, t):
        s = np.clip(self.v * t, 0.0, self.cum[-1])
        i = min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg_len) - 1)
        f = (s - self.cum[i]) / self.seg_len[i]
        return self.p[i] + f * (self.p[i + 1] - self.p[i])

# Domain is Lx x Ly = 2 mm x 1 mm by default
def straight(Lx=2e-3, Ly=1e-3):
    return [(0.2 * Lx, 0.5 * Ly), (0.8 * Lx, 0.5 * Ly)]

def l_turn(Lx=2e-3, Ly=1e-3):
    return [(0.2 * Lx, 0.3 * Ly), (0.6 * Lx, 0.3 * Ly), (0.6 * Lx, 0.8 * Ly)]

def zigzag(Lx=2e-3, Ly=1e-3, hatch=150e-6, n=3):
    pts, y = [], 0.5 * Ly - hatch * (n - 1) / 2
    for i in range(n):
        xa, xb = (0.25 * Lx, 0.75 * Lx) if i % 2 == 0 else (0.75 * Lx, 0.25 * Lx)
        pts += [(xa, y), (xb, y)]
        y += hatch
    return pts
