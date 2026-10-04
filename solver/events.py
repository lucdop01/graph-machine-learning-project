"""Melting and solidification events per point, from a sequence of temperature states.

The same rules are used in two places, so solver reference values and evaluations are defined identically:
- inside the solver, fed every integration step (the reference event times stored in the dataset);
- in an evaluation, fed only the saved frames or a model's rollout outputs.
Between two consecutive states the temperature is taken as linear in time, so a crossing time is interpolated.

Definitions, with T_S the solidus and T_L the liquidus:
  n_melt      number of times T rose through T_L (full melting events; above 1 means the point remelted)
  t_liq_end   time T last fell through T_L, i.e. the end of the final fully liquid period
  t_sol_end   first time T fell through T_S after t_liq_end, i.e. the end of the final solidification.
              Reset to NaN when the point melts again. NaN if the point never fully melted or has not finished.
  T_peak      highest temperature seen
  t_peak      time of T_peak (resolution of the input sequence)
Final solidification duration = t_sol_end - t_liq_end. It is defined only for points that fully melted at least
once (n_melt >= 1) and finished solidifying. Partially melted points (T_S <= T_peak < T_L) have no duration and
are reported separately; they still count in melt-pool overlap through the melted fraction.
A point that fully melts, solidifies, and is later only reheated into the melting range (not above T_L) keeps
its earlier duration. A point that fully melts again gets the duration of its last solidification."""
import numpy as np
from .ti64 import T_S, T_L

FIELDS = ("n_melt", "t_liq_end", "t_sol_end", "T_peak", "t_peak")


class PhaseEvents:
    def __init__(self, T, t0=0.0, T_sol=T_S, T_liq=T_L):
        T = np.array(T, dtype=float)
        self.T_sol, self.T_liq = T_sol, T_liq
        self.prev, self.t_prev = T, float(t0)
        self.n_melt = (T >= T_liq).astype(np.int16)          # already liquid at the start counts as melted
        self.t_liq_end = np.full(T.shape, np.nan)
        self.t_sol_end = np.full(T.shape, np.nan)
        self.T_peak = T.copy()
        self.t_peak = np.full(T.shape, float(t0))

    def _cross(self, a, b, level, t, m):
        return self.t_prev + (a[m] - level) / (a[m] - b[m]) * (t - self.t_prev)

    def update(self, T, t):
        a, b = self.prev, np.asarray(T, dtype=float)
        up_L = (a < self.T_liq) & (b >= self.T_liq)
        if up_L.any():                                       # (re)melted: earlier solidification is superseded
            self.n_melt[up_L] += 1
            self.t_sol_end[up_L] = np.nan
        dn_L = (a >= self.T_liq) & (b < self.T_liq)
        if dn_L.any():
            self.t_liq_end[dn_L] = self._cross(a, b, self.T_liq, t, dn_L)
            self.t_sol_end[dn_L] = np.nan
        dn_S = (a >= self.T_sol) & (b < self.T_sol) & np.isfinite(self.t_liq_end) & np.isnan(self.t_sol_end)
        if dn_S.any():                                       # first solidus crossing after the last liquidus one
            self.t_sol_end[dn_S] = self._cross(a, b, self.T_sol, t, dn_S)
        hot = b > self.T_peak
        if hot.any():
            self.T_peak[hot] = b[hot]
            self.t_peak[hot] = t
        self.prev, self.t_prev = b.copy(), float(t)

    def result(self):
        return {f: getattr(self, f).copy() for f in FIELDS}


def duration(ev):
    """Final solidification duration per point (s); NaN where it is not defined."""
    return np.where(ev["n_melt"] >= 1, ev["t_sol_end"] - ev["t_liq_end"], np.nan)


def events_from_frames(T_frames, t):
    """Run the event rules on saved frames or model outputs: T_frames (n_frames, ...), t (n_frames,)."""
    rec = PhaseEvents(T_frames[0], t[0])
    for k in range(1, len(t)):
        rec.update(T_frames[k], t[k])
    return rec.result()


def bilinear_sampler(points, nx, ny, dx):
    """Fixed bilinear weights from the cell-centred solver grid to arbitrary points (metres, shape (m, 2)).
    Returns f(T) -> T at the points. Points within half a cell of the edge use the edge value."""
    p = np.asarray(points, float)
    fx, fy = p[:, 0] / dx - 0.5, p[:, 1] / dx - 0.5
    i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    wx, wy = np.clip(fx - i0, 0.0, 1.0), np.clip(fy - j0, 0.0, 1.0)

    def sample(T):
        return ((1 - wx) * (1 - wy) * T[..., i0, j0] + wx * (1 - wy) * T[..., i0 + 1, j0]
                + (1 - wx) * wy * T[..., i0, j0 + 1] + wx * wy * T[..., i0 + 1, j0 + 1])
    return sample
