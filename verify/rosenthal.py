"""Check 2: with constant properties and no melting, the steady field around the laser must match
Rosenthal's thin-plate solution, superposed over the Gaussian spot.
Usage: python -m verify.rosenthal     Pass: error < 2 % of the temperature rise at every probe."""
import numpy as np
from scipy.special import k0
from solver.solver import simulate
from solver.paths import straight
from solver.ti64 import RHO, T0, ETA, R_SPOT

K, CP = 20.0, 600.0
props = dict(k=lambda T: np.full_like(np.asarray(T, float), K), cp=lambda T: np.full_like(np.asarray(T, float), CP),
             H_from_T=lambda T: CP * (T - T0), T_from_H=lambda H: T0 + H / CP)
alpha = K / (RHO * CP)

def rosenthal_point(P, v, xi, eta, d):
    return ETA * P / (2 * np.pi * K * d) * np.exp(-v * xi / (2 * alpha)) * k0(v * np.hypot(xi, eta) / (2 * alpha))

def rosenthal_gauss(P, v, xi, eta, d, h=2.5e-6):
    s = np.arange(-3 * R_SPOT + h / 2, 3 * R_SPOT, h)
    SX, SY = np.meshgrid(s, s, indexing="ij")
    w = np.exp(-2 * (SX**2 + SY**2) / R_SPOT**2)
    w /= w.sum()
    return T0 + np.sum(w * rosenthal_point(P, v, xi - SX, eta - SY, d))

P, v, dx, nx, ny = 120.0, 0.8, 5e-6, 400, 200
# save (almost) only the final step; the margin of 2 steps guards against rounding in the step count
n_steps = int(np.ceil(1.2e-3 / v / (0.2 * dx**2 / alpha)))
r = simulate(P, v, straight(), nx=nx, ny=ny, dx=dx, save_every=max(1, n_steps - 2), t_cool=0.0, props=props)
T, (xl, yl) = r["T"][-1].astype(float), r["laser"][-1]
x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dx
failed = 0
for xi, eta in [(-400e-6, 0), (-200e-6, 0), (-100e-6, 0), (-50e-6, 0), (0, 60e-6), (-200e-6, 150e-6)]:
    col = [np.interp(xl + xi, x, T[:, j]) for j in range(ny)]
    Tsim, Tex = np.interp(yl + eta, y, col), rosenthal_gauss(P, v, xi, eta, r["d"])
    err = 100 * (Tsim - Tex) / (Tex - T0)
    failed += abs(err) >= 2
    print(f"xi={xi*1e6:5.0f} um eta={eta*1e6:4.0f} um: solver={Tsim:7.1f} K exact={Tex:7.1f} K error={err:+.1f}% of rise  {'PASS' if abs(err) < 2 else 'FAIL'}")
if failed:
    raise SystemExit(1)
