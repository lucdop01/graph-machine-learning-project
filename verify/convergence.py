"""Check 4: grid and time-step convergence with the real Ti-6Al-4V properties (150 W, 0.8 m/s, straight track).
Runs 20, 10 and 5 um grids (time step shrinks with dx^2) and a 10 um run with half the time step.
Fields are compared on the 5 um grid at identical times.
Usage: python -m verify.convergence
Pass: 10 vs 5 um peak T within 2%. The melting-range difference is reported as the resolution sensitivity
of the 10 um data, the solver-vs-solver gap to show next to model errors on test_fine."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from solver.solver import simulate
from solver.paths import straight
from solver.ti64 import liquid_fraction

P, V, LX, LY = 150.0, 0.8, 2e-3, 1e-3
runs = {  # name: (dx, save_every); every run saves at the same physical times
    "20um": dict(dx=20e-6, save_every=10),
    "10um": dict(dx=10e-6, save_every=40),
    "5um": dict(dx=5e-6, save_every=160),
    "10um_halfdt": dict(dx=10e-6, save_every=80, cfl=0.1),
}
res = {}
for name, kw in runs.items():
    dx = kw["dx"]
    res[name] = simulate(P, V, straight(LX, LY), nx=round(LX / dx), ny=round(LY / dx), **kw)
    print(f"{name:12s} dt = {res[name]['dt']*1e9:7.1f} ns, frames = {len(res[name]['t'])}, T_max = {res[name]['T_max']:.0f} K")

ref = res["5um"]
xr = (np.arange(ref["T"].shape[1]) + 0.5) * 5e-6
yr = (np.arange(ref["T"].shape[2]) + 0.5) * 5e-6
X, Y = np.meshgrid(xr, yr, indexing="ij")
pts = np.c_[X.ravel(), Y.ravel()]
n = min(len(r["t"]) for r in res.values())
frames = range(n // 4, n, max(1, n // 12))   # skip the start-up, sample through the track and cooling

def on_ref(r, k):
    dx = r["dx"]
    nx, ny = r["T"].shape[1:]
    g = RegularGridInterpolator(((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx), r["T"][k].astype(float),
                                bounds_error=False, fill_value=None)
    return g(pts).reshape(X.shape)

peak10 = None
print("\nvs 5 um reference      peak T diff   mean |dT| near track   mean |dT| melting range   melt-pool area diff")
for name in ("20um", "10um", "10um_halfdt"):
    pk, near, band, area = [], [], [], []
    for k in frames:
        assert np.isclose(res[name]["t"][k], ref["t"][k])
        A, B = on_ref(res[name], k), ref["T"][k].astype(float)
        pk.append(100 * (A.max() - B.max()) / B.max())
        m_near = np.abs(Y - LY / 2) < 150e-6
        near.append(np.abs(A - B)[m_near].mean())
        m_band = (B >= 1800) & (B <= 2000)
        if m_band.any():
            band.append(np.abs(A - B)[m_band].mean())
        aB = (liquid_fraction(B) > 0.5).sum()
        if aB > 0:
            area.append(100 * ((liquid_fraction(A) > 0.5).sum() - aB) / aB)
    if name == "10um":
        peak10 = np.mean(np.abs(pk))
    print(f"{name:20s} {np.mean(pk):+8.2f}%   {np.mean(near):10.2f} K   {np.mean(band):14.2f} K   "
          f"{np.mean(area) if area else float('nan'):+12.1f}%")

# time step alone: 10 um at cfl 0.2 vs cfl 0.1, compared on the 10 um grid
a, b = res["10um"]["T"], res["10um_halfdt"]["T"]
d = [np.abs(a[k].astype(float) - b[k].astype(float)).max() for k in frames]
print(f"\ntime step only (10 um, dt vs dt/2): max |dT| = {max(d):.2f} K")
print(f"10 vs 5 um mean |peak T difference| = {peak10:.2f}%  {'PASS' if peak10 < 2 else 'FAIL'}")
if peak10 >= 2:
    raise SystemExit(1)
