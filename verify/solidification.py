"""Check 6: how sensitive the final solidification duration (solver/events.py) is to the grid, the integration
time step and the output cadence, measured separately.

All variants are sampled at the same fixed points with the same bilinear weights: a 10 um grid of points around
the track, each jittered by up to 20% (a stand-in for the planned mesh nodes). Reference event times are recorded
at every integration step. Cadence variants rebuild the events from the 10 um run's temperature at every n-th
solver step only, as an evaluation would from saved frames or model outputs (50 solver steps = one model step).

Differences are per point, relative to the reference, over points where both define a duration. "missed" counts
points that fully melted in the reference but never show T >= T_L in the sampled sequence; "spurious" the reverse.
Usage: python -m verify.solidification
The last block repeats two comparisons per solver cell instead of at the points. Cell values step through the
melting range one cell at a time, so per-cell durations depend on the cell size; evaluate at mesh nodes instead.
Pass: halving the time step changes the median duration by less than 1% (stored event times are converged in
time). Grid and cadence numbers are reported, not gated."""
import numpy as np
from solver.solver import simulate
from solver.paths import straight, zigzag
from solver.events import events_from_frames, duration, bilinear_sampler
from solver.ti64 import T_S

LX, LY = 2e-3, 1e-3
CADENCES = (10, 20, 50, 100)              # solver steps between samples: saved frames, E3 small, model step, E3 large

def points(x0, x1, y0, y1, h=10e-6, jitter=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h), indexing="ij")
    p = np.c_[X.ravel(), Y.ravel()]
    return p + rng.uniform(-jitter, jitter, p.shape) * h

def compare(ref, ev, label):
    d0, d1 = duration(ref), duration(ev)
    both = np.isfinite(d0) & np.isfinite(d1)
    rel = (d1[both] - d0[both]) / d0[both]
    missed = int(((ref["n_melt"] >= 1) & (ev["n_melt"] == 0)).sum())
    spurious = int(((ref["n_melt"] == 0) & (ev["n_melt"] >= 1)).sum())
    med0, med1 = np.median(d0[both]), np.median(d1[both])
    print(f"  {label:26s} median {med1*1e6:6.1f} us ({100*(med1-med0)/med0:+6.2f}% vs ref)   per point: "
          f"median |diff| {100*np.median(np.abs(rel)):5.1f}%, p90 {100*np.percentile(np.abs(rel), 90):5.1f}%   "
          f"missed {missed:4d}  spurious {spurious:3d}  of {int((ref['n_melt'] >= 1).sum())}")
    return 100 * (med1 - med0) / med0

def describe(ev, name):
    d = duration(ev)
    full = ev["n_melt"] >= 1
    partial = (ev["T_peak"] >= T_S) & ~full
    print(f"  {name}: {full.sum()} fully melted ({np.isfinite(d).sum()} finished, median {np.nanmedian(d)*1e6:.1f} us), "
          f"{partial.sum()} partially melted (no duration), {(ev['n_melt'] > 1).sum()} remelted")

worst_dt = 0.0
cases = [("straight", straight(), 195, 0.5, True), ("straight", straight(), 150, 0.65, True),
         ("straight", straight(), 120, 0.65, True), ("zigzag", zigzag(), 190, 0.66, False)]
for name, wp, P, v, grid_study in cases:
    if name == "straight":
        pts = points(0.3e-3, 1.7e-3, LY / 2 - 120e-6, LY / 2 + 120e-6)
    else:
        pts = points(0.4e-3, 1.6e-3, LY / 2 - 300e-6, LY / 2 + 300e-6)
    print(f"\n{name} {P} W {v} m/s, {len(pts)} points")
    run = lambda **kw: simulate(P, v, wp, t_cool_max=5e-3, events=pts, **kw)
    r10 = run(save_every=10)
    ref = r10["events"]
    describe(ref, "reference (10 um, every step)")
    s = bilinear_sampler(pts, 200, 100, 10e-6)
    Tp, tp = s(r10["T"].astype(float)), r10["t"]
    for c in CADENCES:
        idx = np.unique(np.r_[np.arange(0, len(tp), c // 10), len(tp) - 1])   # keep the final, fully solid frame
        compare(ref, events_from_frames(Tp[idx], tp[idx]), f"cadence {c:3d} steps ({c*r10['dt']*1e6:5.1f} us)")
    worst_dt = max(worst_dt, abs(compare(ref, run(cfl=0.1)["events"], "time step halved")))
    if grid_study:
        compare(ref, run(nx=100, ny=50, dx=20e-6)["events"], "grid 20 um")
        compare(ref, run(nx=400, ny=200, dx=5e-6, save_every=40)["events"], "grid 5 um")

print("\nper solver cell, straight 150 W 0.65 m/s (not used for evaluation)")
c10 = simulate(150, 0.65, straight(), t_cool_max=5e-3, events="grid")
c5 = simulate(150, 0.65, straight(), t_cool_max=5e-3, events="grid", nx=400, ny=200, dx=5e-6, save_every=40)
d10 = np.nanmedian(duration(c10["events"]))
print(f"  10 um cells, every step {d10*1e6:.1f} us, every 10th step "
      f"{np.nanmedian(duration(events_from_frames(c10['T'].astype(float), c10['t'])))*1e6:.1f} us, "
      f"5 um cells every step {np.nanmedian(duration(c5['events']))*1e6:.1f} us")

print(f"\nlargest change of the median from halving the time step: {worst_dt:.2f}%  {'PASS' if worst_dt < 1 else 'FAIL'}")
if worst_dt >= 1:
    raise SystemExit(1)
