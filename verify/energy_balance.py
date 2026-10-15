"""Check 1: energy in from the laser must equal the change in stored enthalpy (edges are insulated).
Usage: python -m verify.energy_balance     Pass: |error| < 1 %."""
from solver.solver import simulate
from solver.paths import straight, zigzag
from solver.ti64 import RHO, H_from_T, ETA

failed = 0
for name, wp, P, v in [("straight", straight(), 120.0, 0.8), ("straight", straight(), 195.0, 0.5), ("zigzag", zigzag(), 137.0, 1.06)]:
    r = simulate(P, v, wp)
    e_in = ETA * P * r["laser_off"]                                     # J (laser on for laser_off seconds)
    e_stored = RHO * r["d"] * r["dx"] ** 2 * (H_from_T(r["T"][-1].astype(float)) - H_from_T(r["T"][0].astype(float))).sum()
    err = 100 * (e_stored - e_in) / e_in
    failed += abs(err) >= 1
    print(f"{name:8s} P={P:.0f} W v={v} m/s: in={e_in*1e3:.3f} mJ stored={e_stored*1e3:.3f} mJ error={err:+.2f}%  {'PASS' if abs(err) < 1 else 'FAIL'}")
if failed:
    raise SystemExit(1)
