"""Check 5: every path and split cools until the plate is fully solid, well before the cooling cap.
Uses the generator's own cooling settings (data/generate.py) at the highest-energy settings each split allows,
and at the coldest one. Runs that reach boiling are rejected by the generator, so they are only reported.
Usage: python -m verify.cooling
Pass: every run below boiling is complete, needs at most half of T_COOL_MAX, ends on a saved frame,
and a run without melting is not extended past T_COOL_MIN plus one saved frame."""
from solver.solver import simulate
from solver.paths import straight, l_turn, zigzag
from solver.ti64 import T_S
from data.generate import COOL, T_BOIL, T_COOL_MIN, T_COOL_MAX

# Splits other than test_extrap exclude the hot corner (P > 170 W and v < 0.65 m/s), so their highest-energy
# settings lie on its edges. (190 W, 0.66 m/s) is the zigzag that needed more than 1 ms of cooling.
EDGE = [(195.0, 0.65), (170.0, 0.5), (190.0, 0.66)]
CASES = [
    ("train, val, test_interp", "straight", straight(), {}, EDGE + [(50.0, 1.2)]),
    ("test_extrap", "straight", straight(), {}, [(195.0, 0.5)]),
    ("test_path", "l_turn", l_turn(), {}, EDGE + [(50.0, 1.2)]),
    ("test_path", "zigzag", zigzag(), {}, EDGE + [(50.0, 1.2)]),
    ("test_fine", "straight, 5 um", straight(), dict(nx=400, ny=200, dx=5e-6, save_every=40), EDGE[:2]),
    ("test_large", "straight, 4 x 2 mm", straight(4e-3, 2e-3), dict(nx=400, ny=200), EDGE[:2]),
]

failed, slowest = 0, 0.0
for split, name, wp, grid, settings in CASES:
    for P, v in settings:
        r = simulate(P, v, wp, **grid, **COOL)
        boils = r["T_max"] >= T_BOIL
        frame = grid.get("save_every", 10) * r["dt"]
        ok = r["cooling_complete"] and r["t_cool_actual"] <= T_COOL_MAX / 2 and r["terminal_saved"]
        if r["T_max"] < T_S:                                 # nothing melted: no extension
            ok = ok and r["t_cool_actual"] <= T_COOL_MIN + frame + r["dt"]
        if not boils:
            slowest = max(slowest, r["t_cool_actual"])
        failed += not (ok or boils)
        verdict = "boils, rejected by the generator" if boils else ("PASS" if ok else "FAIL")
        print(f"{split:24s} {name:19s} P={P:5.0f} W v={v:4.2f} m/s: T_max={r['T_max']:5.0f} K, "
              f"cooled {r['t_cool_actual']*1e3:5.3f} ms, complete={r['cooling_complete']}  {verdict}")
print(f"slowest run below boiling needs {slowest*1e3:.2f} ms; cap is {T_COOL_MAX*1e3:.1f} ms")
if failed:
    raise SystemExit(1)
