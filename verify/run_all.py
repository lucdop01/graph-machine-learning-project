"""Run all checks: the solver and the graph preprocessing. Exits with code 1 if any check fails.
Usage (from the repo root):  python -m verify.run_all      (about 11 minutes)
                             python -m verify.run_all --quick   (the first four checks only, about 2 minutes)"""
import runpy
import sys
import time

CHECKS = ["verify.energy_balance", "verify.rosenthal", "verify.stefan", "verify.convergence"]
if "--quick" not in sys.argv:
    CHECKS += ["verify.cooling", "verify.solidification", "verify.graph_energy"]

failed = []
for mod in CHECKS:
    t0 = time.time()
    print(f"\n=== {mod}")
    try:
        runpy.run_module(mod, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            failed.append(mod)
    print(f"--- {time.time() - t0:.0f} s")
print("\nALL CHECKS PASS" if not failed else f"\nFAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
