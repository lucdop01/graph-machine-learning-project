"""Generate every simulation run and the splits.
Usage (from repo root):
  python -m data.generate --only week1     # 50 constant-property runs for week 1 (a few minutes)
  python -m data.generate                  # everything, under an hour on a laptop
  python -m data.generate --limit 1        # one run per split, smoke test
Every run except week1 cools until the whole plate is below the solidus, for at least T_COOL_MIN and at most
T_COOL_MAX. A run that reaches T_COOL_MAX with liquid left is kept but flagged (complete = false)."""
import argparse
import json
import os
import h5py
import numpy as np
from scipy.stats import qmc
from solver.solver import simulate
from solver.paths import straight, l_turn, zigzag
from solver.ti64 import CONST_PROPS

P_RANGE, V_RANGE = (50.0, 195.0), (0.5, 1.2)      # W, m/s (range of doi:10.3390/ma18020368, Table 4)
HOT = dict(P=(170.0, 195.0), v=(0.5, 0.65))       # highest energy input: held out for extrapolation
T_BOIL = 3660.0                                    # K; above this the conduction model is invalid
T_COOL_MIN, T_COOL_MAX = 0.2e-3, 5e-3              # s; slowest run below boiling needs 1.9 ms (verify/cooling.py)
COOL = dict(t_cool=T_COOL_MIN, t_cool_max=T_COOL_MAX)
OUT = "data/runs"
SPLITS = ("week1", "train", "val", "test_interp", "test_extrap", "test_path", "test_fine", "test_large")

def lhs(n, P_rng, v_rng, seed):
    u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    return [(P_rng[0] + a * (P_rng[1] - P_rng[0]), v_rng[0] + b * (v_rng[1] - v_rng[0])) for a, b in u]

def in_hot(P, v):
    return HOT["P"][0] < P and v < HOT["v"][1]

def save(name, split, path_name, r, cooled=True):
    fn = f"{OUT}/{name}.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("T", data=r["T"], compression="gzip", compression_opts=4)
        for key in ("t", "laser", "waypoints"):
            f.create_dataset(key, data=r[key])
        for key in ("P", "v", "dx", "d", "dt", "laser_off", "T_max", "n_steps", "t_end", "t_cool_actual",
                    "terminal_frame", "terminal_saved") + (("cooling_complete",) if cooled else ()):
            f.attrs[key] = r[key]
        f.attrs["path"], f.attrs["split"] = path_name, split
    row = dict(file=fn, split=split, path=path_name, P=float(r["P"]), v=float(r["v"]), dx=r["dx"], Tmax=r["T_max"],
               complete=r["cooling_complete"] if cooled else None, t_cool=float(r["t_cool_actual"]), frames=len(r["t"]))
    print(row)
    return row

def run_until(n, candidates, split, path_name, sim, name_fmt, check_boil=True, cooled=True):
    """Simulate candidates in order, skip any that reach boiling, keep the first n.
    cooled: the run cools until solid (COOL). Runs still liquid at T_COOL_MAX are kept and flagged, not replaced.
    week1 uses a fixed 0.2 ms and constant properties without melting, so it gets no completion flag."""
    rows = []
    for P, v in candidates:
        if len(rows) == n:
            break
        r = sim(P, v)
        if check_boil and r["T_max"] >= T_BOIL:
            print(f"rejected {split} P={P:.0f} W v={v:.2f} m/s: T_max={r['T_max']:.0f} K")
            continue
        if cooled and not r["cooling_complete"]:
            print(f"INCOMPLETE {split} P={P:.0f} W v={v:.2f} m/s: liquid left after {r['t_cool_actual']*1e3:.2f} ms of cooling")
        rows.append(save(name_fmt.format(len(rows)), split, path_name, r, cooled))
    if len(rows) < n:
        print(f"WARNING: only {len(rows)} of {n} runs kept for {split}")
    return rows

def main(limit=None, only=None):
    os.makedirs(OUT, exist_ok=True)
    def want(s):
        return only is None or s in only

    def cap(n):
        return min(n, limit) if limit else n

    rng = np.random.default_rng(0)
    interior = [p for p in lhs(260, P_RANGE, V_RANGE, seed=0) if not in_hot(*p)]
    rng.shuffle(interior)
    pools = dict(train=interior[:150], val=interior[150:170], test_interp=interior[170:200])
    # replacements for boiled runs; disjoint per split so no setting can appear in two splits
    spares = dict(train=interior[200:225], val=interior[225:235], test_interp=interior[235:])

    def std(P, v):
        return simulate(P, v, straight(), **COOL)
    index = []
    if want("week1"):
        index += run_until(cap(50), lhs(50, P_RANGE, V_RANGE, seed=5), "week1", "straight",
                           lambda P, v: simulate(P, v, straight(), props=CONST_PROPS), "week1_{:03d}",
                           check_boil=False, cooled=False)
    for split, n in (("train", 150), ("val", 20), ("test_interp", 30)):
        if want(split):   # spares replace any rejected run so the split keeps its size
            index += run_until(cap(n), pools[split] + spares[split], split, "straight", std, split + "_{:03d}")
    if want("test_extrap"):
        index += run_until(cap(15), lhs(30, HOT["P"], HOT["v"], seed=1), "test_extrap", "straight", std, "test_extrap_{:03d}")
    if want("test_path"):
        for path_name, wp, seed in (("l_turn", l_turn(), 2), ("zigzag", zigzag(), 3)):
            cands = [p for p in lhs(40, P_RANGE, V_RANGE, seed=seed) if not in_hot(*p)]
            index += run_until(cap(10), cands, "test_path", path_name,
                               lambda P, v, wp=wp: simulate(P, v, wp, **COOL), f"test_path_{path_name}_" + "{:03d}")
    if want("test_fine"):   # test_interp settings on a 2x finer grid (dt is exactly 4x smaller)
        index += run_until(cap(30), pools["test_interp"] + spares["test_interp"], "test_fine", "straight",
                           lambda P, v: simulate(P, v, straight(), nx=400, ny=200, dx=5e-6, save_every=40, **COOL),
                           "test_fine_{:03d}")
    if want("test_large"):  # 4 x 2 mm plate, 2.4 mm track, same 10 um grid
        cands = [p for p in lhs(30, P_RANGE, V_RANGE, seed=4) if not in_hot(*p)]
        index += run_until(cap(10), cands, "test_large", "straight",
                           lambda P, v: simulate(P, v, straight(4e-3, 2e-3), nx=400, ny=200, **COOL), "test_large_{:03d}")
    fn = "data/splits.json"
    old = json.load(open(fn)) if os.path.exists(fn) else []
    keep = [row for row in old if row["split"] not in {r["split"] for r in index}]
    json.dump(keep + index, open(fn, "w"), indent=1)
    bad = [row["file"] for row in index if row["complete"] is False]
    print(f"{len(index)} runs written, {len(bad)} flagged incomplete" + (": " + ", ".join(bad) if bad else ""))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="+", choices=SPLITS)
    a = ap.parse_args()
    main(a.limit, a.only)
