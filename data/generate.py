"""Generate every simulation run and the splits.
Usage (from repo root):
  python -m data.generate --only week1     # 50 constant-property runs for week 1 (a few minutes)
  python -m data.generate                  # everything, under an hour on a laptop
  python -m data.generate --limit 1        # one run per split, smoke test"""
import argparse, json, os
import h5py
import numpy as np
from scipy.stats import qmc
from solver.solver import simulate
from solver.paths import straight, l_turn, zigzag
from solver.ti64 import CONST_PROPS

P_RANGE, V_RANGE = (50.0, 195.0), (0.5, 1.2)      # W, m/s (range of doi:10.3390/ma18020368, Table 4)
HOT = dict(P=(170.0, 195.0), v=(0.5, 0.65))       # highest energy input: held out for extrapolation
T_BOIL = 3660.0                                    # K; above this the conduction model is invalid
OUT = "data/runs"
SPLITS = ("week1", "train", "val", "test_interp", "test_extrap", "test_path", "test_fine", "test_large")

def lhs(n, P_rng, v_rng, seed):
    u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    return [(P_rng[0] + a * (P_rng[1] - P_rng[0]), v_rng[0] + b * (v_rng[1] - v_rng[0])) for a, b in u]

def in_hot(P, v):
    return HOT["P"][0] < P and v < HOT["v"][1]

def save(name, split, path_name, r):
    fn = f"{OUT}/{name}.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("T", data=r["T"], compression="gzip", compression_opts=4)
        for key in ("t", "laser", "waypoints"):
            f.create_dataset(key, data=r[key])
        for key in ("P", "v", "dx", "d", "dt", "laser_off", "T_max"):
            f.attrs[key] = r[key]
        f.attrs["path"], f.attrs["split"] = path_name, split
    row = dict(file=fn, split=split, path=path_name, P=float(r["P"]), v=float(r["v"]), dx=r["dx"], Tmax=r["T_max"])
    print(row)
    return row

def run_until(n, candidates, split, path_name, sim, name_fmt, check_boil=True):
    """Simulate candidates in order, skip any that reach boiling, keep the first n."""
    rows = []
    for P, v in candidates:
        if len(rows) == n:
            break
        r = sim(P, v)
        if check_boil and r["T_max"] >= T_BOIL:
            print(f"rejected {split} P={P:.0f} W v={v:.2f} m/s: T_max={r['T_max']:.0f} K")
            continue
        rows.append(save(name_fmt.format(len(rows)), split, path_name, r))
    if len(rows) < n:
        print(f"WARNING: only {len(rows)} of {n} runs kept for {split}")
    return rows

def main(limit=None, only=None):
    os.makedirs(OUT, exist_ok=True)
    want = (lambda s: only is None or s in only)
    cap = (lambda n: min(n, limit) if limit else n)
    rng = np.random.default_rng(0)
    interior = [p for p in lhs(260, P_RANGE, V_RANGE, seed=0) if not in_hot(*p)]
    rng.shuffle(interior)
    pools = dict(train=interior[:150], val=interior[150:170], test_interp=interior[170:200], spare=interior[200:])
    std = lambda P, v: simulate(P, v, straight())
    index = []
    if want("week1"):
        index += run_until(cap(50), lhs(50, P_RANGE, V_RANGE, seed=5), "week1", "straight",
                           lambda P, v: simulate(P, v, straight(), props=CONST_PROPS), "week1_{:03d}", check_boil=False)
    for split, n in (("train", 150), ("val", 20), ("test_interp", 30)):
        if want(split):   # spares replace any rejected run so the split keeps its size
            index += run_until(cap(n), pools[split] + pools["spare"], split, "straight", std, split + "_{:03d}")
    if want("test_extrap"):
        index += run_until(cap(15), lhs(30, HOT["P"], HOT["v"], seed=1), "test_extrap", "straight", std, "test_extrap_{:03d}")
    if want("test_path"):
        for path_name, wp, seed in (("l_turn", l_turn(), 2), ("zigzag", zigzag(), 3)):
            cands = [p for p in lhs(40, P_RANGE, V_RANGE, seed=seed) if not in_hot(*p)]
            index += run_until(cap(10), cands, "test_path", path_name,
                               lambda P, v, wp=wp: simulate(P, v, wp), f"test_path_{path_name}_" + "{:03d}")
    if want("test_fine"):   # test_interp settings on a 2x finer grid (dt is exactly 4x smaller)
        index += run_until(cap(30), pools["test_interp"], "test_fine", "straight",
                           lambda P, v: simulate(P, v, straight(), nx=400, ny=200, dx=5e-6, save_every=40),
                           "test_fine_{:03d}")
    if want("test_large"):  # 4 x 2 mm plate, 2.4 mm track, same 10 um grid
        cands = [p for p in lhs(30, P_RANGE, V_RANGE, seed=4) if not in_hot(*p)]
        index += run_until(cap(10), cands, "test_large", "straight",
                           lambda P, v: simulate(P, v, straight(4e-3, 2e-3), nx=400, ny=200), "test_large_{:03d}")
    fn = "data/splits.json"
    old = json.load(open(fn)) if os.path.exists(fn) else []
    keep = [row for row in old if row["split"] not in {r["split"] for r in index}]
    json.dump(keep + index, open(fn, "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="+", choices=SPLITS)
    a = ap.parse_args()
    main(a.limit, a.only)
