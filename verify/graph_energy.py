"""Check 7: the graph preprocessing keeps the solver's energy (README, evaluation notes).

For every saved frame the mesh energy, sum over nodes of rho d A_i H(T_i) with A_i the lumped node area and T_i the
node temperature, is compared with the solver energy, sum over cells of rho d dx^2 H(T_c). The difference dE is
reported in two ways:
  (a) relative to the laser energy that has arrived up to that frame, summed over the same solver steps the solver
      applied. Early frames have a tiny input, so (a) is gated only from the frame where 5% of the run's input has
      arrived. The largest value before that frame is printed but not gated.
  (b) relative to the run's total laser input, over every frame.
Laser check: for every model step of 5 and of 2 frames, the laser energy on the mesh, sum over nodes of A_i d E_i,
against the solver's input over the same solver steps.
Band check: every node whose temperature passes the solidus in some frame lies inside the fine band (graphs/mesh.py).

Usage (from the repo root):
  python -m verify.graph_energy               simulates 7 runs across the splits and paths, 8 checks (about 1.5 minutes)
  python -m verify.graph_energy --dataset     every run in data/splits.json (needs data/runs/), about 10 minutes
  options: --splits ..., --mesh-seed 1 (new-mesh test), --interp H, --processed (check the stored files in
           data/graphs/ instead of recomputing them), --csv results/graph_energy.csv
Pass: (a), (b) and the laser check below 1% for every run, and no melted node outside the fine band."""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import h5py
import numpy as np

from solver.solver import simulate
from solver.ti64 import RHO, T_S
from data.generate import COOL
from graphs.mesh import PATHS, BAND
from graphs.preprocess import (material, grid_of, mesh_for_run, node_temperature, laser_schedule, node_laser_energy,
                               grid_laser_input, load_run, processed_file, processed_problems, read_processed_header)

TOL = 0.01          # 1%, the same target as the solver's energy check
FRAC = 0.05         # (a) is gated from the frame where 5% of the run's laser input has arrived
STRIDES = (5, 2)    # windows in saved frames: 66.7 and 26.7 us (model step, E3 small step); week1 frames are 2x longer

# split, mesh, P (W), v (m/s), extra simulate() arguments, mesh seeds
CASES = [
    ("test_extrap", "straight", 195.0, 0.50, {}, (0,)),                  # largest straight melt pool
    ("train", "straight", 50.0, 1.20, {}, (0,)),                          # smallest laser input
    ("test_interp", "straight", 120.0, 0.80, {}, (0, 1)),                 # seed 1: the new-mesh test
    ("test_path", "l_turn", 170.0, 0.50, {}, (0,)),                       # melted cells farthest from the path
    ("test_path", "zigzag", 195.0, 0.65, {}, (0,)),                       # longest cooling
    ("test_fine", "straight", 150.0, 0.80, dict(nx=400, ny=200, dx=5e-6, save_every=40), (0,)),
    ("test_large", "straight_large", 195.0, 0.65, dict(nx=400, ny=200), (0,)),
]


def check(run, mesh, interp="T", Tn=None, En=None):
    """Energy, laser and band check of one run on one mesh. Tn, En: stored node data, recomputed if None."""
    HfT, _ = material(run["split"])
    nx, ny, dx, _ = grid_of(run)
    d = float(run["d"])
    e_sol = np.concatenate([RHO * d * dx ** 2 * HfT(c.astype(float)).sum(axis=(1, 2))
                            for c in np.array_split(run["T"], max(1, len(run["T"]) // 64))])
    if Tn is None:
        Tn = node_temperature(run, mesh, interp)
    Tn = np.asarray(Tn, float)
    e_mesh = RHO * d * (HfT(Tn) * mesh.area).sum(axis=1)

    sched = laser_schedule(run)
    g = grid_laser_input(run, sched)                                      # J per frame interval, solver
    cum = np.concatenate([[0.0], np.cumsum(g)])
    total = cum[-1]
    dE = e_mesh - e_sol
    late = cum >= FRAC * total
    early = (cum > 0) & ~late
    out = dict(nodes=mesh.n_nodes, frames=len(cum),
               a=np.max(np.abs(dE[late]) / cum[late]),
               a_early=np.max(np.abs(dE[early]) / cum[early]) if early.any() else np.nan,
               b=np.max(np.abs(dE)) / total,
               final=dE[-1] / total,
               solver=np.max(np.abs(e_sol - e_sol[0] - cum)) / total)  # the solver's own balance, for reference

    if En is None:
        En = node_laser_energy(run, mesh.pos, sched)
    m = d * np.asarray(En, float) @ mesh.area                             # J per frame interval, mesh
    for s in STRIDES:
        n = len(g) // s
        gs, ms = g[:n * s].reshape(n, s).sum(axis=1), m[:n * s].reshape(n, s).sum(axis=1)
        on = gs > 0
        out[f"laser{s}"] = np.max(np.abs(ms[on] - gs[on]) / gs[on])

    melted = Tn.max(axis=0) >= T_S
    out["band"] = mesh.dist[melted].max() if melted.any() and run["split"] != "week1" else np.nan
    out["ok"] = bool(out["a"] < TOL and out["b"] < TOL and out["laser5"] < TOL and out["laser2"] < TOL
                     and not out["band"] > BAND)
    return out


def line(r):
    pct = lambda x: "    -" if np.isnan(x) else f"{100 * x:5.2f}"
    band = "    -" if np.isnan(r["band"]) else f"{r['band'] * 1e6:5.1f}"
    return (f"{r['run']:28s} {r['split']:11s} {r['mesh']:20s} {r['nodes']:6d}  {pct(r['a'])}  {pct(r['a_early'])}  "
            f"{pct(r['b'])}  {100 * r['final']:+6.2f}  {pct(r['laser5'])}  {pct(r['laser2'])}  {band}  "
            f"{'PASS' if r['ok'] else 'FAIL'}")


HEADER = (f"{'run':28s} {'split':11s} {'mesh':20s} {'nodes':>6s}  {'(a)%':>5s}  {'early':>5s}  {'(b)%':>5s}  "
          f"{'end%':>6s}  {'las5%':>5s}  {'las2%':>5s}  {'band':>5s}\n"
          f"{'':28s} {'':11s} {'':20s} {'':6s}  dE/input so far, from 5% | before 5% | dE/total input | "
          f"dE at the end | laser on mesh vs solver per step | farthest melted node (um, band {BAND * 1e6:.0f})")


def summary(rows):
    for key in ("mesh", "split"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        print(f"\nworst per {key}: {'':12s} runs   (a)%   (b)%  las5%  las2%   band   pass")
        for name, g in groups.items():
            worst = lambda f: np.nanmax([r[f] for r in g]) if not all(np.isnan(r[f]) for r in g) else np.nan
            band = worst("band")
            print(f"  {name:28s} {len(g):4d}  {100 * worst('a'):5.2f}  {100 * worst('b'):5.2f}  "
                  f"{100 * worst('laser5'):5.2f}  {100 * worst('laser2'):5.2f}  "
                  f"{'    -' if np.isnan(band) else f'{band * 1e6:5.1f}'}  {sum(r['ok'] for r in g)}/{len(g)}")


def own_runs(interp):
    rows = []
    for split, mesh, P, v, kw, seeds in CASES:
        t0 = time.time()
        run = simulate(P, v, PATHS[mesh][0], **COOL, **kw)
        run.update(split=split, path=mesh.replace("_large", ""), name=f"{mesh} {P:.0f} W {v:.2f} m/s"
                   + (" 5 um" if kw.get("dx") == 5e-6 else ""))
        for seed in seeds:
            m = mesh_for_run(run, seed)
            r = check(run, m, interp)
            r.update(run=run["name"], split=split if seed == 0 else "new_mesh", mesh=f"{m.name} seed {seed}")
            rows.append(r)
            print(line(r) + f"   ({time.time() - t0:.0f} s)", flush=True)
    return rows


def dataset_runs(splits, seed, interp, processed, index="data/splits.json"):
    rows = []
    for row in json.load(open(index)):
        if splits and row["split"] not in splits:
            continue
        t0 = time.time()
        run = load_run(row["file"])
        m = mesh_for_run(run, seed)
        Tn = En = None
        if processed:
            fn = processed_file(run["name"], seed)
            problems = processed_problems(fn, *read_processed_header(fn))
            if problems:
                raise ValueError(f"{fn}: " + "; ".join(problems) + "; rerun python -m graphs.preprocess --force")
            with h5py.File(fn, "r") as f:
                if f.attrs["interp"] != interp:
                    raise ValueError(f"{run['name']}: stored with interp={f.attrs['interp']}, not {interp}")
                Tn, En = f["T"][:], f["E_frame"][:]
        r = check(run, m, interp, Tn, En)
        r.update(run=run["name"], split=row["split"], mesh=f"{m.name} seed {seed}")
        rows.append(r)
        print(line(r) + f"   ({time.time() - t0:.1f} s)", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="check every run in data/splits.json")
    ap.add_argument("--splits", nargs="+")
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--interp", choices=("T", "H"), default="T")
    ap.add_argument("--processed", action="store_true", help="check the stored node data in data/graphs/")
    ap.add_argument("--csv", help="write one row per run")
    a = ap.parse_args()
    print(f"interpolating {a.interp}; tolerance {100 * TOL:.0f}%; (a) gated from {100 * FRAC:.0f}% of the input\n")
    print(HEADER)
    rows = dataset_runs(a.splits, a.mesh_seed, a.interp, a.processed) if a.dataset else own_runs(a.interp)
    summary(rows)
    print(f"\nlargest solver energy error vs its own laser input: {100 * max(r['solver'] for r in rows):.4f}% of input")
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        keys = ["run", "split", "mesh", "nodes", "frames", "a", "a_early", "b", "final", "laser5", "laser2", "band",
                "solver", "ok"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    failed = [r["run"] + " / " + r["mesh"] for r in rows if not r["ok"]]
    print(f"{len(rows) - len(failed)}/{len(rows)} PASS" + ("" if not failed else "; FAILED: " + ", ".join(failed)))
    if failed:
        raise SystemExit(1)
