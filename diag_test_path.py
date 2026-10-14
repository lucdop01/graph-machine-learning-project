"""Diagnostic, not part of the pipeline (delete after use): does a model deposit too little heat when the laser moves
in a direction other than +x (the only direction in the train split)?

For each run, every model step is predicted twice:
  one-step  from the TRUE state at frame k, so no error carried over from earlier steps
  rollout   from the model's own previous output, as in evaluation.rollout
Per step it prints the laser centroid and direction, the laser input of the step Q (J), the true energy change over Q
(about 1: the plate has no heat loss), the one-step predicted energy change over Q (1 is right, below 1 is heat
lost), the one-step nRMSE, and the rollout energy error relative to the laser input so far. Then the one-step
energy ratio and nRMSE per laser direction (+x, +y, -x, -y). Writes every row to --out.

Usage (repo root):
  python diag_test_path.py --checkpoint checkpoints/a_seed0/best.pt --device cuda
  python diag_test_path.py --checkpoint ... --runs test_interp:test_interp_000 test_path:test_path_l_turn_003
"""
import argparse
import csv
import math

import numpy as np

from evaluation import metrics as M
from evaluation.rollout import checkpoint_predictor, rollout
from graphs.dataset import load_runs

DEFAULT_RUNS = ["test_interp:test_interp_000", "test_path:test_path_l_turn_000", "test_path:test_path_zigzag_000"]


def direction(angle):
    if -45 < angle <= 45:
        return "+x"
    if 45 < angle <= 135:
        return "+y"
    if -135 < angle <= -45:
        return "-y"
    return "-x"


def energy(run, T):
    return float(M.mesh_energy(T, run.H_from_T, run.mesh.area, float(run.attrs["d"])))


def diagnose(run, predictor, stride):
    if hasattr(predictor, "reset"):
        predictor.reset(run)
    ro = rollout(run, predictor, stride)
    cum = run.laser_input()
    rows, prev = [], None
    for j, f in enumerate(ro["frames"]):
        k = f - stride
        q = run.laser(k, stride) * run.V                      # J per node in this step
        Q = float(q.sum())
        T_k, T_next = run.T(k), run.T(f)
        T_hat = np.asarray(predictor.step(run, T_k, k, stride), dtype=np.float64)
        e_k, e_next = energy(run, T_k), energy(run, T_next)
        rise = T_next - 300.0
        row = dict(run=run.name, step=j + 1, frame=int(f), t_us=float(run.t[f] * 1e6), Q_J=Q,
                   true_dE_over_Q=(e_next - e_k) / Q if Q > 0 else math.nan,
                   onestep_dE_over_Q=(energy(run, T_hat) - e_k) / Q if Q > 0 else math.nan,
                   onestep_nrmse=float(np.sqrt(np.mean((T_hat - T_next) ** 2) / max(np.mean(rise ** 2), 1e-30))),
                   rollout_energy_err=(energy(run, ro["T"][j]) - e_next) / cum[f] if cum[f] > 0 else math.nan,
                   x_mm=math.nan, y_mm=math.nan, angle_deg=math.nan, dir="")
        if Q > 0:
            c = (q[:, None] * run.mesh.pos).sum(axis=0) / Q
            row["x_mm"], row["y_mm"] = c[0] * 1e3, c[1] * 1e3
            if prev is not None and np.hypot(*(c - prev)) > 1e-6:
                row["angle_deg"] = math.degrees(math.atan2(c[1] - prev[1], c[0] - prev[0]))
                row["dir"] = direction(row["angle_deg"])
            prev = c
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="split:run_name")
    ap.add_argument("--out", default="diag_test_path.csv")
    a = ap.parse_args()
    predictor, stride, _ = checkpoint_predictor(a.checkpoint, a.device)
    split_runs, everything = {}, []
    for spec in a.runs:
        split, name = spec.split(":")
        if split not in split_runs:
            split_runs[split] = {r.name: r for r in load_runs(split, 0)}
        rows = diagnose(split_runs[split][name], predictor, stride)
        everything += rows
        print(f"\n{name}")
        print(f"{'step':>4} {'t_us':>7} {'x_mm':>6} {'y_mm':>6} {'dir':>3} {'Q_mJ':>7} {'true/Q':>7} {'pred/Q':>7} "
              f"{'1-step nRMSE':>12} {'rollout E err':>13}")
        for r in rows:
            print(f"{r['step']:>4} {r['t_us']:7.1f} {r['x_mm']:6.3f} {r['y_mm']:6.3f} {r['dir']:>3} "
                  f"{r['Q_J'] * 1e3:7.3f} {r['true_dE_over_Q']:7.3f} {r['onestep_dE_over_Q']:7.3f} "
                  f"{r['onestep_nrmse']:12.4f} {r['rollout_energy_err'] * 100:12.2f}%")
        print("by laser direction (steps with the laser on; Q-weighted one-step energy ratio, mean one-step nRMSE):")
        for d in ("+x", "+y", "-x", "-y"):
            sel = [r for r in rows if r["dir"] == d]
            if sel:
                ratio = sum(r["onestep_dE_over_Q"] * r["Q_J"] for r in sel) / sum(r["Q_J"] for r in sel)
                print(f"  {d}: {len(sel):3d} steps, pred/Q {ratio:.3f}, true/Q "
                      f"{sum(r['true_dE_over_Q'] * r['Q_J'] for r in sel) / sum(r['Q_J'] for r in sel):.3f}, "
                      f"nRMSE {np.mean([r['onestep_nrmse'] for r in sel]):.4f}")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(everything[0]))
        w.writeheader()
        w.writerows(everything)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
