"""Where does Model B's energy surplus on zigzags come from? Teacher-forced (true T at every start frame), per step:
  laser   sum V_i E_i, the laser energy of the step
  fixed   sum rho V_i [H(T + dT_fixed) - H(T)], the fixed diffusion branch alone (constant k=20, cp=600)
  src     sum rho V_i [H(T + E_i/(rho cp)) - H(T)], the fixed branch's laser source alone, no diffusion
  full    sum rho V_i [H(T_pred) - H(T)], B's whole prediction (fixed + learned correction)
  true    sum rho V_i [H(T_k+s) - H(T_k)], the solver
All in J, summed over the steps of each third of the run and shown relative to that third's laser input.
Laser-off steps (cooling only) are skipped. Follow-ups: analysis.model_b_energy_bands, analysis.model_b_k_test.

Usage (from the repo root):
  python -m analysis.model_b_energy [--checkpoint results/models/model_b_seed1/best.pt]
"""
import argparse

import numpy as np
import torch

from evaluation.rollout import checkpoint_predictor
from graphs.dataset import RunGraph
from solver.ti64 import CP_CONST, RHO

RUNS = ["data/graphs/seed0/val_000.h5", "data/graphs/seed0/test_path_l_turn_000.h5",
        "data/graphs/seed0/test_path_zigzag_000.h5", "data/graphs/seed0/test_path_zigzag_001.h5"]


def node_energy(run, T):
    """rho V H(T) per node (J)."""
    return RHO * run.V * run.H_from_T(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/models/model_b_seed0/best.pt")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pred, stride, _ = checkpoint_predictor(a.checkpoint, a.device)
    model = pred.model
    for fn in a.runs:
        try:
            run = RunGraph(fn)
        except (FileNotFoundError, OSError):
            print(f"{fn}: missing")
            continue
        rows = []
        for k in range(0, run.last - stride + 1, stride):
            T = run.T(k).astype(np.float64)
            E = run.laser(k, stride)
            laser = float(run.V @ E)
            if laser <= 0:
                continue
            with torch.no_grad():
                g = pred.builder.sample(run, k, T)
                d_fixed = model.diffusion(g).reshape(-1).double().cpu().numpy()
            T_pred = pred.step(run, T, k, stride)
            base = node_energy(run, T).sum()
            rows.append((k, laser,
                         node_energy(run, T + d_fixed).sum() - base,
                         node_energy(run, T + E / (RHO * CP_CONST)).sum() - base,
                         node_energy(run, T_pred).sum() - base,
                         node_energy(run, run.T(k + stride).astype(np.float64)).sum() - base))
        r = np.array(rows)
        print(f"\n{run.name}: {len(r)} laser-on steps; values / laser input of each third")
        print(f"{'third':>6s} {'fixed':>8s} {'src':>8s} {'full':>8s} {'true':>8s} {'corr=full-fixed':>16s}")
        for i, part in enumerate(np.array_split(r, 3)):
            L = part[:, 1].sum()
            f, s, p, t = (part[:, j].sum() / L for j in (2, 3, 4, 5))
            print(f"{i + 1:>6d} {f:8.3f} {s:8.3f} {p:8.3f} {t:8.3f} {p - f:16.3f}")
        L = r[:, 1].sum()
        f, s, p, t = (r[:, j].sum() / L for j in (2, 3, 4, 5))
        print(f"{'all':>6s} {f:8.3f} {s:8.3f} {p:8.3f} {t:8.3f} {p - f:16.3f}")


if __name__ == "__main__":
    main()
