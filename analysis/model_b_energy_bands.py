"""Follow-up to analysis.model_b_energy: in which temperature bands does B's learned correction add or remove energy,
and what does A do on the same steps? Teacher-forced, laser-on steps; node energy changes are grouped by the true
temperature at the start of the step and shown relative to the laser input:
  B corr   rho V [H(T_B) - H(T + dT_fixed)], the learned correction alone
  B full   rho V [H(T_B) - H(T)], B's whole prediction
  A full   rho V [H(T_A) - H(T)], Model A's prediction
  true     rho V [H(T_k+s) - H(T_k)], the solver

Usage (from the repo root):
  python -m analysis.model_b_energy_bands [--checkpoint results/models/model_b_seed1/best.pt]
"""
import argparse

import numpy as np
import torch

from evaluation.rollout import checkpoint_predictor
from graphs.dataset import RunGraph
from solver.ti64 import RHO, T_L, T_S

RUNS = ["data/graphs/seed0/val_000.h5", "data/graphs/seed0/test_path_zigzag_000.h5",
        "data/graphs/seed0/test_path_zigzag_001.h5"]
BANDS = [("<800 K", 0, 800), ("800-1878", 800, T_S), ("melt band", T_S, T_L), (">1928", T_L, 1e9)]


def node_energy(run, T):
    """rho V H(T) per node (J)."""
    return RHO * run.V * run.H_from_T(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/models/model_b_seed0/best.pt")
    ap.add_argument("--compare", default="results/models/model_a_seed0/best.pt", help="Model A checkpoint")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    b, stride, _ = checkpoint_predictor(a.checkpoint, a.device)
    m_a, _, _ = checkpoint_predictor(a.compare, a.device)
    for fn in a.runs:
        run = RunGraph(fn)
        acc = {name: np.zeros(4) for name, _, _ in BANDS}        # correction, B full, A full, true
        L = 0.0
        for k in range(0, run.last - stride + 1, stride):
            T = run.T(k).astype(np.float64)
            E = run.laser(k, stride)
            lk = float(run.V @ E)
            if lk <= 0:
                continue
            L += lk
            with torch.no_grad():
                d_fixed = b.model.diffusion(b.builder.sample(run, k, T)).reshape(-1).double().cpu().numpy()
            Tb, Ta = b.step(run, T, k, stride), m_a.step(run, T, k, stride)
            Tt = run.T(k + stride).astype(np.float64)
            h0 = node_energy(run, T)
            hb = node_energy(run, Tb)
            parts = (hb - node_energy(run, T + d_fixed), hb - h0, node_energy(run, Ta) - h0,
                     node_energy(run, Tt) - h0)
            for name, lo, hi in BANDS:
                m = (T >= lo) & (T < hi)
                acc[name] += [p[m].sum() for p in parts]
        print(f"\n{run.name}: per band, / laser input")
        print(f"{'band (true T at start)':>24s} {'B corr':>8s} {'B full':>8s} {'A full':>8s} {'true':>8s}")
        tot = np.zeros(4)
        for name, _, _ in BANDS:
            tot += acc[name]
            print(f"{name:>24s} " + " ".join(f"{v / L:8.3f}" for v in acc[name]))
        print(f"{'total':>24s} " + " ".join(f"{v / L:8.3f}" for v in tot))


if __name__ == "__main__":
    main()
