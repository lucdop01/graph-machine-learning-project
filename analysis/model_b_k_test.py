"""Does B's learned correction compensate the fixed branch's constant properties? Teacher-forced, laser-on steps.
The fixed branch (k=20, cp=600, no latent heat) is rerun on the same mesh, cotangent weights, source and explicit
substeps with real Ti-6Al-4V properties from solver/ti64.py:
  k        k(T) on each edge (mean of the two nodes' k), cp = 600, no latent heat
  k+cp     k(T) and cp(T) per node, no latent heat
  full     k(T), heat stepped in enthalpy, T from the material table (cp(T) and latent heat)
"needed" = variant - fixed, the error of the constant branch; compare it with B's correction (B output - fixed).
Per band of true start temperature, energy / laser input; then the volume-weighted node correlation and slope of the
two dT (slope 1 and r = 1: the correction is exactly the missing physics).

Usage (from the repo root):
  python -m analysis.model_b_k_test [--checkpoint results/models/model_b_seed1/best.pt]
"""
import argparse
import math

import numpy as np
import torch

from evaluation.rollout import checkpoint_predictor
from graphs.dataset import RunGraph
from solver import ti64
from solver.ti64 import CP_CONST, RHO, T_L, T_S

RUNS = ["data/graphs/seed0/val_000.h5", "data/graphs/seed0/test_path_l_turn_000.h5",
        "data/graphs/seed0/test_path_zigzag_000.h5", "data/graphs/seed0/test_path_zigzag_001.h5"]
BANDS = [("<800 K", 0, 800), ("800-1878", 800, T_S), ("melt band", T_S, T_L), (">1928", T_L, 1e9)]
MODES = ("k", "k+cp", "full")
K_MAX = float(ti64.k(np.array([T_L]))[0])
CP_MIN = float(ti64.cp(np.array([ti64.T0]))[0])


def diffuse(T, E, w, src, dst, area, dt, mode):
    """dT of one model step with the variant's properties; same operator and source split as LinearDiffusion."""
    rows = np.zeros_like(T)
    np.add.at(rows, src, np.abs(w))
    alpha_max = K_MAX / (RHO * min(CP_MIN, CP_CONST))
    n = max(1, math.ceil(float((2 * alpha_max * dt * rows / area).max()) / 0.9))
    state, H = T.copy(), ti64.H_from_T(T)
    for _ in range(n):
        k = ti64.k(state)
        flux = np.zeros_like(T)
        np.add.at(flux, src, w * 0.5 * (k[src] + k[dst]) * (state[dst] - state[src]))
        q = dt / n * flux / area + E / n                   # J/m^3 into each node this substep
        if mode == "k":
            state = state + q / (RHO * CP_CONST)
        elif mode == "k+cp":
            state = state + q / (RHO * ti64.cp(state))
        else:
            H = H + q / RHO
            state = ti64.T_from_H(H)
    return state - T


def node_energy(run, T):
    """rho V H(T) per node (J)."""
    return RHO * run.V * run.H_from_T(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/models/model_b_seed0/best.pt")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    b, stride, _ = checkpoint_predictor(a.checkpoint, a.device)
    for fn in a.runs:
        run = RunGraph(fn)
        acc = {name: np.zeros(1 + len(MODES)) for name, _, _ in BANDS}
        xy = {m: [] for m in MODES}
        L = 0.0
        for k in range(0, run.last - stride + 1, stride):
            T = run.T(k).astype(np.float64)
            E = run.laser(k, stride)
            lk = float(run.V @ E)
            if lk <= 0:
                continue
            L += lk
            with torch.no_grad():
                g = b.builder.sample(run, k, T)
                d_fixed = b.model.diffusion(g).reshape(-1).double().cpu().numpy()
            src, dst = (x.cpu().numpy() for x in g.edge_index)
            w, area = g.cotangent_weight.double().cpu().numpy(), g.area.reshape(-1).double().cpu().numpy()
            dt = float(torch.as_tensor(g.dt).reshape(-1)[0])
            Tb = b.step(run, T, k, stride)
            base = node_energy(run, T + d_fixed)
            parts = [node_energy(run, Tb) - base]
            for m in MODES:
                d_var = diffuse(T, E, w, src, dst, area, dt, m)
                parts.append(node_energy(run, T + d_var) - base)
                xy[m].append(((d_var - d_fixed), (Tb - T - d_fixed), run.V.copy()))
            for name, lo, hi in BANDS:
                msk = (T >= lo) & (T < hi)
                acc[name] += [p[msk].sum() for p in parts]
        print(f"\n{run.name}: energy / laser input, by true start temperature")
        print(f"{'band':>12s} {'B corr':>8s} " + " ".join(f"{'need ' + m:>11s}" for m in MODES))
        tot = 0
        for name, _, _ in BANDS:
            tot = tot + acc[name]
            print(f"{name:>12s} " + " ".join(f"{v / L:8.3f}" if i == 0 else f"{v / L:11.3f}"
                                              for i, v in enumerate(acc[name])))
        print(f"{'total':>12s} " + " ".join(f"{v / L:8.3f}" if i == 0 else f"{v / L:11.3f}"
                                           for i, v in enumerate(tot)))
        for m in MODES:
            x = np.concatenate([p[0] for p in xy[m]])
            y = np.concatenate([p[1] for p in xy[m]])
            v = np.concatenate([p[2] for p in xy[m]])
            xm, ym = np.average(x, weights=v), np.average(y, weights=v)
            cov = np.average((x - xm) * (y - ym), weights=v)
            var_x, var_y = np.average((x - xm) ** 2, weights=v), np.average((y - ym) ** 2, weights=v)
            print(f"  node dT: correction vs need ({m:>4s}): r = {cov / math.sqrt(var_x * var_y):+.3f}, "
                  f"slope = {cov / var_x:+.3f}")


if __name__ == "__main__":
    main()
