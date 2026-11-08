"""Report figure: one zigzag frame, truth vs. Models A-D.

Layout: true temperature (large, left) | 2 x 2 grid of error maps T_pred - T_true (shared symmetric limits) for
the models, each with the true solidus (solid lime), the predicted solidus (dashed cyan) and the energy error at this
frame relative to the laser input so far. Laser path drawn faintly on every panel, so earlier (still warm) passes
are readable. Every model is a closed-loop rollout from frame 0 (evaluation.rollout).

Usage (from the repo root):
  python -m analysis.plot_zigzag_comparison --run data/graphs/seed0/test_path_zigzag_000.h5 --list-frames
  python -m analysis.plot_zigzag_comparison --run data/graphs/seed0/test_path_zigzag_000.h5 --frame 60 --device cuda
Options: --models A B D  (drop C), --window x0 x1 y0 y1 (um), --out path.pdf
"""
import argparse
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from evaluation import metrics as M
from evaluation.rollout import checkpoint_predictor, rollout
from graphs.dataset import RunGraph
from graphs.mesh import PATHS
from solver.ti64 import T_S

UM = 1e6

# Verified: seed 0, 100k steps, augment True, lr 3e-4, noise 0.03, stride 5; stored val scores match
# the seed-0 nrmse_mean in the results tables (A 0.00364, B 0.003121, C 0.005925, D 0.006086).
CHECKPOINTS = {
    "A": "results/models/model_a_seed0/best.pt",
    "B": "results/models/model_b_seed0/best.pt",
    "C": "results/models/model_c_seed0/best.pt",
    "D": "results/models/model_d_seed0/best.pt",
}


def load_rollout(checkpoint, run, stride=5, device="cpu"):
    """Predicted T (frames, nodes) with row j = frame stride*j; row 0 is the true initial state."""
    predictor, s, _ = checkpoint_predictor(checkpoint, device, stride)
    ro = rollout(run, predictor, s)
    return np.vstack([run.T(0)[None], ro["T"]])


def laser_path(run):
    """Waypoints (m) of the run's laser path, or None if the mesh has no entry in graphs.mesh.PATHS."""
    path = PATHS.get(run.attrs["mesh"])
    return None if path is None else np.asarray(path[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="processed zigzag run (.h5)")
    ap.add_argument("--frame", type=int, help="model-step index j (solver frame = stride * j)")
    ap.add_argument("--list-frames", action="store_true", help="print j, t and the hottest true node, then exit")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="+", default=list(CHECKPOINTS))
    ap.add_argument("--window", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"),
                    help="plot window in um (default: whole plate)")
    ap.add_argument("--out", default="results/figures/zigzag_comparison.pdf")
    a = ap.parse_args()

    run = RunGraph(a.run)
    n = run.last // a.stride
    frames = a.stride * np.arange(n + 1)
    T_true_all = run.T_frames(frames)                        # row j = frame stride*j, row 0 = initial state

    if a.list_frames:
        pos = np.load(f"data/meshes/{run.attrs['mesh']}_seed{run.attrs['mesh_seed']}.npz")["pos"] * UM
        for j in range(n + 1):
            i = int(np.argmax(T_true_all[j]))
            print(f"j {j:4d}  t {run.t[frames[j]] * 1e3:6.3f} ms  hottest node ({pos[i, 0]:6.0f}, {pos[i, 1]:6.0f}) um"
                  f"  T_max {T_true_all[j].max():6.0f} K")
        return
    if a.frame is None:
        raise SystemExit("give --frame (use --list-frames to find pass 3)")
    j = a.frame

    with np.load(f"data/meshes/{run.attrs['mesh']}_seed{run.attrs['mesh_seed']}.npz") as m:
        pos, tri_idx = m["pos"], m["tri"]
        Lx, Ly = float(m["Lx"]), float(m["Ly"])
    tri = mtri.Triangulation(pos[:, 0] * UM, pos[:, 1] * UM, tri_idx)

    T_true = T_true_all[j]
    preds, errs, dE = {}, {}, {}
    d = float(run.attrs["d"])
    cum = run.laser_input()
    e_true = M.mesh_energy(T_true[None], run.H_from_T, run.mesh.area, d)[0]
    for name in a.models:
        T_hat_all = load_rollout(CHECKPOINTS[name], run, a.stride, a.device)
        if T_hat_all.shape != T_true_all.shape:
            raise ValueError(f"{name}: predicted {T_hat_all.shape} vs true {T_true_all.shape}")
        preds[name] = T_hat_all[j]
        errs[name] = preds[name] - T_true
        e_hat = M.mesh_energy(preds[name][None], run.H_from_T, run.mesh.area, d)[0]
        dE[name] = (e_hat - e_true) / cum[frames[j]]          # relative to laser input so far

    t_lo, t_hi = 300.0, float(np.percentile(T_true, 99.9))
    e_lim = float(np.percentile(np.abs(np.concatenate(list(errs.values()))), 99.5))
    if a.window:
        xlim, ylim = (a.window[0], a.window[1]), (a.window[2], a.window[3])
    else:                                                   # tight box around the warm region, 60 um padding
        warm = T_true > 400.0
        x, y = pos[warm, 0] * UM, pos[warm, 1] * UM
        xlim = (max(0, x.min() - 60), min(Lx * UM, x.max() + 60))
        ylim = (max(0, y.min() - 60), min(Ly * UM, y.max() + 60))
    r = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])           # window aspect, kept exact (no distortion)
    wp = laser_path(run)

    # Fixed layout in inches, sized to the NeurIPS text width (5.5 in), so fonts print at true size.
    # [ truth (2 rows) | T bar ]  [ A | B ]  [ err bar ]
    #                             [ C | D ]
    FW, g, cb, lab = 5.5, 0.05, 0.07, 0.5                   # width, gap, colorbar width, room for bar labels
    ncol = 2
    nrow = int(np.ceil(len(a.models) / ncol))
    # truth height = nrow*H + (nrow-1)*g, truth width = r * that; models: W = r*H
    # FW = r*(nrow*H+(nrow-1)*g) + g + cb + lab + ncol*r*H + (ncol-1)*g + g + cb + lab
    H = (FW - r * (nrow - 1) * g - 2 * (g + cb + lab) - (ncol - 1) * g) / (r * (nrow + ncol))
    W = r * H
    FH = nrow * H + (nrow - 1) * g + 0.02
    fig = plt.figure(figsize=(FW, FH))

    def axes_in(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    def decorate(ax, label, T_pred=None):
        if wp is not None:
            ax.plot(wp[:, 0] * UM, wp[:, 1] * UM, color="0.7", lw=0.4, ls=":")
        if T_true.max() >= T_S:
            ax.tricontour(tri, T_true, levels=[T_S], colors="lime", linewidths=0.6)
        if T_pred is not None and T_pred.max() >= T_S:
            ax.tricontour(tri, T_pred, levels=[T_S], colors="cyan", linewidths=0.6, linestyles="--")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.96, label, transform=ax.transAxes, fontsize=7, va="top", ha="left",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})

    th = nrow * H + (nrow - 1) * g
    ax = axes_in(0, 0, r * th, th)
    # Fills are rasterized: vector gouraud triangles leave speckles in some PDF viewers. Text and lines stay vector.
    im_t = ax.tripcolor(tri, T_true, cmap="inferno", vmin=t_lo, vmax=t_hi, shading="gouraud", rasterized=True)
    decorate(ax, f"Truth, t = {run.t[frames[j]] * 1e3:.2f} ms")
    x = r * th + g
    cb1 = fig.colorbar(im_t, cax=axes_in(x, 0, cb, th))
    cb1.set_label("T (K)", fontsize=7)
    x0 = x + cb + lab

    for i, name in enumerate(a.models):
        rr, cc = divmod(i, ncol)
        ax = axes_in(x0 + cc * (W + g), (nrow - 1 - rr) * (H + g), W, H)
        im_e = ax.tripcolor(tri, errs[name], cmap="RdBu_r", vmin=-e_lim, vmax=e_lim, shading="gouraud",
                            rasterized=True)
        decorate(ax, name, preds[name])
        ax.text(0.98, 0.04, f"$\\Delta E$ = {100 * dE[name]:+.1f}%", transform=ax.transAxes, fontsize=6,
                ha="right", bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    cb2 = fig.colorbar(im_e, cax=axes_in(x0 + ncol * W + (ncol - 1) * g + g, 0, cb, th))
    cb2.set_label(r"$T_\mathrm{pred} - T_\mathrm{true}$ (K)", fontsize=7)
    for c in (cb1, cb2):
        c.ax.tick_params(labelsize=6)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=300)              # dpi sets the rasterized fills
    print(f"Saved {a.out}  (j {j}, frame {frames[j]}, error limit +/-{e_lim:.0f} K, "
          + ", ".join(f"{k} dE {100 * v:+.2f}%" for k, v in dE.items()) + ")")


if __name__ == "__main__":
    main()
