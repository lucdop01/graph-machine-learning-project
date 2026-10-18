import os
import argparse
import time

# Ensure writable matplotlib cache dir
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
torch.set_num_threads(min(8, os.cpu_count() or 4))

from solver.ti64 import T_S, T_L
from graphs.mesh import BAND
from graphs.figures import band_outline, _style
from evaluation.rollout import RunGraph, checkpoint_predictor


def rollout_until(run, predictor, stride: int, target_step: int):
    """Roll out only until target_step (1-indexed) to avoid unnecessary computation."""
    if hasattr(predictor, "reset"):
        predictor.reset(run)
    T = run.T(0)
    for j in range(target_step):
        k = j * stride
        t0 = time.perf_counter()
        T = np.asarray(predictor.step(run, T, k, stride), dtype=np.float64)
    frame_k = target_step * stride
    return frame_k, T


def visualize_comparison(
    run_file: str,
    ckpt_b: str,
    ckpt_d: str,
    output_path: str = "results/figures/heat_prediction_comparison.png",
    device: str = "cpu",
    step_idx: int = 10,
):
    print(f"Loading run {run_file}...")
    run = RunGraph(run_file)
    stride = 5

    print(f"Loading Model B ({ckpt_b})...")
    pred_b, _, _ = checkpoint_predictor(ckpt_b, device=device, stride=stride)
    
    print(f"Loading Model D ({ckpt_d})...")
    pred_d, _, _ = checkpoint_predictor(ckpt_d, device=device, stride=stride)

    print(f"Executing Model B rollout up to step {step_idx}...")
    frame_k, T_pred_b = rollout_until(run, pred_b, stride=stride, target_step=step_idx)

    print(f"Executing Model D rollout up to step {step_idx}...")
    _, T_pred_d = rollout_until(run, pred_d, stride=stride, target_step=step_idx)
    t_ms = float(run.t[frame_k]) * 1e3
    T_true = run.T(frame_k)

    err_b = T_pred_b - T_true
    err_d = T_pred_d - T_true

    mesh = run.mesh
    UM = 1e6
    pos_x = mesh.pos[:, 0] * UM
    pos_y = mesh.pos[:, 1] * UM
    tri = mtri.Triangulation(pos_x, pos_y, mesh.tri)

    # Laser position at this frame
    xl, yl = run.laser[frame_k] * UM
    zoom_x = (max(0, xl - 450), min(mesh.Lx * UM, xl + 350))
    zoom_y = (max(0, yl - 200), min(mesh.Ly * UM, yl + 200))

    vmin, vmax = 300.0, float(max(T_true.max(), T_pred_b.max(), T_pred_d.max()))
    err_max = float(max(np.percentile(np.abs(err_b), 99.5), np.percentile(np.abs(err_d), 99.5), 15.0))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)

    # 1. Ground Truth
    ax_gt = axes[0, 0]
    im_gt = ax_gt.tripcolor(tri, T_true, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_gt.tricontour(tri, T_true, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_gt.plot([xl], [yl], marker="+", color="white", ms=10, mew=1.5, label="Laser")
    band_outline(ax_gt, run.waypoints, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
    ax_gt.set_xlim(*zoom_x)
    ax_gt.set_ylim(*zoom_y)
    ax_gt.set_title(f"(A) Ground Truth Thermal Field (t = {t_ms:.2f} ms)", fontsize=11, fontweight="bold")
    _style(ax_gt)

    # 2. Centerline 1D Profile (x-slice through track)
    ax_prof = axes[0, 1]
    track_y = yl
    near_track = np.abs(pos_y - track_y) < 15.0  # nodes within 15 um of laser line
    sort_idx = np.argsort(pos_x[near_track])
    xs = pos_x[near_track][sort_idx]
    ax_prof.plot(xs, T_true[near_track][sort_idx], "k-", lw=2.0, label="Ground Truth")
    ax_prof.plot(xs, T_pred_b[near_track][sort_idx], "r--", lw=1.5, label="Model B (Diffusion Res.)")
    ax_prof.plot(xs, T_pred_d[near_track][sort_idx], "b-.", lw=1.5, label="Model D (Cons. Flux)")
    ax_prof.axhline(T_S, color="#00a8a8", ls=":", lw=1.0, label=f"Solidus ({T_S:.0f} K)")
    ax_prof.axhline(T_L, color="#888888", ls=":", lw=1.0, label=f"Liquidus ({T_L:.0f} K)")
    ax_prof.set_xlim(*zoom_x)
    ax_prof.set_xlabel("x (um)", fontsize=9)
    ax_prof.set_ylabel("Temperature (K)", fontsize=9)
    ax_prof.set_title("(B) Laser Track Centerline Temperature Profile", fontsize=11, fontweight="bold")
    ax_prof.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_prof.grid(True, alpha=0.3, ls=":")

    # 3. Model B Prediction
    ax_b = axes[1, 0]
    ax_b.tripcolor(tri, T_pred_b, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_b.tricontour(tri, T_pred_b, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_b.plot([xl], [yl], marker="+", color="white", ms=10, mew=1.5)
    band_outline(ax_b, run.waypoints, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
    ax_b.set_xlim(*zoom_x)
    ax_b.set_ylim(*zoom_y)
    ax_b.set_title("(C) Model B Prediction (Physics-Guided Residual)", fontsize=11, fontweight="bold")
    _style(ax_b)

    # 4. Model B Error
    ax_err_b = axes[1, 1]
    im_err_b = ax_err_b.tripcolor(tri, err_b, cmap="bwr", vmin=-err_max, vmax=err_max, shading="gouraud")
    band_outline(ax_err_b, run.waypoints, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
    ax_err_b.set_xlim(*zoom_x)
    ax_err_b.set_ylim(*zoom_y)
    ax_err_b.set_title(f"(D) Model B Error Map (max |err| = {np.max(np.abs(err_b)):.1f} K)", fontsize=11, fontweight="bold")
    _style(ax_err_b)

    # 5. Model D Prediction
    ax_d = axes[2, 0]
    ax_d.tripcolor(tri, T_pred_d, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_d.tricontour(tri, T_pred_d, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_d.plot([xl], [yl], marker="+", color="white", ms=10, mew=1.5)
    band_outline(ax_d, run.waypoints, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
    ax_d.set_xlim(*zoom_x)
    ax_d.set_ylim(*zoom_y)
    ax_d.set_title(f"(E) Model D Prediction (Strict Heat Conservation)", fontsize=11, fontweight="bold")
    _style(ax_d)

    # 6. Model D Error
    ax_err_d = axes[2, 1]
    im_err_d = ax_err_d.tripcolor(tri, err_d, cmap="bwr", vmin=-err_max, vmax=err_max, shading="gouraud")
    band_outline(ax_err_d, run.waypoints, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
    ax_err_d.set_xlim(*zoom_x)
    ax_err_d.set_ylim(*zoom_y)
    ax_err_d.set_title(f"(F) Model D Error Map (max |err| = {np.max(np.abs(err_d)):.1f} K)", fontsize=11, fontweight="bold")
    _style(ax_err_d)

    # Colorbars
    cb_temp = fig.colorbar(im_gt, ax=axes[:, 0], orientation="horizontal", fraction=0.03, pad=0.05)
    cb_temp.set_label("Temperature T (K)", fontsize=10)

    cb_err = fig.colorbar(im_err_b, ax=axes[1:, 1], orientation="horizontal", fraction=0.04, pad=0.07)
    cb_err.set_label("Temperature Error: Prediction - Truth (K)", fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="data/graphs/seed0/val_000.h5")
    parser.add_argument("--ckpt-b", default="checkpoints/model_b_seed0/best.pt")
    parser.add_argument("--ckpt-d", default="checkpoints/model_d_seed0/best.pt")
    parser.add_argument("--out", default="results/figures/heat_prediction_comparison.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step", type=int, default=12)
    args = parser.parse_args()

    visualize_comparison(args.run, args.ckpt_b, args.ckpt_d, args.out, args.device, args.step)

