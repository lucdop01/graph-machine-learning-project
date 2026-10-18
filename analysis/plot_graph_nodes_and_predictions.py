"""Generate comprehensive visualization highlighting graph nodes, mesh edges, and cross-model predictions.

Demonstrates:
1. Graph Discretization: Fine-band (10 um) vs Coarse (50 um) node density with message-passing Delaunay edges.
2. Node-Level Thermal State: Discrete node scatter colored by temperature at the melt pool.
3. Cross-Model Comparison: Ground Truth vs Model B vs Model D thermal fields and absolute error maps.
4. Centerline Profile: Temperature along the laser track displaying exact graph node positions.
"""
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import torch

from solver.ti64 import T_S, T_L
from graphs.mesh import BAND
from graphs.figures import band_outline, _style
from evaluation.rollout import RunGraph, checkpoint_predictor


def run_rollout(run, predictor, stride: int, target_step: int):
    if hasattr(predictor, "reset"):
        predictor.reset(run)
    T = run.T(0)
    for j in range(target_step):
        k = j * stride
        T = np.asarray(predictor.step(run, T, k, stride), dtype=np.float64)
    frame_k = target_step * stride
    return frame_k, T


def generate_node_level_comparison(
    run_path: str = "data/graphs/seed0/val_000.h5",
    ckpt_b: str = "checkpoints/model_b_seed0/best.pt",
    ckpt_d: str = "checkpoints/model_d_seed0/best.pt",
    out_path: str = "results/figures/graph_nodes_heat_comparison.png",
    step_idx: int = 12,
    device: str = "cpu",
):
    print(f"Loading run: {run_path}...")
    run = RunGraph(run_path)
    stride = 5

    print(f"Loading Model B ({ckpt_b})...")
    pred_b, _, _ = checkpoint_predictor(ckpt_b, device=device, stride=stride)

    print(f"Loading Model D ({ckpt_d})...")
    pred_d, _, _ = checkpoint_predictor(ckpt_d, device=device, stride=stride)

    print(f"Computing rollouts to step {step_idx} (frame {step_idx * stride})...")
    frame_k, T_b = run_rollout(run, pred_b, stride, step_idx)
    _, T_d = run_rollout(run, pred_d, stride, step_idx)

    T_gt = run.T(frame_k)
    t_ms = float(run.t[frame_k]) * 1e3

    err_b = np.abs(T_b - T_gt)
    err_d = np.abs(T_d - T_gt)

    mesh = run.mesh
    UM = 1e6
    pos_x = mesh.pos[:, 0] * UM
    pos_y = mesh.pos[:, 1] * UM
    tri = mtri.Triangulation(pos_x, pos_y, mesh.tri)

    raw_fn = os.path.join("data/runs", os.path.basename(run_path))
    if os.path.exists(raw_fn):
        import h5py
        with h5py.File(raw_fn, "r") as f:
            xl, yl = f["laser"][frame_k] * UM
    else:
        xl, yl = mesh.pos[np.argmax(run.laser(frame_k, 1))] * UM

    # Coordinates for melt-pool close-up
    zoom_w, zoom_h = 320.0, 160.0
    xlim_zoom = (xl - 200, xl + 120)
    ylim_zoom = (yl - 80, yl + 80)

    # Nodes inside the zoom box
    in_zoom = (
        (pos_x >= xlim_zoom[0]) & (pos_x <= xlim_zoom[1]) &
        (pos_y >= ylim_zoom[0]) & (pos_y <= ylim_zoom[1])
    )

    vmin, vmax = 300.0, float(max(T_gt.max(), T_b.max(), T_d.max()))
    err_vmax = 25.0  # Max error scale for visibility

    # 3 rows x 3 columns publication figure
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9])

    # -------------------------------------------------------------
    # ROW 1: THE GRAPH DISCRETIZATION & NODE TOPOLOGY
    # -------------------------------------------------------------
    # (1) Full plate adaptive mesh showing zones & node density
    ax_mesh = fig.add_subplot(gs[0, 0])
    zone_colors = np.array(["#2b83ba", "#abdda4", "#fdae61"])  # fine, transition, coarse
    ax_mesh.triplot(tri, color="#d0d0d0", lw=0.3, zorder=1)
    ax_mesh.scatter(pos_x, pos_y, c=mesh.zone, cmap=matplotlib.colors.ListedColormap(["#1b7837", "#f1a340", "#762a83"]),
                    s=1.5, alpha=0.7, zorder=2)
    band_outline(ax_mesh, mesh.waypoints, BAND, colors="black", linewidths=1.0, linestyles="--")
    ax_mesh.plot([xl], [yl], marker="*", color="red", ms=8, label="Laser Spot", zorder=5)
    # Draw zoom rectangle
    rect = plt.Rectangle((xlim_zoom[0], ylim_zoom[0]), zoom_w, zoom_h, fill=False, edgecolor="crimson", lw=1.5, ls="-")
    ax_mesh.add_patch(rect)
    ax_mesh.text(xlim_zoom[0], ylim_zoom[1] + 30, "Melt Pool ROI", color="crimson", fontsize=8, fontweight="bold")
    ax_mesh.set_title(f"(A) Adaptive Graph Mesh (N={mesh.n_nodes} Nodes, E={mesh.n_edges} Edges)", fontsize=10, fontweight="bold")
    ax_mesh.set_xlabel("x (um)", fontsize=8)
    ax_mesh.set_ylabel("y (um)", fontsize=8)
    ax_mesh.set_aspect("equal")

    # Legend for zones
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fine Band (h=10 um)', markerfacecolor='#1b7837', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Transition (h=15-33 um)', markerfacecolor='#f1a340', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Coarse Bulk (h=50 um)', markerfacecolor='#762a83', markersize=6),
    ]
    ax_mesh.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.85)

    # (2) Zoom into Melt Pool: Graph Nodes & Edges (Topology)
    ax_nodes_topo = fig.add_subplot(gs[0, 1])
    ax_nodes_topo.triplot(tri, color="#777777", lw=0.6, zorder=1)
    ax_nodes_topo.scatter(pos_x[in_zoom], pos_y[in_zoom], c="navy", s=18, edgecolors="white", lw=0.3, zorder=3, label="Graph Nodes")
    ax_nodes_topo.plot([xl], [yl], marker="*", color="red", ms=12, label="Laser Focus", zorder=5)
    ax_nodes_topo.set_xlim(*xlim_zoom)
    ax_nodes_topo.set_ylim(*ylim_zoom)
    ax_nodes_topo.set_title(f"(B) Graph Node Connectivity & Edge Channels (ROI)", fontsize=10, fontweight="bold")
    ax_nodes_topo.set_xlabel("x (um)", fontsize=8)
    ax_nodes_topo.set_ylabel("y (um)", fontsize=8)
    ax_nodes_topo.set_aspect("equal")
    ax_nodes_topo.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # (3) Zoom into Melt Pool: Discrete Nodes Colored by Ground Truth Temperature
    ax_nodes_temp = fig.add_subplot(gs[0, 2])
    ax_nodes_temp.triplot(tri, color="#555555", lw=0.4, alpha=0.5, zorder=1)
    sc = ax_nodes_temp.scatter(pos_x[in_zoom], pos_y[in_zoom], c=T_gt[in_zoom], cmap="inferno",
                               vmin=vmin, vmax=vmax, s=26, edgecolors="black", lw=0.4, zorder=3)
    ax_nodes_temp.tricontour(tri, T_gt, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.2, 1.5], zorder=4)
    ax_nodes_temp.plot([xl], [yl], marker="*", color="cyan", ms=10, zorder=5)
    ax_nodes_temp.set_xlim(*xlim_zoom)
    ax_nodes_temp.set_ylim(*ylim_zoom)
    ax_nodes_temp.set_title(f"(C) Discrete Node Temperatures (GT, t={t_ms:.2f} ms)", fontsize=10, fontweight="bold")
    ax_nodes_temp.set_xlabel("x (um)", fontsize=8)
    ax_nodes_temp.set_ylabel("y (um)", fontsize=8)
    ax_nodes_temp.set_aspect("equal")
    cb_nodes = plt.colorbar(sc, ax=ax_nodes_temp, fraction=0.046, pad=0.04)
    cb_nodes.set_label("Node T (K)", fontsize=8)

    # -------------------------------------------------------------
    # ROW 2: CROSS-MODEL HEAT FIELD PREDICTIONS (CONTINUOUS + NODES)
    # -------------------------------------------------------------
    # (4) Ground Truth Continuous Field
    ax_gt = fig.add_subplot(gs[1, 0])
    im_gt = ax_gt.tripcolor(tri, T_gt, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_gt.tricontour(tri, T_gt, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_gt.plot([xl], [yl], marker="+", color="white", ms=8, mew=1.2)
    ax_gt.set_xlim(*xlim_zoom)
    ax_gt.set_ylim(*ylim_zoom)
    ax_gt.set_title(f"(D) Ground Truth Thermal Field", fontsize=10, fontweight="bold")
    ax_gt.set_xlabel("x (um)", fontsize=8)
    ax_gt.set_ylabel("y (um)", fontsize=8)
    ax_gt.set_aspect("equal")

    # (5) Model B: Physics-Guided Spectral Residual
    ax_mb = fig.add_subplot(gs[1, 1])
    ax_mb.tripcolor(tri, T_b, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_mb.tricontour(tri, T_b, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_mb.plot([xl], [yl], marker="+", color="white", ms=8, mew=1.2)
    ax_mb.set_xlim(*xlim_zoom)
    ax_mb.set_ylim(*ylim_zoom)
    ax_mb.set_title(f"(E) Model B Prediction (Diffusion Res.)", fontsize=10, fontweight="bold")
    ax_mb.set_xlabel("x (um)", fontsize=8)
    ax_mb.set_ylabel("y (um)", fontsize=8)
    ax_mb.set_aspect("equal")

    # (6) Model D: Skew-Symmetric Conservative Edge Fluxes
    ax_md = fig.add_subplot(gs[1, 2])
    ax_md.tripcolor(tri, T_d, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    ax_md.tricontour(tri, T_d, levels=[T_S, T_L], colors=["#00ffff", "white"], linewidths=[1.0, 1.2])
    ax_md.plot([xl], [yl], marker="+", color="white", ms=8, mew=1.2)
    ax_md.set_xlim(*xlim_zoom)
    ax_md.set_ylim(*ylim_zoom)
    ax_md.set_title(f"(F) Model D Prediction (Conservative Flux)", fontsize=10, fontweight="bold")
    ax_md.set_xlabel("x (um)", fontsize=8)
    ax_md.set_ylabel("y (um)", fontsize=8)
    ax_md.set_aspect("equal")

    # -------------------------------------------------------------
    # ROW 3: ERROR MAPS & CENTERLINE PROFILE WITH DISCRETE GRAPH NODES
    # -------------------------------------------------------------
    # (7) Model B Node Error Map
    ax_err_b = fig.add_subplot(gs[2, 0])
    im_eb = ax_err_b.tripcolor(tri, err_b, cmap="magma", vmin=0, vmax=err_vmax, shading="gouraud")
    ax_err_b.scatter(pos_x[in_zoom], pos_y[in_zoom], c=err_b[in_zoom], cmap="magma", vmin=0, vmax=err_vmax, s=8, alpha=0.6)
    ax_err_b.set_xlim(*xlim_zoom)
    ax_err_b.set_ylim(*ylim_zoom)
    ax_err_b.set_title(f"(G) Model B Absolute Error |T_pred - T_true|\n(Max ROI Err = {np.max(err_b[in_zoom]):.1f} K)", fontsize=9, fontweight="bold")
    ax_err_b.set_xlabel("x (um)", fontsize=8)
    ax_err_b.set_ylabel("y (um)", fontsize=8)
    ax_err_b.set_aspect("equal")
    cb_eb = plt.colorbar(im_eb, ax=ax_err_b, fraction=0.046, pad=0.04)
    cb_eb.set_label("|Err| (K)", fontsize=8)

    # (8) Model D Node Error Map
    ax_err_d = fig.add_subplot(gs[2, 1])
    im_ed = ax_err_d.tripcolor(tri, err_d, cmap="magma", vmin=0, vmax=err_vmax, shading="gouraud")
    ax_err_d.scatter(pos_x[in_zoom], pos_y[in_zoom], c=err_d[in_zoom], cmap="magma", vmin=0, vmax=err_vmax, s=8, alpha=0.6)
    ax_err_d.set_xlim(*xlim_zoom)
    ax_err_d.set_ylim(*ylim_zoom)
    ax_err_d.set_title(f"(H) Model D Absolute Error |T_pred - T_true|\n(Max ROI Err = {np.max(err_d[in_zoom]):.1f} K)", fontsize=9, fontweight="bold")
    ax_err_d.set_xlabel("x (um)", fontsize=8)
    ax_err_d.set_ylabel("y (um)", fontsize=8)
    ax_err_d.set_aspect("equal")
    cb_ed = plt.colorbar(im_ed, ax=ax_err_d, fraction=0.046, pad=0.04)
    cb_ed.set_label("|Err| (K)", fontsize=8)

    # (9) 1D Track Profile displaying DISCRETE GRAPH NODES
    ax_prof = fig.add_subplot(gs[2, 2])
    near_track = np.abs(pos_y - yl) < 12.0  # within 12 um of laser path
    sort_idx = np.argsort(pos_x[near_track])
    xs = pos_x[near_track][sort_idx]
    gt_nodes = T_gt[near_track][sort_idx]
    b_nodes = T_b[near_track][sort_idx]
    d_nodes = T_d[near_track][sort_idx]

    # Continuous lines
    ax_prof.plot(xs, gt_nodes, "k-", lw=1.8, label="Ground Truth", alpha=0.85)
    ax_prof.plot(xs, b_nodes, "r--", lw=1.4, label="Model B (Res.)", alpha=0.9)
    ax_prof.plot(xs, d_nodes, "b-.", lw=1.4, label="Model D (Cons.)", alpha=0.9)

    # Discrete graph nodes plotted as markers!
    ax_prof.scatter(xs, gt_nodes, color="black", s=14, zorder=4, marker="o", alpha=0.7, label="GT Graph Nodes")
    ax_prof.scatter(xs, b_nodes, color="red", s=14, zorder=4, marker="^", alpha=0.7, label="Model B Nodes")
    ax_prof.scatter(xs, d_nodes, color="blue", s=14, zorder=4, marker="s", alpha=0.7, label="Model D Nodes")

    ax_prof.axhline(T_S, color="#00a8a8", ls=":", lw=1.0, label=f"Solidus ({T_S:.0f} K)")
    ax_prof.axhline(T_L, color="#888888", ls=":", lw=1.0, label=f"Liquidus ({T_L:.0f} K)")
    ax_prof.set_xlim(*xlim_zoom)
    ax_prof.set_ylim(250, vmax + 50)
    ax_prof.set_xlabel("x (um) along Track", fontsize=8)
    ax_prof.set_ylabel("Temperature (K)", fontsize=8)
    ax_prof.set_title("(I) Centerline Profile with Discrete Node Markers", fontsize=9, fontweight="bold")
    ax_prof.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=2)
    ax_prof.grid(True, alpha=0.25, ls=":")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Successfully generated node-level comparison figure at: {out_path}")


if __name__ == "__main__":
    generate_node_level_comparison()
