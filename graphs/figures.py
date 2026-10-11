"""Figures for the graph preprocessing (plan step 2).

  mesh_straight.png        the straight-track mesh with the laser track and the fine band, plus a zoom on the
                           step from 10 to 50 um
  interp_midtrack.png      solver temperature on its cells next to the node temperature on the mesh, with the laser
                           at mid-track, solidus and liquidus contours on both

Usage (from the repo root):
  python -m graphs.figures                                   simulates straight 195 W, 0.5 m/s (largest pool)
  python -m graphs.figures --run data/runs/test_interp_000.h5
Output goes to results/figures/ (git-ignored)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from solver.ti64 import T_S, T_L
from graphs.mesh import get_mesh, BAND, H_FINE, H_COARSE, GRADE
from graphs.preprocess import load_run, mesh_for_run, node_temperature, grid_of

TRACK, BAND_C, EDGE_C, INK, MUTED = "#eb6834", "#2a78d6", "#8d8c88", "#0b0b0b", "#52514e"
UM = 1e6


def _style(ax, xlabel=True, ylabel=True):
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.6)
    if xlabel:
        ax.set_xlabel("x (um)", color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel("y (um)", color=MUTED, fontsize=9)


def band_outline(ax, waypoints, dist, **kw):
    """Contour of the fine band edge, from a fine evaluation grid of the distance to the path."""
    from graphs.mesh import path_distance
    xs, ys = np.linspace(*ax.get_xlim(), 600) / UM, np.linspace(*ax.get_ylim(), 400) / UM
    X, Y = np.meshgrid(xs, ys)
    D = path_distance(np.c_[X.ravel(), Y.ravel()], waypoints).reshape(X.shape)
    ax.contour(X * UM, Y * UM, D, levels=[dist], **kw)


def mesh_figure(out, seed=0):
    m = get_mesh("straight", seed)
    q = m.quality()
    tri = mtri.Triangulation(m.pos[:, 0] * UM, m.pos[:, 1] * UM, m.tri)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[2, 1.15]))
    w = m.waypoints * UM
    for ax, lim in zip(axes, [((0, m.Lx * UM), (0, m.Ly * UM)), ((1100, 1500), (450, 880))]):
        ax.triplot(tri, color=EDGE_C, lw=0.25 if ax is axes[0] else 0.45)
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        band_outline(ax, m.waypoints, BAND, colors=BAND_C, linewidths=1.2, linestyles="--")
        band_outline(ax, m.waypoints, BAND + (H_COARSE - H_FINE) / GRADE, colors=BAND_C, linewidths=0.8,
                     linestyles=":")
        ax.plot(w[:, 0], w[:, 1], color=TRACK, lw=2, solid_capstyle="round", zorder=5)
        _style(ax, ylabel=ax is axes[0])
    axes[0].plot([1100, 1500, 1500, 1100, 1100], [450, 450, 880, 880, 450], color=INK, lw=0.8)
    axes[0].set_title(f"Straight-track mesh: {m.n_nodes:,} nodes, {len(m.tri):,} triangles, "
                      f"smallest angle {q['min']:.1f} deg", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Zoom: 10 um band, graded step, 50 um lattice", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=TRACK, lw=2, label="laser track"),
               plt.Line2D([], [], color=BAND_C, lw=1.2, ls="--", label=f"fine band edge ({BAND * UM:.0f} um)"),
               plt.Line2D([], [], color=BAND_C, lw=0.8, ls=":",
                          label=f"50 um spacing reached ({(BAND + (H_COARSE - H_FINE) / GRADE) * UM:.0f} um)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return m


def interp_figure(run, out, seed=0):
    m = mesh_for_run(run, seed)
    nx, ny, dx, _ = grid_of(run)
    xl_mid = 0.5 * (run["waypoints"][0] + run["waypoints"][-1])
    k = int(np.argmin(np.linalg.norm(run["laser"] - xl_mid, axis=1)))    # frame with the laser at mid-track
    T_cell = run["T"][k].astype(float)
    T_node = node_temperature(dict(run, T=run["T"][k:k + 1]), m)[0].astype(float)
    xc, yc = run["laser"][k] * UM
    lim = ((xc - 450, xc + 250), (yc - 200, yc + 200))
    vmin, vmax = 300.0, float(np.ceil(T_cell.max() / 100) * 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    xe, ye = np.arange(nx + 1) * dx * UM, np.arange(ny + 1) * dx * UM
    im = axes[0].pcolormesh(xe, ye, T_cell.T, cmap="inferno", vmin=vmin, vmax=vmax, shading="flat")
    xcen, ycen = (np.arange(nx) + 0.5) * dx * UM, (np.arange(ny) + 0.5) * dx * UM
    axes[0].contour(xcen, ycen, T_cell.T, levels=[T_S, T_L], colors=["#c3c2b7", "white"], linewidths=0.9)
    tri = mtri.Triangulation(m.pos[:, 0] * UM, m.pos[:, 1] * UM, m.tri)
    axes[1].tripcolor(tri, T_node, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[1].triplot(tri, color="white", lw=0.15, alpha=0.35)
    axes[1].tricontour(tri, T_node, levels=[T_S, T_L], colors=["#c3c2b7", "white"], linewidths=0.9)
    inside = ((m.pos[:, 0] * UM > lim[0][0]) & (m.pos[:, 0] * UM < lim[0][1])
              & (m.pos[:, 1] * UM > lim[1][0]) & (m.pos[:, 1] * UM < lim[1][1]))
    for ax in axes:
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.plot([xc], [yc], marker="+", color="white", ms=9, mew=1.2)
        band_outline(ax, run["waypoints"], BAND, colors=BAND_C, linewidths=1.0, linestyles="--")
        _style(ax, ylabel=ax is axes[0])
    axes[0].set_title(f"Solver, {dx * UM:.0f} um cells", fontsize=10, color=INK, loc="left")
    axes[1].set_title(f"Mesh nodes, bilinear T ({int(inside.sum())} nodes in view)", fontsize=10, color=INK,
                      loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("temperature (K)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(f"{run['path']} {run['P']:.0f} W, {run['v']:.2f} m/s at t = {run['t'][k] * 1e3:.3f} ms (frame {k}); "
                 f"grey line solidus {T_S:.0f} K, white line liquidus {T_L:.0f} K, dashed fine band edge, + laser",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run file; default: simulate straight 195 W, 0.5 m/s")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--mesh-seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mesh_figure(os.path.join(a.out, "mesh_straight.png"), a.mesh_seed)
    if a.run:
        run = load_run(a.run)
    else:
        from solver.solver import simulate
        from solver.paths import straight
        run = simulate(195.0, 0.5, straight())
        run.update(path="straight", split="test_extrap")
    k = interp_figure(run, os.path.join(a.out, "interp_midtrack.png"), a.mesh_seed)
    print(f"wrote {a.out}/mesh_straight.png and {a.out}/interp_midtrack.png (frame {k})")
