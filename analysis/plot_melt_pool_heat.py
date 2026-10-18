"""Publication-quality visualization of melt-pool thermal dynamics and isotherms.

Generates:
1. results/figures/melt_pool_thermal_evolution.png:
   Multi-temporal thermal fields across heating, steady-state melting, and solidification.
2. results/figures/melt_pool_physics_overview.png:
   Detailed zoom on the melt pool with isotherms (Solidus 1,878 K, Liquidus 1,928 K),
   adaptive mesh overlay, and 1D centerline temperature profile.
"""
import os
import h5py
import numpy as np

# Use non-interactive backend with temporary cache dir
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from solver.ti64 import T_S, T_L
from solver.paths import Path, straight
from graphs.mesh import BAND
from graphs.figures import band_outline, _style


def plot_thermal_dynamics(run_h5_path: str = "data/graphs/seed0/val_000.h5", out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading {run_h5_path}...")
    with h5py.File(run_h5_path, "r") as f:
        T_all = f["T"][:]          # (frames, nodes)
        t_all = f["t"][:]          # (frames,)
        attrs = dict(f.attrs)
    
    mesh_path = f"data/meshes/{attrs['mesh']}_seed{attrs['mesh_seed']}.npz"
    print(f"Loading mesh {mesh_path}...")
    with np.load(mesh_path) as m_data:
        pos = m_data["pos"]        # (nodes, 2)
        tri_indices = m_data["tri"] # (triangles, 3)
        Lx, Ly = float(m_data["Lx"]), float(m_data["Ly"])

    UM = 1e6
    pos_x, pos_y = pos[:, 0] * UM, pos[:, 1] * UM
    tri = mtri.Triangulation(pos_x, pos_y, tri_indices)
    wp = straight(Lx, Ly)
    v_laser = float(attrs.get("v", 0.8))
    path = Path(wp, v=v_laser)

    # -------------------------------------------------------------
    # Figure 1: Thermal Evolution Across 3 Distinct Regimes
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), constrained_layout=True)
    frames = [
        int(len(t_all) * 0.20),   # Frame 1: Early development
        int(len(t_all) * 0.50),   # Frame 2: Quasi-steady melt pool
        int(len(t_all) * 0.85),   # Frame 3: Laser off / solidification
    ]
    regimes = ["(A) Early Laser Heating", "(B) Quasi-Steady Melt Pool Formation", "(C) Post-Laser Solidification & Cooling"]

    vmin, vmax = 300.0, float(np.percentile(T_all, 99.9))

    for ax, k, title in zip(axes, frames, regimes):
        t_ms = float(t_all[k]) * 1e3
        T_k = T_all[k]
        xl, yl = path.pos(t_all[k]) * UM

        im = ax.tripcolor(tri, T_k, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
        
        # Melt pool isotherms
        if np.any(T_k >= T_S):
            ax.tricontour(tri, T_k, levels=[T_S], colors=["cyan"], linewidths=1.2)
        if np.any(T_k >= T_L):
            ax.tricontour(tri, T_k, levels=[T_L], colors=["white"], linewidths=1.5)

        # Plot laser spot if active
        laser_off_t = float(attrs.get("laser_off", 1.5e-3))
        if t_all[k] <= laser_off_t:
            ax.plot([xl], [yl], marker="+", color="white", ms=12, mew=2.0, label="Laser Spot")
            circle = plt.Circle((xl, yl), 50.0, color="white", fill=False, ls="--", lw=0.8, alpha=0.8)
            ax.add_patch(circle)

        band_outline(ax, wp, BAND, colors="#2a78d6", linewidths=1.0, linestyles="--")
        
        ax.set_xlim(0, Lx * UM)
        ax.set_ylim(0, Ly * UM)
        ax.set_title(f"{title} (t = {t_ms:.2f} ms | T_max = {T_k.max():.0f} K)", fontsize=11, fontweight="bold", loc="left")
        _style(ax, xlabel=(ax is axes[-1]))

    # Shared colorbar
    cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cb.set_label("Temperature T (K)", fontsize=10)
    cb.ax.axhline(T_S, color="cyan", lw=1.5, label="Solidus")
    cb.ax.axhline(T_L, color="white", lw=1.5, label="Liquidus")

    fig1_path = os.path.join(out_dir, "melt_pool_thermal_evolution.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    print(f"Generated {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Detailed Melt Pool Overview with Isotherms & Profile
    # -------------------------------------------------------------
    k_mid = int(len(t_all) * 0.45)
    T_mid = T_all[k_mid]
    xl, yl = path.pos(t_all[k_mid]) * UM

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[1.2, 1.0]), constrained_layout=True)

    # Top: Zoom on the melt pool with mesh wireframe
    zoom_x = (xl - 400, xl + 300)
    zoom_y = (yl - 180, yl + 180)

    im2 = ax_top.tripcolor(tri, T_mid, cmap="inferno", vmin=300.0, vmax=vmax, shading="gouraud")
    ax_top.triplot(tri, color="white", lw=0.25, alpha=0.35)
    if np.any(T_mid >= T_S):
        cs = ax_top.tricontour(tri, T_mid, levels=[T_S, T_L], colors=["cyan", "white"], linewidths=[1.5, 1.8])
    ax_top.plot([xl], [yl], marker="+", color="white", ms=14, mew=2.2, label="Laser Center")
    ax_top.plot([xl - 400, xl + 300], [yl, yl], color="yellow", ls=":", lw=1.2, label="Centerline Cut")
    band_outline(ax_top, wp, BAND, colors="#2a78d6", linewidths=1.2, linestyles="--")

    ax_top.set_xlim(*zoom_x)
    ax_top.set_ylim(*zoom_y)
    ax_top.set_title(f"Melt-Pool Interface Zoom on Unstructured Mesh (t = {t_all[k_mid]*1e3:.2f} ms)", fontsize=11, fontweight="bold")
    _style(ax_top)
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.9)

    cb2 = fig.colorbar(im2, ax=ax_top, orientation="vertical", fraction=0.03, pad=0.02)
    cb2.set_label("Temperature (K)", fontsize=9)

    # Bottom: Centerline Temperature Profile T(x)
    near_line = np.abs(pos_y - yl) < 15.0
    sort_i = np.argsort(pos_x[near_line])
    xs = pos_x[near_line][sort_i]
    Ts = T_mid[near_line][sort_i]

    ax_bot.plot(xs, Ts, "r-", lw=2.2, label="Centerline Temperature T(x)")
    ax_bot.axhline(T_S, color="cyan", ls="--", lw=1.2, label=f"Solidus T_S = {T_S:.0f} K")
    ax_bot.axhline(T_L, color="white", ls="-.", lw=1.2, label=f"Liquidus T_L = {T_L:.0f} K")
    ax_bot.axvspan(xl - 50, xl + 50, color="orange", alpha=0.15, label="Laser Spot Diameter (100 um)")
    ax_bot.set_xlim(*zoom_x)
    ax_bot.set_xlabel("x (um)", fontsize=10)
    ax_bot.set_ylabel("Temperature (K)", fontsize=10)
    ax_bot.set_title("1D Centerline Thermal Profile Across Melt Pool & Mushy Zone", fontsize=11, fontweight="bold")
    ax_bot.grid(True, alpha=0.3, ls=":")
    ax_bot.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax_bot.set_facecolor("#1a1a1a")

    fig2_path = os.path.join(out_dir, "heat_prediction_overview.png")
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    print(f"Generated {fig2_path}")


if __name__ == "__main__":
    plot_thermal_dynamics()
