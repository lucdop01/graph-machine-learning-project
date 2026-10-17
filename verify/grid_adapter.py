"""Synthetic-field checks of the U-Net mesh/grid adapter.

Run from the repository root: python -m verify.grid_adapter
Resolution diagnostic: python -m verify.grid_adapter --resolution-sweep
The hotspot diagnostic measures interpolation loss; it does not train a model.
"""

import argparse

import numpy as np

from graphs.grid import GRID_DX, get_grid_adapter
from graphs.mesh import get_mesh
from solver.events import bilinear_sampler
from solver.ti64 import T_S


def check_geometry(mesh):
    adapter = get_grid_adapter(mesh)
    assert get_grid_adapter(mesh) is adapter  # geometry reused across time steps
    assert adapter.shape == (round(mesh.Ly / GRID_DX), round(mesh.Lx / GRID_DX))
    np.testing.assert_allclose(adapter.grid_weights.sum(axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(adapter.mesh_weights.sum(axis=1), 1, atol=1e-15)
    assert np.min(adapter.grid_weights) >= 0 and np.min(adapter.mesh_weights) >= -1e-9

    constant = np.full(mesh.n_nodes, 543.0)
    np.testing.assert_allclose(adapter.nodes_to_grid(constant), 543.0, atol=1e-11)
    np.testing.assert_allclose(adapter.grid_to_nodes(adapter.nodes_to_grid(constant)), constant, atol=1e-11)

    # Different slopes distinguish swapped x/y axes and accidental array transposes.
    slope_x, slope_y = 200.0 / mesh.Lx, 80.0 / mesh.Ly
    affine = 300.0 + slope_x * mesh.pos[:, 0] + slope_y * mesh.pos[:, 1]
    X, Y = np.meshgrid(adapter.x, adapter.y, indexing="xy")
    grid_truth = 300.0 + slope_x * X + slope_y * Y
    raster = adapter.nodes_to_grid(affine)
    np.testing.assert_allclose(raster, grid_truth, rtol=0, atol=1e-9)

    restored = adapter.grid_to_nodes(raster)
    inside = ((mesh.pos[:, 0] >= adapter.x[0]) & (mesh.pos[:, 0] <= adapter.x[-1])
              & (mesh.pos[:, 1] >= adapter.y[0]) & (mesh.pos[:, 1] <= adapter.y[-1]))
    np.testing.assert_allclose(restored[inside], affine[inside], rtol=0, atol=1e-9)
    # Cell-centred grids clamp within half a cell of each physical edge.
    bound = 0.5 * GRID_DX * (abs(slope_x) + abs(slope_y))
    assert np.max(np.abs(restored - affine)) <= bound + 1e-9

    # The array-axis convention matches the project's existing cell sampler.
    solver_layout = grid_truth.T  # solver: (x, y); CNN: (y, x)
    np.testing.assert_allclose(adapter.grid_to_nodes(grid_truth),
                               bilinear_sampler(mesh.pos, adapter.nx, adapter.ny, adapter.dx)(solver_layout),
                               rtol=0, atol=1e-12)

    # Channels and batches must retain their leading dimensions.
    batch = np.stack((affine, constant))[None]
    assert adapter.nodes_to_grid(batch).shape == (1, 2, *adapter.shape)
    assert adapter.grid_to_nodes(adapter.nodes_to_grid(batch)).shape == batch.shape
    return float(np.max(np.abs(restored - affine)))


def check_hotspot(mesh, dx, offset):
    """Quantify round-trip loss near a synthetic melt-pool contour."""
    adapter = get_grid_adapter(mesh, dx)
    centre = np.array([mesh.Lx / 2 + offset, mesh.Ly / 2 + offset])
    radius = 45e-6
    r2 = np.sum((mesh.pos - centre) ** 2, axis=1)
    temperature = 300.0 + 2200.0 * np.exp(-0.5 * r2 / radius ** 2)
    recovered = adapter.grid_to_nodes(adapter.nodes_to_grid(temperature))
    error = np.abs(recovered - temperature)
    contour = np.abs(temperature - T_S) <= 100.0
    if not np.any(contour):
        raise AssertionError("synthetic hotspot has no nodes near the solidus")
    hot = temperature >= T_S
    recovered_hot = recovered >= T_S
    intersection = mesh.area[hot & recovered_hot].sum()
    union = mesh.area[hot | recovered_hot].sum()
    return dict(rmse_K=float(np.sqrt(np.mean(error ** 2))),
                max_K=float(error.max()), contour_nodes=int(contour.sum()),
                contour_mae_K=float(error[contour].mean()),
                contour_max_K=float(error[contour].max()),
                solidus_iou=float(intersection / union))


def check_gradient(mesh):
    import torch

    adapter = get_grid_adapter(mesh)
    grid = torch.randn((2, 1, *adapter.shape), dtype=torch.float64, requires_grad=True)
    values = adapter.grid_to_nodes(grid)
    assert values.shape == (2, 1, mesh.n_nodes)
    values.square().mean().backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()
    assert torch.count_nonzero(grid.grad) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-sweep", action="store_true",
                        help="compare the same synthetic hotspot at 10, 5, about 3, 2.5 and 2 um")
    args = parser.parse_args()
    for name, seed in (("straight", 0), ("straight", 1), ("l_turn", 0),
                       ("zigzag", 0), ("straight_large", 0)):
        mesh = get_mesh(name, seed)
        boundary_error = check_geometry(mesh)
        print(f"{name} seed {seed}: grid {get_grid_adapter(mesh).shape}, "
              f"max affine round-trip error {boundary_error:.3f} K")
    mesh = get_mesh("straight", 0)
    check_gradient(mesh)
    cases = [(GRID_DX, 0.0), (GRID_DX, 5e-6), (5e-6, 5e-6)]
    if args.resolution_sweep:
        # 3 um cannot divide a 1 mm plate exactly. 333 cells give a uniform
        # 3.003003... um spacing on both axes without extrapolating outside it.
        cases.extend(((mesh.Ly / 333, 5e-6), (2.5e-6, 5e-6), (2e-6, 5e-6)))
    for dx, offset in cases:
        print(f"straight hotspot, dx={dx * 1e6:g} um, offset={offset * 1e6:g} um: "
              f"{check_hotspot(mesh, dx, offset)}", flush=True)
    print("mesh/grid adapter: constants, affine field, boundaries, shape, cache and gradient pass")


if __name__ == "__main__":
    main()
