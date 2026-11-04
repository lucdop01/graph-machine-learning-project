"""Fixed physical mesh-to-grid and grid-to-mesh maps for the U-Net baseline.

The grid has 10 um square cells, centred at ((i + 1/2) dx, (j + 1/2) dx),
with array layout (..., y, x). This is the standard solver cell layout, but
the values here always come from the processed mesh nodes, never from solver
grid frames. The same dx is used for every split, including test_fine.
"""

import numpy as np
import matplotlib.tri as mtri

GRID_DX = 10e-6
_ADAPTERS = {}


class GridAdapter:
    """Precomputed geometry maps for one mesh and one physical grid spacing.

    Mesh -> grid uses the mesh's existing triangles and barycentric weights.
    Grid -> mesh uses bilinear weights on cell centres, clamped at the plate
    boundary like the solver's cell-centred sampler. Both maps preserve units.
    """

    def __init__(self, mesh, dx=GRID_DX):
        dx = float(dx)
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("grid spacing must be finite and positive")
        nx, ny = round(mesh.Lx / dx), round(mesh.Ly / dx)
        if (nx < 2 or ny < 2 or not np.isclose(nx * dx, mesh.Lx, rtol=0, atol=1e-12)
                or not np.isclose(ny * dx, mesh.Ly, rtol=0, atol=1e-12)):
            raise ValueError("grid spacing must divide both plate dimensions into at least two cells")
        self.mesh, self.dx = mesh, dx
        self.nx, self.ny = nx, ny
        self.shape = (ny, nx)
        self.x = (np.arange(nx) + 0.5) * dx
        self.y = (np.arange(ny) + 0.5) * dx

        # The supplied triangles cover the exact plate. The finder uses their
        # connectivity; it does not produce a second triangulation.
        X, Y = np.meshgrid(self.x, self.y, indexing="xy")
        points = np.column_stack((X.ravel(), Y.ravel()))
        triangulation = mtri.Triangulation(mesh.pos[:, 0], mesh.pos[:, 1], triangles=mesh.tri)
        triangle_id = np.asarray(triangulation.get_trifinder()(points[:, 0], points[:, 1]))
        if np.any(triangle_id < 0):
            raise ValueError(f"{np.count_nonzero(triangle_id < 0)} grid cells are outside the mesh triangles")
        vertices = np.asarray(mesh.tri[triangle_id], dtype=np.int64)
        p0, p1, p2 = (mesh.pos[vertices[:, i]] for i in range(3))
        u, v, q = p1 - p0, p2 - p0, points - p0
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        w1 = (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]) / det
        w2 = (u[:, 0] * q[:, 1] - u[:, 1] * q[:, 0]) / det
        weights = np.column_stack((1 - w1 - w2, w1, w2))
        if np.min(weights) < -1e-9 or not np.allclose(weights.sum(axis=1), 1, atol=1e-12):
            raise ValueError("mesh-to-grid barycentric weights are invalid")
        self.mesh_indices, self.mesh_weights = vertices, weights

        fx, fy = mesh.pos[:, 0] / dx - 0.5, mesh.pos[:, 1] / dx - 0.5
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        wx, wy = np.clip(fx - i0, 0.0, 1.0), np.clip(fy - j0, 0.0, 1.0)
        self.grid_indices = np.column_stack((j0 * nx + i0, j0 * nx + i0 + 1,
                                             (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1))
        self.grid_weights = np.column_stack(((1 - wx) * (1 - wy), wx * (1 - wy),
                                             (1 - wx) * wy, wx * wy))
        if np.min(self.grid_weights) < 0 or not np.allclose(self.grid_weights.sum(axis=1), 1, atol=1e-12):
            raise ValueError("grid-to-mesh bilinear weights are invalid")
        self._torch_maps = {}

    def nodes_to_grid(self, values):
        """Rasterize node values with shape (..., nodes) to (..., y, x)."""
        values = np.asarray(values)
        if values.ndim < 1 or values.shape[-1] != self.mesh.n_nodes:
            raise ValueError(f"expected (..., {self.mesh.n_nodes}) node values")
        raster = np.sum(values[..., self.mesh_indices] * self.mesh_weights, axis=-1)
        return raster.reshape(*values.shape[:-1], *self.shape)

    def grid_to_nodes(self, values):
        """Bilinearly sample (..., y, x) at nodes; supports autograd for torch tensors."""
        if len(values.shape) < 2 or tuple(values.shape[-2:]) != self.shape:
            raise ValueError(f"expected (..., {self.ny}, {self.nx}) grid values")
        flat = values.reshape(*values.shape[:-2], self.nx * self.ny)
        if isinstance(values, np.ndarray):
            return np.sum(flat[..., self.grid_indices] * self.grid_weights, axis=-1)
        import torch
        if not isinstance(values, torch.Tensor):
            raise TypeError("grid values must be a numpy array or torch tensor")
        key = (values.device, values.dtype)
        if key not in self._torch_maps:
            self._torch_maps[key] = (torch.as_tensor(self.grid_indices, device=values.device),
                                     torch.as_tensor(self.grid_weights, device=values.device, dtype=values.dtype))
        indices, weights = self._torch_maps[key]
        return (flat[..., indices] * weights).sum(dim=-1)


def get_grid_adapter(mesh, dx=GRID_DX):
    """Reuse the geometry maps across runs and time steps of the same mesh."""
    key = (mesh.fingerprint(), float(dx), float(mesh.Lx), float(mesh.Ly))
    if key not in _ADAPTERS:
        _ADAPTERS[key] = GridAdapter(mesh, dx)
    return _ADAPTERS[key]
