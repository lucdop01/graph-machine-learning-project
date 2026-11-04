"""Mass-lumped P1 finite-element diffusion on triangular meshes (SI units).

K has diagonal sum(w_ij), off-diagonal -w_ij; L=M^-1 K.
Keep signed cotangents: clipping them changes the finite-element operator.
Natural boundary conditions are insulated. No mesh-length normalisation is used.

cotangent_weights and stiffness serve Model B's fixed branch and the fixed linear filter. cotangent_laplacian
assembles the same stiffness per triangle and returns the area-lumped operator -diag(1 / A_i) K, which the fitted
LF-T filter uses.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, diags


def cotangent_weights(pos, tri, edge_index):
    pos, tri = np.asarray(pos, float), np.asarray(tri, int)
    weights = {}
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = pos[tri[:, a]] - pos[tri[:, c]]
        v = pos[tri[:, b]] - pos[tri[:, c]]
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        if np.any(cross <= 0):
            raise ValueError("degenerate mesh triangle")
        cot = 0.5 * np.sum(u * v, axis=1) / cross
        for i, j, w in zip(tri[:, a], tri[:, b], cot):
            key = tuple(sorted((int(i), int(j))))
            weights[key] = weights.get(key, 0.0) + float(w)
    return np.array([weights[tuple(sorted((int(i), int(j))))]
                     for i, j in np.asarray(edge_index).T])


def stiffness(mesh):
    w = cotangent_weights(mesh.pos, mesh.tri, mesh.edge_index)
    i, j = mesh.edge_index
    off = coo_matrix((-w, (i, j)), shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()
    return off + diags(-np.asarray(off.sum(axis=1)).ravel())


def cotangent_laplacian(mesh):
    """Return the area-lumped diffusion Laplacian as an ``(n, n)`` CSR matrix.

    For the piecewise-linear basis functions on a mesh, let S be the finite-element
    stiffness matrix and A_i the lumped node areas in ``Mesh.area``. This returns
    L = -diag(1 / A_i) S, with units m^-2. Thus ``L @ T`` points in the direction
    of heat diffusion, ``L @ constant`` is zero, and the mesh boundary has the
    natural zero-flux condition. The physical operator is left unscaled; models may
    choose a separate numerical scaling for polynomial features.

    The triangle stiffness assembly is equivalent to cotangent edge weights.
    ``mesh.area`` supplies the same node areas used by graph preprocessing and
    evaluation, so no second area convention is introduced here.
    """
    pos = np.asarray(mesh.pos, dtype=np.float64)
    tri = np.asarray(mesh.tri, dtype=np.int64)
    area = np.asarray(mesh.area, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("expected 2D node positions and triangular faces")
    n = len(pos)
    if not len(tri) or np.any(tri < 0) or np.any(tri >= n):
        raise ValueError("triangles must contain valid node indices")
    if area.shape != (n,) or not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("every node needs a positive finite lumped area")

    p = pos[tri]
    x, y = p[:, :, 0], p[:, :, 1]
    twice_signed_area = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                         - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0]))
    if not np.all(np.isfinite(twice_signed_area)) or np.any(twice_signed_area == 0):
        raise ValueError("triangles must have nonzero finite area")

    # Gradient of each linear nodal basis function on each triangle.
    gradients = np.stack((
        np.stack((y[:, 1] - y[:, 2], x[:, 2] - x[:, 1]), axis=1),
        np.stack((y[:, 2] - y[:, 0], x[:, 0] - x[:, 2]), axis=1),
        np.stack((y[:, 0] - y[:, 1], x[:, 1] - x[:, 0]), axis=1),
    ), axis=1) / twice_signed_area[:, None, None]
    local = (0.5 * np.abs(twice_signed_area))[:, None, None] * np.einsum(
        "tic,tjc->tij", gradients, gradients)

    rows = np.broadcast_to(tri[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(tri[:, None, :], local.shape).ravel()
    stiffness = sparse.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    stiffness.sum_duplicates()
    # Remove accumulated roundoff from the row sums without changing symmetry.
    stiffness.setdiag(stiffness.diagonal() - np.asarray(stiffness.sum(axis=1)).ravel())
    stiffness.eliminate_zeros()

    laplacian = -sparse.diags(1.0 / area) @ stiffness
    laplacian = laplacian.tocsr()
    laplacian.sort_indices()
    return laplacian
