"""Mass-lumped P1 finite-element diffusion on triangular meshes (SI units).

K has diagonal sum(w_ij), off-diagonal -w_ij; L=M^-1 K.
Keep signed cotangents: clipping them changes the finite-element operator.
Natural boundary conditions are insulated. No mesh-length normalisation is used.
"""
import numpy as np
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
