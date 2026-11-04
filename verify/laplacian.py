"""Check the cotangent diffusion operator on an exact fixture and every committed mesh.

Usage: python -m verify.laplacian
"""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from graphs.laplacian import cotangent_laplacian
from graphs.mesh import get_mesh


def check(mesh):
    L = cotangent_laplacian(mesh)
    mass_L = sparse.diags(mesh.area) @ L
    symmetry = mass_L - mass_L.T
    scale = max(1.0, np.max(np.abs(mass_L.data)))
    if symmetry.nnz and np.max(np.abs(symmetry.data)) >= 1e-12 * scale:
        raise AssertionError("area-weighted Laplacian is not symmetric")

    rng = np.random.default_rng(0)
    u = rng.standard_normal(mesh.n_nodes)
    diffusion = mass_L @ u
    if u @ diffusion > 1e-10:
        raise AssertionError("diffusion increases the quadratic energy")
    if abs(np.sum(diffusion)) >= 1e-10 * np.sum(np.abs(diffusion)):
        raise AssertionError("diffusion does not conserve area-weighted temperature")
    if np.max(np.abs(L @ np.ones(mesh.n_nodes))) >= 1e-10 * np.max(np.abs(L.data)):
        raise AssertionError("constant field is not stationary")
    return L


def main():
    # Two right triangles forming a unit square: perimeter cotangent weights
    # are 1/2; the diagonal's weight is zero.
    toy = SimpleNamespace(
        pos=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        tri=np.array([[0, 1, 2], [0, 2, 3]]),
        area=np.array([2., 1., 2., 1.]) / 6,
        n_nodes=4,
    )
    expected = np.array([
        [1., -0.5, 0., -0.5],
        [-0.5, 1., -0.5, 0.],
        [0., -0.5, 1., -0.5],
        [-0.5, 0., -0.5, 1.],
    ])
    np.testing.assert_allclose((-sparse.diags(toy.area) @ check(toy)).toarray(), expected, atol=1e-14)
    print("unit square: exact stiffness and zero-flux checks pass")

    for name, seed in (("straight", 0), ("straight", 1), ("l_turn", 0),
                       ("zigzag", 0), ("straight_large", 0)):
        mesh = get_mesh(name, seed)
        L = check(mesh)
        print(f"{name} seed {seed}: {mesh.n_nodes} nodes, {L.nnz} nonzeros, checks pass")


if __name__ == "__main__":
    main()
