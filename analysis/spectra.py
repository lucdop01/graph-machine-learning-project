"""Mass-normalised mesh spectrum: python -m analysis.spectra --mesh straight --seeds 0 1."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from graphs.mesh import get_mesh, PATHS
from graphs.laplacian import stiffness


def spectrum(mesh, modes=16):
    if not 2 <= modes < mesh.n_nodes:
        raise ValueError("modes must be between 2 and node count - 1")
    inv = diags(1 / np.sqrt(mesh.area))
    symmetric = inv @ stiffness(mesh) @ inv
    # Small negative shift avoids factoring the singular constant mode.
    values = np.sort(eigsh(symmetric, k=modes, sigma=-1., which="LM", return_eigenvectors=False,
                           v0=np.ones(mesh.n_nodes)))
    largest = float(eigsh(symmetric, k=1, which="LA", return_eigenvectors=False,
                          v0=np.linspace(1, 2, mesh.n_nodes))[0])
    return dict(mesh=mesh.name, seed=mesh.seed, nodes=mesh.n_nodes,
                eigenvalues_m_inv2=values.tolist(), lambda_max_m_inv2=largest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", choices=PATHS, default="straight")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--out", default="results/lei/spectra.json")
    a = ap.parse_args()
    rows = [spectrum(get_mesh(a.mesh, seed), a.modes) for seed in a.seeds]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
