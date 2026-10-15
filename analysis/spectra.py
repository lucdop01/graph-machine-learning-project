"""Mass-normalised mesh spectrum: python -m analysis.spectra --mesh straight --seeds 0 1."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from graphs.mesh import get_mesh, PATHS
from graphs.laplacian import stiffness
from solver.ti64 import K_CONST, CP_CONST, RHO


def spectrum(mesh, modes=16, step_us=66.7):
    if not 2 <= modes < mesh.n_nodes:
        raise ValueError("modes must be between 2 and node count - 1")
    K = stiffness(mesh)
    inv = diags(1 / np.sqrt(mesh.area))
    symmetric = inv @ K @ inv
    # Small negative shift avoids factoring the singular constant mode.
    values = np.sort(eigsh(symmetric, k=modes, sigma=-1., which="LM", return_eigenvectors=False,
                           v0=np.ones(mesh.n_nodes)))
    largest = float(eigsh(symmetric, k=1, which="LA", return_eigenvectors=False,
                          v0=np.linspace(1, 2, mesh.n_nodes))[0])
    if not np.isfinite(step_us) or step_us <= 0:
        raise ValueError("step_us must be finite and positive")
    alpha, dt = K_CONST / (RHO * CP_CONST), step_us * 1e-6
    off = K.copy()
    off.setdiag(0)
    bound = 2 * (np.asarray(abs(off).sum(axis=1)).ravel() / mesh.area).max()
    n = max(1, int(np.ceil(dt * alpha * bound / .9)))
    frequencies = np.r_[0., np.geomspace(max(values[1], 1e-6), largest, 256)]
    exact = np.exp(-alpha * dt * frequencies)
    discrete = (1 - alpha * dt * frequencies / n) ** n
    return dict(mesh=mesh.name, seed=mesh.seed, nodes=mesh.n_nodes,
                eigenvalues_m_inv2=values.tolist(), lambda_max_m_inv2=largest,
                step_us=step_us, alpha_m2_s=alpha, substeps=n,
                frequency_response=dict(lambda_m_inv2=frequencies.tolist(),
                                        exact_heat_kernel=exact.tolist(), fixed_filter=discrete.tolist(),
                                        max_abs_difference=float(np.max(np.abs(exact - discrete)))),
                interpretation="Homogeneous linear diffusion only; excludes laser source, phase change and learned correction.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", choices=PATHS, default="straight")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--step-us", type=float, default=66.7)
    ap.add_argument("--out", default="results/lei/spectra.json")
    a = ap.parse_args()
    rows = [spectrum(get_mesh(a.mesh, seed), a.modes, a.step_us) for seed in a.seeds]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
