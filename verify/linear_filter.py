"""Check LF-T basis invariants and recover known coefficients from a synthetic run.

Usage: python -m verify.linear_filter
"""

import numpy as np

from graphs.dataset import train_fingerprint
from graphs.mesh import get_mesh
from models.linear_filter import FilterBasis, fit_filter, predict_next


class SyntheticRun:
    def __init__(self, mesh, temperatures, laser):
        self.mesh = mesh
        self.name = "synthetic_train"
        self.attrs = dict(split="train", interp="T", mesh=mesh.name, mesh_seed=mesh.seed,
                          mesh_fp=mesh.fingerprint(), data_hash="synthetic")
        self.frame_us = 13.34
        self._T, self._E = temperatures, laser
        self.last = len(temperatures) - 1

    def _read(self, key, start, stop):
        return (self._T if key == "T" else self._E)[start:stop]


def main():
    mesh = get_mesh("straight", 0)
    basis = FilterBasis(mesh, order=3)
    rng = np.random.default_rng(4)
    T = 300 + 50 * rng.standard_normal(mesh.n_nodes)
    np.testing.assert_allclose(basis.features(np.full(mesh.n_nodes, 300)), 0, atol=1e-11)
    np.testing.assert_allclose(basis.features(T + 80), basis.features(T), rtol=0, atol=1e-10)

    weights = np.array([0.06, -0.02, 0.01, 0.4])
    frames = [T]
    laser = []
    for _ in range(8):
        E = rng.uniform(0, 1, mesh.n_nodes)
        T = T + basis.features(T) @ weights[:-1] + weights[-1] * E
        frames.append(T)
        laser.append(E)
    run = SyntheticRun(mesh, np.stack(frames), np.stack(laser))
    stats = dict(stride=1, frame_us=run.frame_us, interp=["T"],
                 train_fingerprint=train_fingerprint({run.name: run.attrs}),
                 node=dict(T=dict(std=1), E=dict(std=1)), target=dict(dT=dict(std=1)))
    fitted = fit_filter([run], stats, stride=1, order=3)
    np.testing.assert_allclose(fitted["weights"], weights, rtol=0, atol=1e-8)
    np.testing.assert_allclose(predict_next(basis, frames[0], laser[0], fitted), frames[1],
                               rtol=0, atol=1e-8)
    if fitted["one_step_rmse_K"] > 1e-5:
        raise AssertionError("synthetic one-step fit did not recover the exact transitions")
    print("LF-T: constant field, temperature-shift invariance, and coefficient recovery pass")


if __name__ == "__main__":
    main()
