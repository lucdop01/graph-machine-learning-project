"""Fit the temperature linear-filter baseline on the processed training runs.

The one-step model is T_next = T + sum_k a_k F_k(T) + b E, where F_k is a
Chebyshev polynomial of the mesh Laplacian with its constant response removed.
The laser input E is the energy per volume over the same model step. Coefficients
are fitted by ordinary least squares, with no intercept: a uniform temperature
without laser input must remain stationary.

Usage: python -m models.linear_filter --stride 5 --order 3
The saved checkpoint contains one set of coefficients; no validation or test
data is used here. Rollout evaluation is the next implementation step.
"""

import argparse
import json
import os

import numpy as np

from graphs.dataset import (check_stats, check_stride, load_runs, load_stats,
                            train_fingerprint)
from graphs.laplacian import cotangent_laplacian


class FilterBasis:
    """Stable degree-1..K diffusion features for one triangular mesh.

    If L is the physical Laplacian, B = I + 2 L / radius has eigenvalues in
    [-1, 1]. The maximum absolute row sum is a safe spectral-radius bound.
    Feature k is (Chebyshev_k(B) - I) T, which is zero for a constant field.
    """

    def __init__(self, mesh, order):
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("filter order must be a positive integer")
        self.order = order
        self.laplacian = cotangent_laplacian(mesh)
        self.radius = float(np.max(np.asarray(abs(self.laplacian).sum(axis=1))))
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("mesh Laplacian has no finite positive spectral bound")
        self.n_nodes = mesh.n_nodes

    def _apply_scaled(self, values):
        """Apply B to (steps, nodes) without forming a new sparse matrix."""
        return values + (2.0 / self.radius) * (self.laplacian @ values.T).T

    def features(self, temperature):
        """Return (steps, nodes, order), or (nodes, order) for a single state."""
        value = np.asarray(temperature, dtype=np.float64)
        single = value.ndim == 1
        if single:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != self.n_nodes:
            raise ValueError(f"temperature must have {self.n_nodes} nodes")

        # Removing the mean reduces roundoff; each feature is unchanged because
        # the Laplacian annihilates constants.
        base = value - value.mean(axis=1, keepdims=True)
        previous = base
        current = self._apply_scaled(base)
        out = [current - base]
        for _ in range(2, self.order + 1):
            following = 2 * self._apply_scaled(current) - previous
            out.append(following - base)
            previous, current = current, following
        result = np.stack(out, axis=-1)
        return result[0] if single else result


def predict_next(basis, temperature, laser, fit):
    """Apply fitted LF-T coefficients to one or several true or predicted states."""
    if basis.order != fit["order"]:
        raise ValueError(f"basis order {basis.order} differs from fitted order {fit['order']}")
    T = np.asarray(temperature, dtype=np.float64)
    E = np.asarray(laser, dtype=np.float64)
    if E.shape != T.shape:
        raise ValueError("laser energy must have the same shape as temperature")
    weights = np.asarray(fit["weights"], dtype=np.float64)
    if weights.shape != (basis.order + 1,):
        raise ValueError("fitted weights must contain one value per filter order plus laser")
    scales = fit["scales"]
    change = (basis.features(T) @ weights[:-1] / scales["T"]
              + weights[-1] * E / scales["E"]) * scales["dT"]
    return T + change


def fit_filter(runs, stats, stride=5, order=3):
    """Fit one LF-T model from train runs; return its coefficients and diagnostics."""
    stride = check_stride(stride)
    if not runs or any(run.attrs["split"] != "train" for run in runs):
        raise ValueError("the linear filter must be fitted on train runs only")
    check_stats(stats, stride, runs, "linear-filter train runs")
    fingerprint = train_fingerprint({run.name: run.attrs for run in runs})
    if stats.get("train_fingerprint") != fingerprint:
        raise ValueError("training runs do not match the saved training statistics")

    T_scale = float(stats["node"]["T"]["std"])
    E_scale = float(stats["node"]["E"]["std"])
    dT_scale = float(stats["target"]["dT"]["std"])
    if not all(np.isfinite(s) and s > 0 for s in (T_scale, E_scale, dT_scale)):
        raise ValueError("training statistics need positive finite T, E and dT scales")

    gram = np.zeros((order + 1, order + 1), dtype=np.float64)
    rhs = np.zeros(order + 1, dtype=np.float64)
    target_sq = 0.0
    count = 0
    basis_by_mesh = {}
    for run in runs:
        key = run.mesh.fingerprint()
        if key not in basis_by_mesh:
            basis_by_mesh[key] = FilterBasis(run.mesh, order)
        basis = basis_by_mesh[key]
        # Read one run at a time. Starts include every saved frame that leaves
        # a full model step, matching MeltPoolGraphs and evaluation.rollout.
        T = run._read("T", 0, run.last + 1).astype(np.float64)
        E_frame = run._read("E_frame", 0, run.last).astype(np.float64)
        E_cumulative = np.concatenate((np.zeros_like(E_frame[:1]),
                                       np.cumsum(E_frame, axis=0)))
        T0, T1 = T[:-stride], T[stride:]
        E = E_cumulative[stride:] - E_cumulative[:-stride]
        feature = basis.features(T0).reshape(-1, order) / T_scale
        design = np.column_stack((feature, E.reshape(-1) / E_scale))
        target = ((T1 - T0) / dT_scale).reshape(-1)
        gram += design.T @ design
        rhs += design.T @ target
        target_sq += target @ target
        count += target.size

    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"filter design is ill-conditioned (X'X condition {condition:.2g}); try a smaller order")
    weights = np.linalg.solve(gram, rhs)
    residual_sq = max(0.0, target_sq - 2 * weights @ rhs + weights @ gram @ weights)
    return dict(kind="LF-T", basis="chebyshev_constant_removed", order=order,
                stride=stride, frame_us=float(stats["frame_us"]),
                train_fingerprint=fingerprint,
                scales=dict(T=T_scale, E=E_scale, dT=dT_scale),
                weights=weights.tolist(), runs=len(runs), nodes_steps=count,
                gram_condition=condition,
                one_step_rmse_K=float(dT_scale * np.sqrt(residual_sq / count)),
                persistence_rmse_K=float(dT_scale * np.sqrt(target_sq / count)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--out", help="checkpoint JSON path; defaults to checkpoints/linear_filter_s<S>_k<K>.json")
    args = parser.parse_args()
    stats = load_stats(args.stride)
    runs = load_runs("train", preload=False)
    result = fit_filter(runs, stats, args.stride, args.order)
    out = args.out or f"checkpoints/linear_filter_s{args.stride}_k{args.order}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as file:
        json.dump(result, file, indent=2)
    os.replace(out + ".tmp", out)
    print(f"saved {out}: {result['runs']} train runs, {result['nodes_steps']} node-steps, "
          f"one-step RMSE {result['one_step_rmse_K']:.3f} K "
          f"(persistence {result['persistence_rmse_K']:.3f} K), "
          f"X'X condition {result['gram_condition']:.3g}")


if __name__ == "__main__":
    main()
