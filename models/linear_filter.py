"""Fit the temperature linear-filter baseline on the processed training runs.

The one-step model is T_next = T + sum_k a_k F_k(T) + b E, where F_k is a
Chebyshev polynomial of the mesh Laplacian with its constant response removed.
The laser input E is the energy per volume over the same model step. Coefficients
are fitted by ordinary least squares, with no intercept: a uniform temperature
without laser input must remain stationary.

Usage: python -m models.linear_filter --stride 5 --order 3
The saved checkpoint contains one set of coefficients; no validation or test
data is used for fitting. LinearFilterPredictor connects it to the shared
rollout harness.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

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


class LinearFilterPredictor:
    """LF-T adapter for evaluation.rollout: one mesh operator, own-state feedback."""

    def __init__(self, fit, stats):
        if fit.get("kind") != "LF-T" or fit.get("basis") != "chebyshev_constant_removed":
            raise ValueError("checkpoint is not a compatible LF-T fit")
        if fit.get("train_fingerprint") != stats.get("train_fingerprint"):
            raise ValueError("checkpoint and current training statistics come from different data")
        self.stride = check_stride(fit["stride"])
        if self.stride != stats.get("stride") or not np.isclose(fit["frame_us"], stats["frame_us"], rtol=1e-6):
            raise ValueError("checkpoint and training statistics use different model steps")
        for field, group, key in (("T", "node", "T"), ("E", "node", "E"), ("dT", "target", "dT")):
            if not np.isclose(fit["scales"][field], stats[group][key]["std"], rtol=1e-12):
                raise ValueError(f"checkpoint and training statistics disagree on the {field} scale")
        self.fit, self.stats = fit, stats
        label = "stable_" if fit.get("constraint") == "spectral_convex" else ""
        self.name = f"lf_t_{label}k{fit['order']}"
        self._basis = {}

    def reset(self, run):
        check_stats(self.stats, self.stride, [run], f"run {run.name}")
        key = run.mesh.fingerprint()
        if key not in self._basis:
            self._basis[key] = FilterBasis(run.mesh, self.fit["order"])
        self.current_basis = self._basis[key]

    def step(self, run, T, k, stride):
        if stride != self.stride:
            raise ValueError(f"{self.name} was fitted for stride {self.stride}, not {stride}")
        if not hasattr(self, "current_basis"):
            self.reset(run)
        return predict_next(self.current_basis, T, run.laser(k, stride), self.fit)


def fit_filter(runs, stats, stride=5, order=3, stable=False):
    """Fit one LF-T model from train runs; return its coefficients and diagnostics.

    With stable=True, nonnegative diffusion coefficients sum to at most one.
    The one-step heat operator is then a convex combination of I and the
    Chebyshev operators, whose eigenvalues lie in [-1, 1]. Its spectrum is
    therefore also in [-1, 1], so free diffusion cannot amplify a mode.
    """
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
    if stable:
        # Optimise the same least-squares objective with a spectral stability
        # constraint. The fitted coefficients are dimensionless after multiplying
        # by dT_scale / T_scale; the laser coefficient must remain nonnegative.
        limit = T_scale / dT_scale
        scale = 1e4 / count
        G, h = gram * scale, rhs * scale
        start = np.zeros(order + 1)
        start[-1] = max(0.0, rhs[-1] / gram[-1, -1])
        diffusion_sum = np.r_[np.ones(order), 0.0][None, :]
        solved = minimize(lambda w: 0.5 * w @ G @ w - h @ w, start,
                          jac=lambda w: G @ w - h, hess=lambda w: G,
                          method="trust-constr",
                          bounds=Bounds(np.zeros(order + 1), np.full(order + 1, np.inf)),
                          constraints=[LinearConstraint(diffusion_sum, -np.inf, limit)],
                          options=dict(gtol=1e-9, xtol=1e-12, barrier_tol=1e-12, maxiter=1000))
        if not solved.success:
            raise RuntimeError(f"stable filter fit failed: {solved.message}")
        weights = solved.x
        if weights[:-1].sum() > limit + 1e-7 or np.min(weights) < -1e-7:
            raise RuntimeError("stable filter solver returned infeasible coefficients")
    else:
        weights = np.linalg.solve(gram, rhs)
    residual_sq = max(0.0, target_sq - 2 * weights @ rhs + weights @ gram @ weights)
    return dict(kind="LF-T", basis="chebyshev_constant_removed",
                constraint="spectral_convex" if stable else "none", order=order,
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
    parser.add_argument("--stable", action="store_true", help="constrain free diffusion to be spectrally stable")
    parser.add_argument("--out", help="checkpoint JSON path; defaults to checkpoints/linear_filter_s<S>_k<K>.json")
    args = parser.parse_args()
    stats = load_stats(args.stride)
    runs = load_runs("train", preload=False)
    result = fit_filter(runs, stats, args.stride, args.order, stable=args.stable)
    label = "_stable" if args.stable else ""
    out = args.out or f"checkpoints/linear_filter{label}_s{args.stride}_k{args.order}.json"
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
