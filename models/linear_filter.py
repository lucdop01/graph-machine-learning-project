"""Area-lumped Laplacian filter, constrained Ridge fit, and rollout predictor.

T_next = T + sum_k a_k (Chebyshev_k(B) - I) T + b E. The polynomial is a
scaled representation of powers of the mesh Laplacian. Fit coefficients use
train data only; the constraint keeps free-diffusion spectral gain at most one.
"""

import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from graphs.dataset import (check_stats, check_stride, load_runs, load_stats,
                            train_fingerprint)
from graphs.laplacian import cotangent_laplacian

DEFAULT_RIDGE = 1e-4
SELECTED_CHECKPOINT = "models/artifacts/lf_t_s5_k5.json"


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

    def __init__(self, fit, stats, name=None):
        if (fit.get("kind") != "LF-T" or fit.get("basis") != "chebyshev_constant_removed"
                or fit.get("constraint") != "spectral_convex" or fit.get("ridge_relative", 0) <= 0):
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
        self.name = name or f"lf_t_k{fit['order']}"
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


def load_predictor(checkpoint=SELECTED_CHECKPOINT, name="lf_t"):
    """Load the selected train-only fit for the shared rollout harness."""
    with open(checkpoint) as file:
        fit = json.load(file)
    return LinearFilterPredictor(fit, load_stats(fit["stride"]), name=name)


def fit_filter(runs, stats, stride=5, order=5, ridge_relative=DEFAULT_RIDGE):
    """Fit constrained Ridge least squares on train; return checkpoint data.

    Effective diffusion weights are nonnegative and sum to at most one. The
    laser coefficient is nonnegative and is excluded from the Ridge penalty.
    """
    if not np.isfinite(ridge_relative) or ridge_relative <= 0:
        raise ValueError("ridge strength must be finite and positive")
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
    if not np.isfinite(condition):
        raise ValueError("filter design has a non-finite X'X condition")
    penalized_gram = gram.copy()
    strength = ridge_relative * float(np.mean(np.diag(gram)[:order]))
    penalized_gram[np.arange(order), np.arange(order)] += strength
    penalized_condition = float(np.linalg.cond(penalized_gram))
    if not np.isfinite(penalized_condition) or penalized_condition > 1e12:
        raise ValueError(f"Ridge fit is still ill-conditioned (X'X condition {penalized_condition:.2g})")
    # Dimensionless fit weights become physical diffusion weights after
    # multiplying by dT_scale / T_scale. Their sum must stay at most one.
    limit = T_scale / dT_scale
    scale = 1e4 / count
    G, h = penalized_gram * scale, rhs * scale
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
        raise RuntimeError(f"constrained Ridge fit failed: {solved.message}")
    weights = solved.x
    if weights[:-1].sum() > limit + 1e-7 or np.min(weights) < -1e-7:
        raise RuntimeError("constrained Ridge fit returned infeasible coefficients")
    residual_sq = max(0.0, target_sq - 2 * weights @ rhs + weights @ gram @ weights)
    return dict(kind="LF-T", basis="chebyshev_constant_removed",
                constraint="spectral_convex", order=order,
                stride=stride, frame_us=float(stats["frame_us"]),
                train_fingerprint=fingerprint,
                scales=dict(T=T_scale, E=E_scale, dT=dT_scale),
                weights=weights.tolist(), runs=len(runs), nodes_steps=count,
                gram_condition=condition,
                one_step_rmse_K=float(dT_scale * np.sqrt(residual_sq / count)),
                persistence_rmse_K=float(dT_scale * np.sqrt(target_sq / count)),
                ridge_relative=ridge_relative,
                penalized_gram_condition=penalized_condition)
