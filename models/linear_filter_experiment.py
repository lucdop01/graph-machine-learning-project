"""Fit LF-T orders 1/3/5/10 on train and select K=5 by val rollout.

Run from the repository root: python -m models.linear_filter_experiment
Writes one comparison CSV and the selected, small checkpoint for the B gate.
No test split is read by this module.
"""

import csv
import json
import os
import time

import numpy as np

from evaluation.rollout import evaluate_split, load_size_bins
from graphs.dataset import check_stats, load_runs, load_stats
from models.linear_filter import (DEFAULT_RIDGE, SELECTED_CHECKPOINT, FilterBasis,
                                  LinearFilterPredictor, fit_filter, predict_next)

ORDERS = (1, 3, 5, 10)
SELECTED_ORDER = 5
STRIDE = 5
REPORT = "results/lf_t_order_comparison_val_s5.csv"


def score_one_step(runs, fit):
    """Pooled one-step temperature errors, including the 1800-2000 K melt band."""
    stride = fit["stride"]
    basis_by_mesh = {}
    squared = baseline_squared = melt_absolute = melt_baseline_absolute = 0.0
    n_nodes = n_melt = 0
    for run in runs:
        key = run.mesh.fingerprint()
        if key not in basis_by_mesh:
            basis_by_mesh[key] = FilterBasis(run.mesh, fit["order"])
        T = run._read("T", 0, run.last + 1).astype(np.float64)
        E_frame = run._read("E_frame", 0, run.last).astype(np.float64)
        E_cumulative = np.concatenate((np.zeros_like(E_frame[:1]),
                                       np.cumsum(E_frame, axis=0)))
        T0, truth = T[:-stride], T[stride:]
        E = E_cumulative[stride:] - E_cumulative[:-stride]
        prediction = predict_next(basis_by_mesh[key], T0, E, fit)
        error = prediction - truth
        baseline = T0 - truth
        squared += float(np.sum(error ** 2))
        baseline_squared += float(np.sum(baseline ** 2))
        n_nodes += error.size
        melt = (truth >= 1800) & (truth <= 2000)
        melt_absolute += float(np.sum(np.abs(error[melt])))
        melt_baseline_absolute += float(np.sum(np.abs(baseline[melt])))
        n_melt += int(np.sum(melt))
    return dict(rmse_K=np.sqrt(squared / n_nodes),
                persistence_rmse_K=np.sqrt(baseline_squared / n_nodes),
                melt_mae_K=melt_absolute / n_melt if n_melt else float("nan"),
                persistence_melt_mae_K=melt_baseline_absolute / n_melt if n_melt else float("nan"),
                node_steps=n_nodes, melt_node_steps=n_melt)


def main():
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=False)
    val = load_runs("val", preload=False)
    check_stats(stats, STRIDE, val, "linear-filter val runs")
    bins = load_size_bins()
    comparison, fits = [], {}
    for order in ORDERS:
        start = time.perf_counter()
        fit = fit_filter(train, stats, STRIDE, order, ridge_relative=DEFAULT_RIDGE)
        fit_seconds = time.perf_counter() - start
        one_step = score_one_step(val, fit)
        _, rollout = evaluate_split(val, LinearFilterPredictor(fit, stats), STRIDE, bins)
        alpha = np.asarray(fit["weights"][:-1]) * fit["scales"]["dT"] / fit["scales"]["T"]
        row = dict(stride=STRIDE, order=order, ridge_relative=DEFAULT_RIDGE,
                   train_runs=fit["runs"], val_runs=len(val),
                   train_one_step_rmse_K=fit["one_step_rmse_K"],
                   val_one_step_rmse_K=one_step["rmse_K"],
                   val_one_step_melt_mae_K=one_step["melt_mae_K"],
                   val_rollout_nrmse=rollout["nrmse_mean"],
                   val_rollout_melt_mae_K=rollout["melt_mae"],
                   val_rollout_iou=rollout["iou"]["iou"],
                   val_rollout_energy_end_median=rollout["energy"]["end"]["median"],
                   raw_filter_coefficients=json.dumps(fit["weights"][:-1]),
                   effective_diffusion=json.dumps(alpha.tolist()),
                   effective_diffusion_sum=float(alpha.sum()),
                   laser_weight=fit["weights"][-1],
                   gram_condition=fit["gram_condition"],
                   penalized_gram_condition=fit["penalized_gram_condition"],
                   fit_seconds=fit_seconds, train_fingerprint=fit["train_fingerprint"])
        comparison.append(row)
        fits[order] = fit
        print(f"K={order}: val rollout nRMSE {row['val_rollout_nrmse']:.4f}, "
              f"melt MAE {row['val_rollout_melt_mae_K']:.1f} K, "
              f"IoU {row['val_rollout_iou']:.4f}", flush=True)

    winner = min(comparison, key=lambda row: row["val_rollout_nrmse"])
    if winner["order"] != SELECTED_ORDER:
        raise RuntimeError(f"expected K={SELECTED_ORDER} to win on this validation split; "
                           f"K={winner['order']} scored best. Review before exporting a checkpoint")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT + ".tmp", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    os.replace(REPORT + ".tmp", REPORT)

    selected = dict(fits[SELECTED_ORDER])
    selected["selection"] = dict(split="val", metric="nrmse_mean", value=winner["val_rollout_nrmse"],
                                 compared_orders=list(ORDERS))
    os.makedirs(os.path.dirname(SELECTED_CHECKPOINT), exist_ok=True)
    with open(SELECTED_CHECKPOINT + ".tmp", "w") as file:
        json.dump(selected, file, indent=2)
    os.replace(SELECTED_CHECKPOINT + ".tmp", SELECTED_CHECKPOINT)
    print(f"selected K={SELECTED_ORDER}; saved {REPORT} and {SELECTED_CHECKPOINT}")


if __name__ == "__main__":
    main()
