"""Reproduce the LF-T order comparison on the fixed train/val split.

Run from the repository root:
  python -m models.linear_filter_experiment --orders 1 3 5 --stride 5

Each order is fitted independently on all train steps, then scored one step at
a time on val with the true current temperature. This is not a rollout. The
checkpoint for each order goes to checkpoints/ and the CSV goes to results/.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from graphs.dataset import check_stats, load_runs, load_stats
from models.linear_filter import FilterBasis, fit_filter, predict_next


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--report", help="CSV path; defaults to results/linear_filter_orders_s<S>.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    if not args.orders or len(set(args.orders)) != len(args.orders) or any(k < 1 for k in args.orders):
        parser.error("--orders needs distinct positive integers")
    report = args.report or f"results/linear_filter_orders_s{args.stride}.csv"

    stats = load_stats(args.stride)
    train = load_runs("train", preload=False)
    val = load_runs("val", preload=False)
    check_stats(stats, args.stride, val, "linear-filter val runs")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    rows = []
    for order in args.orders:
        start = time.perf_counter()
        fit = fit_filter(train, stats, args.stride, order)
        fit_seconds = time.perf_counter() - start
        checkpoint = os.path.join(args.checkpoint_dir, f"linear_filter_s{args.stride}_k{order}.json")
        with open(checkpoint + ".tmp", "w") as file:
            json.dump(fit, file, indent=2)
        os.replace(checkpoint + ".tmp", checkpoint)
        start = time.perf_counter()
        val_score = score_one_step(val, fit)
        val_seconds = time.perf_counter() - start
        row = dict(stride=args.stride, order=order, train_runs=fit["runs"],
                   train_node_steps=fit["nodes_steps"], val_runs=len(val),
                   val_node_steps=val_score["node_steps"], melt_node_steps=val_score["melt_node_steps"],
                   train_rmse_K=fit["one_step_rmse_K"],
                   train_persistence_rmse_K=fit["persistence_rmse_K"],
                   val_rmse_K=val_score["rmse_K"],
                   val_persistence_rmse_K=val_score["persistence_rmse_K"],
                   val_melt_mae_K=val_score["melt_mae_K"],
                   val_persistence_melt_mae_K=val_score["persistence_melt_mae_K"],
                   gram_condition=fit["gram_condition"], fit_seconds=fit_seconds,
                   val_seconds=val_seconds, train_fingerprint=fit["train_fingerprint"],
                   checkpoint=checkpoint)
        rows.append(row)
        print(f"K={order}: train RMSE {row['train_rmse_K']:.2f} K, val RMSE {row['val_rmse_K']:.2f} K, "
              f"val melt MAE {row['val_melt_mae_K']:.2f} K, fit {fit_seconds:.1f} s, "
              f"val {val_seconds:.1f} s, condition {row['gram_condition']:.2g}", flush=True)

    with open(report + ".tmp", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(report + ".tmp", report)
    print(f"saved {report}")


if __name__ == "__main__":
    main()
