"""Compare fitted LF-T orders with the project's validation rollout harness.

Run from the repository root after fitting and computing size bins:
  python -m evaluation.rollout --size-bins
  python -m models.linear_filter_rollout --split val --stride 5 --orders 1 3 5

Each order uses its saved train-only checkpoint. The model feeds its own last
prediction into the next step. The shared harness computes nRMSE, melt-pool
metrics, solidification and energy. Numerical divergence is recorded as a
failed run, not replaced with clipped predictions. A finite nRMSE above 10 is
marked "runaway" because it is over ten times worse than persistence (~1).
"""

import argparse
import csv
import json
import os

import numpy as np

from evaluation.rollout import evaluate_split, load_size_bins, write_results
from graphs.dataset import check_stride, load_runs, load_stats
from models.linear_filter import LinearFilterPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--mesh-seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default="results/eval")
    parser.add_argument("--report", help="comparison CSV; defaults to results/linear_filter_rollout_<split>_s<S>.csv")
    args = parser.parse_args()
    stride = check_stride(args.stride)
    if not args.orders or len(set(args.orders)) != len(args.orders) or any(k < 1 for k in args.orders):
        parser.error("--orders needs distinct positive integers")
    suffix = f"_seed{args.mesh_seed}" if args.mesh_seed else ""
    report = args.report or f"results/linear_filter_rollout_{args.split}{suffix}_s{stride}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)

    stats = load_stats(stride)
    runs = load_runs(args.split, args.mesh_seed, preload=False)
    bins = load_size_bins()
    results = []
    for order in args.orders:
        checkpoint = os.path.join(args.checkpoint_dir, f"linear_filter_s{stride}_k{order}.json")
        with open(checkpoint) as file:
            fit = json.load(file)
        if fit.get("order") != order:
            raise ValueError(f"{checkpoint} has order {fit.get('order')}, expected {order}")
        predictor = LinearFilterPredictor(fit, stats)
        stem = f"{predictor.name}_{args.split}{suffix}_s{stride}"
        result = dict(order=order, status="ok", nrmse_mean="", melt_mae_K="", iou="",
                      energy_end_median="", steps="", predictor_seconds="", error="",
                      train_fingerprint=fit["train_fingerprint"], checkpoint=checkpoint)
        try:
            with np.errstate(over="raise", invalid="raise"):
                rows, summary = evaluate_split(runs, predictor, stride, bins)
            summary["mesh_seed"] = args.mesh_seed
            summary["checkpoint"] = checkpoint
            summary["train_fingerprint"] = fit["train_fingerprint"]
            write_results(rows, summary, args.out, stem)
            result.update(nrmse_mean=summary["nrmse_mean"], melt_mae_K=summary.get("melt_mae", ""),
                          iou=summary.get("iou", {}).get("iou", ""),
                          energy_end_median=summary["energy"]["end"]["median"],
                          steps=summary["steps"], predictor_seconds=summary["wall_s"])
            if result["nrmse_mean"] > 10:
                result["status"] = "runaway"
            print(f"K={order}: {result['status']}, rollout nRMSE {result['nrmse_mean']:.4g}, "
                  f"melt MAE {result['melt_mae_K']:.4g} K, IoU {result['iou']:.4f}; "
                  f"saved {args.out}/{stem}.csv/.json", flush=True)
        except FloatingPointError as error:
            result.update(status="diverged", error=str(error))
            print(f"K={order}: rollout diverged: {error}", flush=True)
        results.append(result)

    with open(report + ".tmp", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    os.replace(report + ".tmp", report)
    print(f"saved {report}")


if __name__ == "__main__":
    main()
