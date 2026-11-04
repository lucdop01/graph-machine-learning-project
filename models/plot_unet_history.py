"""Recreate the selected U-Net training and validation plots from saved CSV/JSON."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_temporal(directory, training, device):
    """Rebuild the validation rollout trace when plotting a reproduced run."""
    path = directory / "temporal_best.csv"
    if path.exists():
        return path
    from evaluation.rollout import rollout
    from graphs.dataset import load_runs
    from models.unet_pipeline import STRIDE, load_checkpoint

    predictor, _ = load_checkpoint(training["config"]["checkpoint"], device)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("run", "output_step", "frame", "progress", "rmse_K"))
        writer.writeheader()
        for run in load_runs("val", preload=False):
            predicted = rollout(run, predictor, STRIDE)
            error = np.sqrt(np.mean((predicted["T"] - run.T_frames(predicted["frames"])) ** 2, axis=1))
            for j, (frame, value) in enumerate(zip(predicted["frames"], error), start=1):
                writer.writerow(dict(run=run.name, output_step=j, frame=int(frame),
                                     progress=j / len(error), rmse_K=float(value)))
    return path


def plot_history(directory, device="cpu"):
    training = json.loads((directory / "history.json").read_text())
    history = training["history"]
    with (directory / "batches.csv").open(newline="") as f:
        batches = list(csv.DictReader(f))
    steps = np.asarray([int(row["step"]) for row in batches])
    losses = np.asarray([float(row["loss"]) for row in batches])
    width = min(200, len(losses))
    smooth = np.convolve(losses, np.ones(width) / width, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps[width - 1:], smooth)
    axes[0].set(xlabel="optimizer step", ylabel=f"{width}-batch mean normalized node ΔT MSE", yscale="log")
    axes[1].plot([row["epoch"] for row in history],
                 [row["val_rollout_nrmse"] for row in history], marker="o")
    axes[1].set(xlabel="epoch", ylabel="mean validation rollout nRMSE", yscale="log")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(directory / "learning.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [row["epoch"] for row in history]
    axes[0].plot(x, [row["val_one_step_rmse_K"] for row in history], marker="o")
    axes[0].set(xlabel="epoch", ylabel="validation one-step RMSE (K)")
    axes[1].plot(x, [row["val_melt_iou"] for row in history], marker="o")
    axes[1].set(xlabel="epoch", ylabel="pooled validation melt-pool IoU", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(directory / "validation_metrics.png", dpi=150)
    plt.close(fig)

    with ensure_temporal(directory, training, device).open(newline="") as f:
        temporal = list(csv.DictReader(f))
    by_run_bin = defaultdict(list)
    for row in temporal:
        index = min(int(float(row["progress"]) * 10), 9)
        by_run_bin[row["run"], index].append(float(row["rmse_K"]))
    values = [np.mean([np.mean(group) for (run, j), group in by_run_bin.items() if j == i])
              for i in range(10)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot((np.arange(10) + .5) / 10, values, marker="o")
    ax.set(xlabel="fraction of closed-loop validation rollout", ylabel="mean per-run RMSE (K)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(directory / "rollout_error.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/unet_early_stop_retry"))
    parser.add_argument("--device", default="cpu", help="used only if the rollout trace is absent")
    args = parser.parse_args()
    plot_history(args.out_dir, args.device)
    print(f"wrote three plots in {args.out_dir}")


if __name__ == "__main__":
    main()
