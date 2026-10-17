"""Evaluate the selected LF-T checkpoint with the shared rollout harness.

Run from the repository root: python -m models.linear_filter_rollout --split val
The default checkpoint is the train-fitted K=5 model selected on val. The
predictor receives its own previous temperature, never the stored truth.
"""

import argparse
import os

import numpy as np

from evaluation.rollout import evaluate_split, load_size_bins, print_summary, write_results
from graphs.dataset import load_runs
from models.linear_filter import SELECTED_CHECKPOINT, load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mesh-seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=SELECTED_CHECKPOINT)
    parser.add_argument("--out", default="results/eval")
    args = parser.parse_args()

    predictor = load_predictor(args.checkpoint)
    runs = load_runs(args.split, args.mesh_seed, preload=False)
    bins = load_size_bins()
    with np.errstate(over="raise", invalid="raise"):
        rows, summary = evaluate_split(runs, predictor, predictor.stride, bins)
    summary.update(mesh_seed=args.mesh_seed, checkpoint=args.checkpoint,
                   train_fingerprint=predictor.fit["train_fingerprint"])
    suffix = f"_seed{args.mesh_seed}" if args.mesh_seed else ""
    stem = f"{predictor.name}_{args.split}{suffix}_s{predictor.stride}"
    write_results(rows, summary, args.out, stem)
    print_summary(summary)
    print(f"wrote {os.path.join(args.out, stem)}.csv and .json")


if __name__ == "__main__":
    main()
