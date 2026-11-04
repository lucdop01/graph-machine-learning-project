"""Evaluate a frozen U-Net with the shared closed-loop and one-step metrics."""

import argparse
import hashlib
from pathlib import Path

import torch

from evaluation.rollout import (SPLITS, evaluate_split, load_size_bins,
                                print_summary, write_results)
from graphs.dataset import load_runs
from models.unet_pipeline import STRIDE, load_checkpoint


def one_step_metrics(runs, predictor):
    """Pooled node RMSE over every valid start frame, using true current T.

    This is a separate diagnostic; closed-loop evaluation never reads future T
    to build a model input. One example at a time also covers mixed mesh sizes.
    """
    predictor.model.eval()
    squared_K = squared_norm = nodes = examples = 0
    builder = predictor.builder
    with torch.no_grad():
        for run in runs:
            predictor.reset(run)
            adapter = builder.adapter(run)
            for k in run.starts(STRIDE):
                temperature = run.T(k)
                grid = builder.input_grid(run, temperature, k)[None].to(predictor.device)
                target = builder.target_nodes(temperature, run.T(k + STRIDE)).to(predictor.device)
                predicted = adapter.grid_to_nodes(predictor.model(grid))[0, 0]
                error = predicted - target
                squared_norm += float(error.square().sum().cpu())
                squared_K += float((error * builder.dT["std"]).square().sum().cpu())
                nodes += error.numel()
                examples += 1
    return dict(temperature_rmse_K=(squared_K / nodes) ** 0.5,
                normalized_dT_mse=squared_norm / nodes,
                samples=examples, node_comparisons=nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--mesh-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/eval")
    args = parser.parse_args()

    torch.set_num_threads(1)
    predictor, checkpoint = load_checkpoint(args.checkpoint, args.device)
    runs = load_runs(args.split, args.mesh_seed, preload=False)
    partners = (load_runs("test_interp", args.mesh_seed, preload=False)
                if args.split == "test_fine" else None)
    rows, summary = evaluate_split(runs, predictor, STRIDE, load_size_bins(), partners)
    summary["one_step"] = one_step_metrics(runs, predictor)
    summary.update(mesh_seed=args.mesh_seed, checkpoint=args.checkpoint,
                   checkpoint_epoch=checkpoint["epoch"],
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   train_fingerprint=checkpoint["train_fingerprint"])
    suffix = f"_seed{args.mesh_seed}" if args.mesh_seed else ""
    stem = f"unet_{args.split}{suffix}_s{STRIDE}"
    write_results(rows, summary, args.out, stem)
    print_summary(summary)
    print(f"  one-step RMSE {summary['one_step']['temperature_rmse_K']:.4g} K "
          f"over {summary['one_step']['samples']} examples")
    print(f"wrote {Path(args.out) / (stem + '.csv')} and {Path(args.out) / (stem + '.json')}")


if __name__ == "__main__":
    main()
