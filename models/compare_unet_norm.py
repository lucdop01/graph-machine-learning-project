"""Reproducible, validation-only pilot comparing the two U-Net normalizations.

Run from the repository root: python -m models.compare_unet_norm
The fixed-budget pilot writes JSON/CSV/plots under results/unet_norm_comparison
and best checkpoints under checkpoints/. No held-out test run is loaded.
"""

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.rollout import (evaluate_split, load_size_bins, rollout,
                                selection_score, write_results)
from graphs.dataset import check_stats, load_runs, load_stats
from models.evaluate_unet import one_step_metrics
from models.train_unet import default_device
from models.unet import UNet
from models.unet_pipeline import (STRIDE, UNetPredictor,
                                  UNetSamples, normalized_node_mse, save_checkpoint)
from verify.unet_norm import hotspot_difference, padding_difference

NORMS = ("group", "pixelwise_layer")
OUT = Path("results/unet_norm_comparison")


def fixed_training_data(runs, stats):
    """Six train-only P/v quantiles, with 8 laser-on and 4 cooling starts each."""
    ranked = sorted(runs, key=lambda r: float(r.attrs["P"]) / float(r.attrs["v"]))
    selected = [ranked[i] for i in np.linspace(0, len(ranked) - 1, 6).round().astype(int)]
    dataset = UNetSamples(selected, stats)
    items, provenance = [], []
    for i, run in enumerate(selected):
        last = run.last - STRIDE
        off = min(int(run.attrs["laser_off"] / run.t[1]), last)
        starts = sorted(set(np.r_[np.linspace(0, off, 8), np.linspace(off, last, 4)].round().astype(int)))
        for k in starts:
            items.append(dataset[dataset.index.index((i, int(k)))])
            provenance.append(dict(run=run.name, frame=int(k), P=float(run.attrs["P"]),
                                   v=float(run.attrs["v"]), P_over_v=float(run.attrs["P"] / run.attrs["v"])))
    grids = torch.stack([x for x, _ in items])
    targets = torch.stack([y for _, y in items])
    return grids, targets, dataset.adapter, provenance


def temporal_diagnostics(runs, predictor):
    """Closed-loop RMSE in K at each output step, with per-run progress bins."""
    records = []
    for run in runs:
        ro = rollout(run, predictor, STRIDE)
        true = run.T_frames(ro["frames"])
        rmse = np.sqrt(np.mean((ro["T"] - true) ** 2, axis=1))
        for j, (frame, value) in enumerate(zip(ro["frames"], rmse)):
            records.append(dict(run=run.name, output_step=j + 1, frame=int(frame),
                                progress=(j + 1) / len(rmse), rmse_K=float(value)))
    return records


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_plot(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for norm, history in histories.items():
        points = history["train"]
        axes[0].plot([r["step"] for r in points], [r["loss"] for r in points], label=norm)
        points = history["selection"]
        axes[1].plot([r["step"] for r in points], [r["val_rollout_nrmse"] for r in points],
                     marker="o", label=norm)
    axes[0].set(xlabel="optimizer step", ylabel="node normalized dT MSE", yscale="log")
    axes[1].set(xlabel="optimizer step", ylabel="mean val rollout nRMSE")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning.png", dpi=150)
    plt.close(fig)


def temporal_plot(temporal):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 11)
    for norm, rows in temporal.items():
        per_run = {}
        for row in rows:
            i = min(int(row["progress"] * 10), 9)
            per_run.setdefault((row["run"], i), []).append(row["rmse_K"])
        means = [np.mean([np.mean(v) for (run, j), v in per_run.items() if j == i]) for i in range(10)]
        ax.plot((bins[:-1] + bins[1:]) / 2, means, marker="o", label=norm)
    ax.set(xlabel="fraction of rollout", ylabel="mean per-run RMSE (K)")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "rollout_error.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or args.steps % args.eval_every
            or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0):
        ap.error("steps must be a positive multiple of eval-every; batch size and LR must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=False)
    val = load_runs("val", preload=False)
    check_stats(stats, STRIDE, val, "normalization comparison val runs")
    grids, targets, adapter, provenance = fixed_training_data(train, stats)
    write_csv(OUT / "train_subset.csv", provenance)
    bins = load_size_bins()
    order = np.random.default_rng(args.seed).integers(0, len(grids), size=(args.steps, args.batch_size))
    device = torch.device(args.device)
    histories, outcomes, temporal = {}, {}, {}
    initial_convolutions = None
    print(f"Pilot: {len(provenance)} samples from {len(set(p['run'] for p in provenance))} train runs; "
          f"{len(val)} val runs, {args.steps} steps/variant, {device}", flush=True)
    for norm in NORMS:
        torch.manual_seed(args.seed)  # convolutional weights start identically
        model = UNet(norm=norm).to(device)
        convolution_weights = {name: (layer.weight.detach().cpu().clone(),
                                      layer.bias.detach().cpu().clone())
                               for name, layer in model.named_modules()
                               if isinstance(layer, torch.nn.Conv2d)}
        if initial_convolutions is None:
            initial_convolutions = convolution_weights
        elif any(not torch.equal(a, b) for name in initial_convolutions
                 for a, b in zip(initial_convolutions[name], convolution_weights[name])):
            raise AssertionError("the two variants did not receive identical initial convolution weights")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history = dict(train=[], selection=[])
        before = hotspot_difference(model, stats["target"]["dT"]["std"])
        padding_before = padding_difference(model, stats["target"]["dT"]["std"])
        best, best_step = float("inf"), None
        start = time.perf_counter()
        peak_driver = 0
        nonfinite = 0
        checkpoint = f"checkpoints/unet_norm_pilot_{norm}_s{STRIDE}_seed{args.seed}.pt"
        for step, indices in enumerate(order, start=1):
            model.train()
            x, target = grids[indices].to(device), targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_node_mse(model(x), target, adapter)
            if not torch.isfinite(loss):
                nonfinite += 1
                raise FloatingPointError(f"{norm}: non-finite train loss at step {step}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                nonfinite += 1
                raise FloatingPointError(f"{norm}: non-finite gradient at step {step}")
            optimizer.step()
            history["train"].append(dict(step=step, loss=float(loss.detach().cpu())))
            if device.type == "mps":
                peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
            if step % args.eval_every == 0:
                model.eval()
                score = selection_score(val, UNetPredictor(model, stats, device), STRIDE)
                history["selection"].append(dict(step=step, val_rollout_nrmse=score))
                if score < best:
                    best, best_step = score, step
                    save_checkpoint(checkpoint, model, stats, step, float(loss.detach().cpu()), score, args.seed)
                print(f"{norm}: step {step}/{args.steps}, train loss {float(loss.detach()):.5g}, "
                      f"val rollout nRMSE {score:.5g}; best {best:.5g} at {best_step}", flush=True)
        train_seconds = time.perf_counter() - start
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state"])
        model.eval()
        after = hotspot_difference(model, stats["target"]["dT"]["std"])
        padding_after = padding_difference(model, stats["target"]["dT"]["std"])
        predictor = UNetPredictor(model, stats, device)
        one = one_step_metrics(val, predictor)
        rows, summary = evaluate_split(val, predictor, STRIDE, bins)
        write_results(rows, summary, str(OUT), f"{norm}_val")
        temporal[norm] = temporal_diagnostics(val, predictor)
        write_csv(OUT / f"{norm}_temporal.csv", temporal[norm])
        nrmse = np.array([r["nrmse"] for r in rows])
        outcomes[norm] = dict(parameters=model.parameter_count, checkpoint=checkpoint,
                              best_step=best_step, best_val_selection_nrmse=best,
                              train_seconds=train_seconds, sampled_peak_mps_driver_bytes=peak_driver or None,
                              nonfinite_steps=nonfinite, one_step=one, summary=summary,
                              nrmse_sd=float(nrmse.std(ddof=1)),
                              worst_runs=sorted(((r["run"], r["nrmse"]) for r in rows),
                                                key=lambda r: r[1], reverse=True)[:3],
                              hotspot_before=before, hotspot_after=after,
                              padding_before=padding_before, padding_after=padding_after)
        histories[norm] = history
        print(f"{norm}: complete, best step {best_step}, val nRMSE {summary['nrmse_mean']:.5g}", flush=True)
        del model, optimizer, predictor
        if device.type == "mps":
            torch.mps.empty_cache()
    training_plot(histories)
    temporal_plot(temporal)
    with (OUT / "pilot.json").open("w") as f:
        json.dump(dict(protocol=dict(seed=args.seed, steps=args.steps, eval_every=args.eval_every,
                                     batch_size=args.batch_size, lr=args.lr, optimizer="Adam",
                                     schedule="constant", stride=STRIDE, grid_dx_um=5,
                                     train_runs=len(train), pilot_runs=len(set(p["run"] for p in provenance)),
                                     pilot_samples=len(provenance), val_runs=len(val),
                                     train_fingerprint=stats["train_fingerprint"],
                                     device=str(device), platform=platform.platform(),
                                     torch=torch.__version__),
                       histories=histories, outcomes=outcomes), f, indent=2, default=float)
    print(f"Saved {OUT / 'pilot.json'}", flush=True)


if __name__ == "__main__":
    main()
