"""Full-data GroupNorm U-Net training with validation-rollout early stopping.

Run from the repository root: python3 -m models.train_unet_early_stop
No held-out split is loaded. The existing three-epoch checkpoint is untouched.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.rollout import evaluate_split, load_size_bins, write_results
from graphs.dataset import check_stats, load_runs, load_stats
from models.evaluate_unet import one_step_metrics
from models.train_unet import default_device, train_one_epoch
from models.unet import UNet
from models.unet_pipeline import STRIDE, UNetPredictor, UNetSamples, save_checkpoint

PREVIOUS_CHECKPOINT = Path("checkpoints/unet_full_group_s5_seed0.pt")


def write_history(path, payload):
    """Replace the history only after a complete epoch has been recorded."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(temporary, path)


def save_training_state(path, epoch, model, optimizer, scheduler, generator,
                        history, best_any, best_epoch, significant_anchor,
                        bad_epochs, elapsed_s):
    """Save enough state to restart at the next epoch after an interruption."""
    state = dict(epoch=epoch, model_state=model.state_dict(),
                 optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                 data_generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                 mps_rng_state=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
                 numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(),
                 history=history, best_any=best_any, best_epoch=best_epoch,
                 significant_anchor=significant_anchor, bad_epochs=bad_epochs,
                 elapsed_s=elapsed_s)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def truncate_batch_log(path, completed_epoch):
    """Discard partial batches from an interrupted epoch before replaying it."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if int(row["epoch"]) <= completed_epoch]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("epoch", "step", "batch", "samples", "lr", "loss"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-relative-improvement", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--checkpoint", default="checkpoints/unet_early_stop_group_s5_seed0.pt")
    ap.add_argument("--out-dir", default="results/unet_early_stop")
    ap.add_argument("--resume", action="store_true", help="resume from the last completed epoch in out-dir")
    args = ap.parse_args()
    if (args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1
            or not 0 < args.min_relative_improvement < 1
            or not 0 < args.min_lr < args.lr):
        ap.error("invalid epoch, patience, batch, improvement or LR settings")
    if Path(args.checkpoint).resolve() == PREVIOUS_CHECKPOINT.resolve():
        ap.error("the early-stopping run must not overwrite the three-epoch checkpoint")
    if Path(args.out_dir).resolve() == Path("results/unet_full").resolve():
        ap.error("the early-stopping run must not overwrite the three-epoch results")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=False)
    val = load_runs("val", preload=False)
    check_stats(stats, STRIDE, val, "U-Net early-stopping validation runs")
    model = UNet(norm="group", grid_dx=5e-6).to(device)
    samples = UNetSamples(train, stats, model.grid_dx)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0,
        threshold=args.min_relative_improvement, threshold_mode="rel", min_lr=args.min_lr)
    bins = load_size_bins()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    if args.resume and not state_path.exists():
        ap.error(f"no completed-epoch training state in {output}")
    if not args.resume and Path(args.checkpoint).exists():
        ap.error(f"checkpoint already exists: {args.checkpoint}; choose a fresh path")
    if not args.resume and (output / "history.json").exists():
        ap.error(f"history already exists in {output}; choose a fresh output directory")

    print(f"GroupNorm U-Net: {len(train)} train runs, {len(samples)} samples/epoch, "
          f"{len(val)} validation runs; up to {args.max_epochs} epochs, "
          f"patience {args.patience}, min relative improvement "
          f"{100 * args.min_relative_improvement:g}%, device {device}", flush=True)
    started = time.perf_counter()
    history = []
    best_any = math.inf
    best_epoch = None
    significant_anchor = math.inf
    bad_epochs = 0
    stopping_reason = "max_epochs"
    first_epoch = 1
    if args.resume:
        recorded = json.loads((output / "history.json").read_text())
        for key in ("max_epochs", "patience", "min_relative_improvement", "seed", "batch_size",
                    "lr", "min_lr", "device", "checkpoint", "out_dir"):
            if recorded["config"][key] != vars(args)[key]:
                ap.error(f"resume configuration differs at {key}")
        if recorded["stopping_reason"] in ("early_stopping", "max_epochs"):
            ap.error("this run already completed")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["epoch"] != len(state["history"]):
            raise ValueError("completed-epoch training state has inconsistent history")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        generator.set_state(state["data_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if state["mps_rng_state"] is not None:
            torch.mps.set_rng_state(state["mps_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        history, best_any, best_epoch = state["history"], state["best_any"], state["best_epoch"]
        significant_anchor, bad_epochs = state["significant_anchor"], state["bad_epochs"]
        first_epoch = state["epoch"] + 1
        started = time.perf_counter() - state["elapsed_s"]
        truncate_batch_log(output / "batches.csv", state["epoch"])
        print(f"resumed after epoch {state['epoch']}; replaying from epoch {first_epoch}", flush=True)
    with (output / "batches.csv").open("a" if args.resume else "w", newline="") as batch_file:
        writer = csv.DictWriter(batch_file, fieldnames=("epoch", "step", "batch", "samples", "lr", "loss"))
        if not args.resume:
            writer.writeheader()
        for epoch in range(first_epoch, args.max_epochs + 1):
            epoch_started = time.perf_counter()
            lr_used = optimizer.param_groups[0]["lr"]

            def record(batch, loss, count, lr):
                step = (epoch - 1) * len(loader) + batch
                writer.writerow(dict(epoch=epoch, step=step, batch=batch, samples=count, lr=lr, loss=loss))
                if batch % 500 == 0:
                    batch_file.flush()
                    print(f"epoch {epoch}/{args.max_epochs}: batch {batch}/{len(loader)}, "
                          f"loss {loss:.5g}, lr {lr:.3g}", flush=True)

            train_loss = train_one_epoch(model, loader, samples.adapter, optimizer, device,
                                         on_batch=record)
            batch_file.flush()
            predictor = UNetPredictor(model, stats, device)
            rows, summary = evaluate_split(val, predictor, STRIDE, bins)
            score = float(summary["nrmse_mean"])
            if not math.isfinite(score):
                raise FloatingPointError(f"non-finite validation rollout nRMSE at epoch {epoch}")
            write_results(rows, summary, str(output), f"val_epoch_{epoch:02d}")

            # The best checkpoint uses *any* strict improvement. The 1% rule
            # applies only to the patience counter and the LR scheduler.
            saved_best = score < best_any
            if saved_best:
                best_any, best_epoch = score, epoch
                save_checkpoint(args.checkpoint, model, stats, epoch, train_loss, score, args.seed)
            one_step = one_step_metrics(val, predictor)
            significant = score <= significant_anchor * (1 - args.min_relative_improvement)
            if significant:
                significant_anchor = score
                bad_epochs = 0
            else:
                bad_epochs += 1
            scheduler.step(score)
            if bad_epochs >= args.patience:
                stopping_reason = "early_stopping"

            history.append(dict(epoch=epoch, step=epoch * len(loader), train_loss=train_loss,
                                val_rollout_nrmse=score,
                                val_one_step_rmse_K=one_step["temperature_rmse_K"],
                                val_one_step_normalized_dT_mse=one_step["normalized_dT_mse"],
                                val_melt_iou=summary["iou"]["iou"],
                                val_melt_mae_K=summary["melt_mae"],
                                val_energy_end_median=summary["energy"]["end"]["median"],
                                lr_used=lr_used, lr_next=optimizer.param_groups[0]["lr"],
                                significant_improvement=significant, checkpoint_saved=saved_best,
                                best_epoch=best_epoch, best_nrmse=best_any,
                                bad_epochs=bad_epochs,
                                elapsed_s=time.perf_counter() - epoch_started))
            save_training_state(state_path, epoch, model, optimizer, scheduler, generator,
                                history, best_any, best_epoch, significant_anchor,
                                bad_epochs, time.perf_counter() - started)
            write_history(output / "history.json", dict(config=vars(args),
                          scheduler="ReduceLROnPlateau: factor 0.5 after one epoch without >=1% "
                                    "relative rollout nRMSE improvement; floor 1e-5",
                          train_runs=len(train), train_samples=len(samples), val_runs=len(val),
                          train_fingerprint=stats["train_fingerprint"], history=history,
                          best_epoch=best_epoch, best_nrmse=best_any,
                          stopping_reason=stopping_reason if bad_epochs >= args.patience else "running",
                          total_elapsed_s=time.perf_counter() - started))
            print(f"epoch {epoch}: train {train_loss:.6g}, val rollout nRMSE {score:.6g}, "
                  f"one-step RMSE {one_step['temperature_rmse_K']:.4g} K, "
                  f"melt IoU {summary['iou']['iou']:.4g}, "
                  f"LR {lr_used:.3g}->{optimizer.param_groups[0]['lr']:.3g}, "
                  f"bad epochs {bad_epochs}/{args.patience}, "
                  f"best epoch {best_epoch}; {history[-1]['elapsed_s'] / 60:.1f} min",
                  flush=True)
            if bad_epochs >= args.patience:
                break

    write_history(output / "history.json", dict(config=vars(args),
                  scheduler="ReduceLROnPlateau: factor 0.5 after one epoch without >=1% "
                            "relative rollout nRMSE improvement; floor 1e-5",
                  train_runs=len(train), train_samples=len(samples), val_runs=len(val),
                  train_fingerprint=stats["train_fingerprint"], history=history,
                  best_epoch=best_epoch, best_nrmse=best_any,
                  stopping_reason=stopping_reason,
                  total_elapsed_s=time.perf_counter() - started))
    print(f"stopped: {stopping_reason} after {len(history)} epochs; best epoch {best_epoch}, "
          f"val nRMSE {best_any:.6g}; checkpoint {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
