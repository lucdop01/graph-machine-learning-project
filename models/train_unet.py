"""Train grid U-Net on mesh-node dT and select by shared val rollout nRMSE.

Run only after choosing the training budget:
python -m models.train_unet --epochs <agreed_epochs> --seed 0
No test split is used by this script.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.rollout import selection_score
from graphs.dataset import check_stats, load_runs, load_stats
from models.unet import UNet
from models.unet_pipeline import (STRIDE, UNetPredictor, UNetSamples,
                                  normalized_node_mse, save_checkpoint)


def train_one_epoch(model, loader, adapter, optimizer, device, scheduler=None, on_batch=None):
    model.train()
    total, samples = 0.0, 0
    for batch_number, (grid, target) in enumerate(loader, start=1):
        grid, target = grid.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_node_mse(model(grid), target, adapter)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite U-Net loss at batch {batch_number}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        if on_batch is not None:
            on_batch(batch_number, value, len(grid), lr)
        total += value * len(grid)
        samples += len(grid)
    return total / samples


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, help="cosine decay to this LR over all training steps")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--channels", type=int, nargs="+")
    parser.add_argument("--grid-dx-um", type=float, default=5.0)
    parser.add_argument("--norm", choices=("group", "pixelwise_layer"), default="group")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir", help="write batch losses and epoch validation history here")
    parser.add_argument("--preload", action="store_true", help="keep the train run frames in CPU memory")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0
            or args.min_lr is not None and (not np.isfinite(args.min_lr) or not 0 < args.min_lr < args.lr)):
        parser.error("epochs, batch size and learning rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=args.preload)
    val = load_runs("val", preload=False)
    check_stats(stats, STRIDE, val, "U-Net val runs")
    model = UNet(depth=args.depth, base_channels=args.base_channels, channels=args.channels,
                 grid_dx=args.grid_dx_um * 1e-6, norm=args.norm).to(device)
    samples = UNetSamples(train, stats, model.grid_dx)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader),
                                                           eta_min=args.min_lr)
                 if args.min_lr is not None else None)
    checkpoint = args.checkpoint or f"checkpoints/unet_s{STRIDE}_seed{args.seed}.pt"
    print(f"U-Net: {model.parameter_count:,} parameters, grid {samples.adapter.shape} at {args.grid_dx_um:g} um, "
          f"{len(samples)} train samples, {len(val)} val runs, device {device}", flush=True)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    history = []
    best = float("inf")
    started = time.perf_counter()
    batch_path = os.path.join(args.out_dir, "batches.csv") if args.out_dir else os.devnull
    with open(batch_path, "w", newline="") as batch_file:
        writer = csv.DictWriter(batch_file, fieldnames=("epoch", "step", "batch", "samples", "lr", "loss"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()

            def record(batch, loss, count, lr):
                step = (epoch - 1) * len(loader) + batch
                writer.writerow(dict(epoch=epoch, step=step, batch=batch, samples=count, lr=lr, loss=loss))
                if batch % 500 == 0:
                    batch_file.flush()
                    print(f"epoch {epoch}/{args.epochs}: batch {batch}/{len(loader)}, "
                          f"loss {loss:.5g}, lr {lr:.3g}", flush=True)

            train_loss = train_one_epoch(model, loader, samples.adapter, optimizer, device,
                                         scheduler=scheduler, on_batch=record)
            batch_file.flush()
            predictor = UNetPredictor(model, stats, device)
            val_nrmse = selection_score(val, predictor, STRIDE)
            improved = val_nrmse < best
            if improved:
                best = val_nrmse
                save_checkpoint(checkpoint, model, stats, epoch, train_loss, val_nrmse, args.seed)
            history.append(dict(epoch=epoch, step=epoch * len(loader), train_loss=train_loss,
                                val_rollout_nrmse=val_nrmse,
                                lr_end=optimizer.param_groups[0]["lr"],
                                elapsed_s=time.perf_counter() - epoch_started,
                                checkpoint_saved=improved))
            if args.out_dir:
                with open(os.path.join(args.out_dir, "history.json"), "w") as f:
                    json.dump(dict(config=vars(args), train_runs=len(train), train_samples=len(samples),
                                   val_runs=len(val), train_fingerprint=stats["train_fingerprint"],
                                   device=str(device), history=history,
                                   total_elapsed_s=time.perf_counter() - started), f, indent=2)
            print(f"epoch {epoch}: train normalized node dT MSE {train_loss:.6g}, "
                  f"val rollout nRMSE {val_nrmse:.6g}, elapsed {history[-1]['elapsed_s']:.1f} s"
                  f"{' (saved)' if improved else ''}", flush=True)
    print(f"best val rollout nRMSE {best:.6g}; checkpoint {checkpoint}")


if __name__ == "__main__":
    main()
