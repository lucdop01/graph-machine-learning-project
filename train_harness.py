"""Unified training harness for all melt-pool GNN architectures and baselines.

Supports:
  - Model registry for lazy/dynamic loading of models (Model A, B, C, D, GCN, UNet, LinearFilter)
  - Resuming runs from checkpoints/<name>/last.pt
  - Verification of GPU-CPU transfers (preventing stale host reads and corrupt Adam states)
  - Single-run training, multi-seed runs, and fixed-budget hyperparameter sweeps
  - Verification of non-finite training loss / gradient norms and model weights
  - Atomic checkpoint writing with Windows file lock retries

Usage:
  python train_harness.py --config configs/model_a.yaml
  python train_harness.py --config configs/model_a.yaml --sweep --trials 6
  python train_harness.py --config configs/model_a.yaml --multi-seed --seeds 0 1 2
  python train_harness.py --resume --name model_a_run
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import math
import os
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from evaluation.rollout import load_size_bins, selection_score
from evaluation.torch_predictor import (
    TorchPredictor,
    check_on_device,
    move_checked,
    to_host,
    tree_to_host,
)
from graphs.dataset import (
    MeltPoolGraphs,
    check_stats,
    check_stride,
    load_runs,
    load_stats,
)

CKPT_DIR = "checkpoints"
SWEEP_DIR = os.path.join("results", "sweeps")
LOG_FIELDS: tuple[str, ...] = (
    "step", "lr", "train_loss", "grad_norm", "grad_norm_max", "ms_per_step",
    "data_ms_per_step", "val_score", "val_s", "time_s", "note",
)
TRAINING_KEYS = {
    "steps": int,
    "batch_size": int,
    "lr": float,
    "lr_end": float,
    "noise": float,
    "seed": int,
    "val_every": int,
    "log_every": int,
    "num_workers": int,
    "device": str,
    "precision": str,
}
OVERRIDES = ("noise", "lr", "steps", "seed", "precision", "device", "batch_size", "val_every", "log_every")
FLAGS = ("augment",)
PRECISIONS = ("fp32", "bf16")

MODEL_REGISTRY: dict[str, type[torch.nn.Module]] = {}


# ---------------------------------------------------------------- model registry

def register_model(name: str) -> Callable[[type[torch.nn.Module]], type[torch.nn.Module]]:
    """Decorator to register a model class with the harness."""
    def decorator(cls: type[torch.nn.Module]) -> type[torch.nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def load_known_model(key: str, name: str) -> type[torch.nn.Module]:
    """Dynamically import a known model class by name if not yet registered."""
    known_models = {
        "model_a": ("models.model_a", "ModelA"),
        "model_b": ("models.model_b", "ModelB"),
        "model_c": ("models.model_c", "ModelC"),
        "model_d": ("models.model_d", "ModelD"),
        "gcn": ("models.gcn", "GCNModel"),
        "unet": ("models.unet", "UNetModel"),
        "linear_filter": ("models.linear_filter", "LinearFilter"),
    }
    if key not in known_models:
        raise ValueError(
            f"Unknown model name '{name}'. Registered models: {sorted(MODEL_REGISTRY.keys())}. "
            f"Available auto-load models: {sorted(known_models.keys())}."
        )

    module_path, class_name = known_models[key]
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        MODEL_REGISTRY[key] = cls
        return cls
    except ModuleNotFoundError as e:
        if module_path in str(e):
            raise NotImplementedError(
                f"Model '{name}' is specified in configuration, but {module_path}.py has not been "
                f"implemented yet. Please implement {module_path}.py or register it with "
                f"train_harness.register_model."
            ) from e
        raise
    except AttributeError as e:
        raise AttributeError(f"Module {module_path} does not export {class_name}.") from e


def get_model_class(name: str) -> type[torch.nn.Module]:
    """Look up a model class by name. Dynamically imports known modules if not yet loaded."""
    key = name.lower()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    return load_known_model(key, name)


def build_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate a model from its model configuration dict."""
    cls = get_model_class(model_cfg["name"])
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    return cls(**kwargs)


# ---------------------------------------------------------------- configuration validation

def validate_sections(cfg: dict[str, Any]):
    """Ensure top-level sections are dictionaries."""
    for section in ("model", "data", "training"):
        val = cfg.get(section)
        if not isinstance(val, dict):
            raise TypeError(f"configuration section '{section}' must be a dict, got {type(val).__name__}")


def validate_model_config(model_cfg: dict[str, Any]):
    """Ensure model config specifies a valid non-empty string name."""
    model_name = model_cfg.get("name")
    if not isinstance(model_name, str):
        raise TypeError(f"model.name must be a string, got {type(model_name).__name__}")
    if not model_name:
        raise ValueError("model.name must be a non-empty string")


def validate_training_types(t: dict[str, Any]):
    """Validate training keys presence, boolean flags, and numeric types."""
    missing = [k for k in (*TRAINING_KEYS, *FLAGS) if k not in t]
    if missing:
        raise ValueError(f"training section missing keys: {', '.join(missing)}")

    for k in FLAGS:
        if not isinstance(t[k], bool):
            raise TypeError(f"training.{k} must be boolean (true/false), got {type(t[k]).__name__}")

    for k, typ in TRAINING_KEYS.items():
        t[k] = typ(t[k])


def validate_training_values(t: dict[str, Any]):
    """Validate training hyperparameters bounds and precision."""
    positive = [k for k in ("steps", "batch_size", "val_every", "log_every") if t[k] < 1]
    if positive or t["num_workers"] < 0 or t["seed"] < 0:
        raise ValueError("steps, batch_size, val_every, log_every must be >= 1; num_workers, seed >= 0")

    if not (t["lr"] > 0 and t["lr_end"] > 0 and math.isfinite(t["lr"]) and math.isfinite(t["lr_end"])):
        raise ValueError("lr and lr_end must be positive finite floats")

    if not (t["noise"] >= 0 and math.isfinite(t["noise"])):
        raise ValueError("noise must be a non-negative finite float")

    if t["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {t['precision']!r}")


def check_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate types, sections, and parameter bounds in configuration."""
    validate_sections(cfg)
    validate_model_config(cfg["model"])
    cfg["data"]["stride"] = check_stride(cfg["data"].get("stride"))
    t = cfg["training"]
    validate_training_types(t)
    validate_training_values(t)
    return cfg


def load_config(path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    with open(path) as f:
        return check_config(yaml.safe_load(f))


# ---------------------------------------------------------------- training maths & targets

def learning_rate(step: int, t: dict[str, Any]) -> float:
    """Exponential learning rate decay from lr at step 0 to lr_end at step (steps - 1)."""
    return t["lr"] * (t["lr_end"] / t["lr"]) ** (step / max(t["steps"] - 1, 1))


def target_of(batch: Any, target: str) -> torch.Tensor:
    """Extract the raw target change for the step."""
    if target == "dT":
        return batch.T_next - batch.T
    if target == "dH":
        return batch.H_next - batch.H
    raise ValueError(f"Unknown target '{target}', expected 'dT' or 'dH'")


def grad_norm(model: torch.nn.Module) -> torch.Tensor:
    """Compute total gradient L2 norm across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    if hasattr(torch.nn.utils, "get_total_norm"):
        return torch.nn.utils.get_total_norm(grads)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


# ---------------------------------------------------------------- data streaming & RNG

def set_all_seeds(seed: int):
    """Seed Python random, NumPy, PyTorch CPU and CUDA generators for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Seed DataLoader worker process deterministically from the PyTorch generator seed."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Batches of one epoch: reproducible pseudo-random permutation split into full batches."""
    k = n // batch_size
    return np.random.default_rng([seed, epoch]).permutation(n)[:k * batch_size].reshape(k, batch_size).tolist()


def batch_stream(dataset: Any, step: int, t: dict[str, Any]):
    """Infinite generator of batches across epochs, resuming at the exact step."""
    per_epoch = len(dataset) // t["batch_size"]
    if per_epoch < 1:
        raise ValueError(f"{len(dataset)} train samples are fewer than one batch of {t['batch_size']}")
    epoch, first = divmod(step, per_epoch)
    gen = torch.Generator().manual_seed(t["seed"])
    while True:
        batches = epoch_batches(len(dataset), t["batch_size"], t["seed"], epoch)[first:]
        yield from DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=t["num_workers"],
            generator=gen,
            worker_init_fn=seed_worker,
        )
        epoch, first = epoch + 1, 0


def rng_state() -> dict[str, Any]:
    """Capture current random state across Python, NumPy, and PyTorch."""
    st_tuple = np.random.get_state()
    name = str(st_tuple[0])
    keys = np.asarray(st_tuple[1]).tolist()
    pos = int(st_tuple[2])
    has_gauss = int(st_tuple[3])
    cached = float(st_tuple[4])
    st = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (name, keys, pos, has_gauss, cached),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict[str, Any]):
    """Restore random state across Python, NumPy, and PyTorch."""
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    name, keys, pos, has_gauss, cached = st["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if "cuda" in st and torch.cuda.is_available() and len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(st["cuda"])


# ---------------------------------------------------------------- metrics accumulator

class Interval:
    """Accumulates training loss and gradient norms on the device without per-step host syncs."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.loss = torch.zeros((), dtype=torch.float64, device=self.device)
        self.gn = torch.zeros((), dtype=torch.float64, device=self.device)
        self.gn_max = torch.zeros((), dtype=torch.float64, device=self.device)
        self.n = 0
        self.bad = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.t0 = time.perf_counter()
        self.data_s = 0.0

    def add(self, step: int, loss: torch.Tensor, gn: torch.Tensor):
        finite = torch.isfinite(loss) & torch.isfinite(gn)
        self.bad = torch.where(~finite & (self.bad < 0), step, self.bad)
        self.loss += loss.detach().double()
        self.gn += gn.detach().double()
        self.gn_max = torch.maximum(self.gn_max, gn.detach().double())
        self.n += 1

    def flush(self) -> tuple[dict[str, float], int]:
        stacked = torch.stack([self.bad.double(), self.loss, self.gn, self.gn_max])
        host_tensor = to_host(stacked)
        assert isinstance(host_tensor, torch.Tensor)
        bad, loss, gn, gn_max = host_tensor.tolist()
        if bad >= 0:
            raise FloatingPointError(
                f"non-finite training loss or gradient norm at step {int(bad)}; run terminated."
            )
        n = max(self.n, 1)
        ms = 1e3 * (time.perf_counter() - self.t0) / n
        out = {
            "train_loss": loss / n,
            "grad_norm": gn / n,
            "grad_norm_max": gn_max,
            "ms_per_step": ms,
            "data_ms_per_step": 1e3 * self.data_s / n,
        }
        steps = self.n
        self.reset()
        return out, steps


# ---------------------------------------------------------------- logging & persistence

def save_atomic(obj: Any, fn: str, tries: int = 10):
    """Write through a temporary file and atomically rename, retrying for Windows file lock delays."""
    tmp = fn + ".tmp"
    torch.save(obj, tmp)
    for i in range(tries):
        try:
            os.replace(tmp, fn)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1.0)


def write_log(fn: str, rows: list[dict[str, Any]], mode: str):
    """Write rows to a CSV file."""
    with open(fn, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOG_FIELDS), restval="")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


class Log:
    """CSV log writer that handles temporary Windows file locks from viewers."""

    def __init__(self, fn: str):
        self.fn = fn
        self.pending: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]):
        self.pending.append(row)
        self.flush()

    def flush(self):
        try:
            write_log(self.fn, self.pending, "a")
            self.pending = []
        except OSError as e:
            print(f"WARNING: cannot write {self.fn} ({e}); keeping {len(self.pending)} rows for next try")


def fmt_row(r: dict[str, Any], steps: int) -> str:
    """Format a log metrics row into a concise console line."""
    s = (f"step {r['step']:>7d}/{steps}  lr {r['lr']:.2e}  loss {r['train_loss']:.3e}  |grad| {r['grad_norm']:.2e} "
         f"(max {r['grad_norm_max']:.2e})  {r['ms_per_step']:.1f} ms/step (data {r['data_ms_per_step']:.1f})")
    if r.get("val_score", "") != "":
        s += f"  val {r['val_score']:.5g} in {r['val_s']:.1f} s"
    return s + (f"  {r['note']}" if r.get("note") else "")


def load_checkpoint(path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained model and checkpoint metadata into evaluation mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return move_checked(model, device, f"weights of {path}").eval(), ckpt


# ---------------------------------------------------------------- training helpers

def init_run_files(name: str) -> dict[str, str]:
    """Generate canonical artifact file paths for a given run name."""
    run_dir = os.path.join(CKPT_DIR, name)
    return {k: os.path.join(run_dir, k) for k in ("best.pt", "last.pt", "log.csv", "config.yaml")}


def setup_data_and_stats(
    stride: int,
    target: str,
    noise_multiplier: float,
    augment: bool,
    ckpt: dict[str, Any] | None = None,
) -> tuple[MeltPoolGraphs, list[Any], dict[str, Any], dict[str, Any], float, float]:
    """Load and verify datasets, size bins, and normalization statistics."""
    stats = load_stats(stride)
    bins = load_size_bins()
    if bins.get("train_fingerprint") != stats["train_fingerprint"]:
        raise ValueError("data/graphs/size_bins.json and statistics come from different training data")
    if ckpt is not None and (ckpt["stats"] != stats or ckpt["size_bins"] != bins):
        raise ValueError("Checkpoint statistics/size_bins mismatch current processed train split")
    if target not in stats["target"]:
        raise ValueError(f"No statistics found for target '{target}'")

    noise_std = noise_multiplier * stats["node"]["T"]["std"]
    train_set = MeltPoolGraphs("train", stride, stats=stats, noise_std=noise_std, augment=augment)
    val_runs = load_runs("val", 0)
    check_stats(stats, stride, val_runs, "split val")
    mean = float(stats["target"][target]["mean"])
    std = float(stats["target"][target]["std"])
    return train_set, val_runs, stats, bins, mean, std


def extract_adam_state(sd: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract flattened tensor parameters from an Adam optimizer state dict."""
    return {
        f"{i}.{k}": v
        for i, s in sd["state"].items()
        for k, v in s.items()
        if torch.is_tensor(v)
    }


def restore_from_checkpoint(
    ckpt: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    log_file: str,
) -> tuple[int, float, int | None, float]:
    """Restore optimizer, Adam device state, log file, and random seed from checkpoint."""
    optimizer.load_state_dict(ckpt["optimizer"])
    check_on_device(
        extract_adam_state(optimizer.state_dict()),
        extract_adam_state(ckpt["optimizer"]),
        f"Adam state of {log_file}",
    )
    step = int(ckpt["step"])
    best = float(ckpt["best_score"])
    best_step = int(ckpt["best_step"]) if ckpt["best_step"] is not None else None
    elapsed0 = float(ckpt["time_s"])

    with open(log_file, newline="") as f:
        kept = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
    write_log(log_file, kept, "w")
    set_rng_state(ckpt["rng"])
    return step, best, best_step, elapsed0


def init_fresh_run(run_dir: str, config_path: str, log_path: str, cfg: dict[str, Any]):
    """Initialize directories, dump configuration YAML, and create empty log CSV."""
    os.makedirs(run_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    write_log(log_path, [], "w")


def build_checkpoint(
    name: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    target: str,
    stride: int,
    stats: dict[str, Any],
    bins: dict[str, Any],
    step: int,
    score: float,
    best: float,
    best_step: int | None,
    elapsed: float,
    precision: str,
    optimizer: torch.optim.Optimizer | None = None,
    rng_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a complete checkpoint dictionary transferred to host tensors."""
    out: dict[str, Any] = {
        "format": 1,
        "name": name,
        "model": {k: to_host(v) for k, v in model.state_dict().items()},
        "model_config": cfg["model"],
        "config": cfg,
        "target": target,
        "stride": stride,
        "stats": stats,
        "train_fingerprint": stats["train_fingerprint"],
        "size_bins": bins,
        "step": step,
        "score": score,
        "best_score": best,
        "best_step": best_step,
        "time_s": elapsed,
        "precision": precision,
        "torch_version": str(torch.__version__),
    }
    if optimizer is not None and rng_data is not None:
        out.update(optimizer=tree_to_host(optimizer.state_dict()), rng=rng_data)
    return out


def assert_weights_finite(model: torch.nn.Module, step: int):
    """Ensure all model weights are finite before persisting checkpoints."""
    finite = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    host_finite = to_host(finite)
    assert isinstance(host_finite, torch.Tensor)
    if not bool(host_finite):
        raise FloatingPointError(f"non-finite model weights at step {step}; checkpoint not saved")


def execute_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    target: str,
    mean: float,
    std: float,
    precision: str,
    device: torch.device,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute a single forward pass, MSE loss calculation, backward pass, and optimizer step."""
    batch = batch.to(device)
    for group in optimizer.param_groups:
        group["lr"] = lr

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        out = model(batch)

    out = out.float()
    goal = (target_of(batch, target) - mean) / std
    if out.shape != goal.shape:
        raise ValueError(f"Model returned shape {tuple(out.shape)}, target has {tuple(goal.shape)}")

    loss = torch.mean((out - goal) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gn = grad_norm(model)
    optimizer.step()
    return loss, gn


def validate(predictor: TorchPredictor, model: torch.nn.Module, val_runs: list[Any], stride: int) -> tuple[float, str, float]:
    """Compute validation rollout selection score. Restores model.train() afterwards."""
    t0 = time.perf_counter()
    try:
        score, note = selection_score(val_runs, predictor, stride), ""
    except FloatingPointError as e:
        score, note = math.inf, f"validation rollout not finite: {e}"
    finally:
        model.train()
    return score, note, time.perf_counter() - t0


def evaluate_validation(
    predictor: TorchPredictor,
    model: torch.nn.Module,
    val_runs: list[Any],
    stride: int,
    best_score: float,
) -> tuple[float, str, float, bool]:
    """Run validation rollouts, return score and whether a new best model was found."""
    score, note, val_s = validate(predictor, model, val_runs, stride)
    improved = score < best_score
    if improved:
        note = (note + " " if note else "") + "best"
    return score, note, val_s, improved


# ---------------------------------------------------------------- main training orchestrator

def train(cfg: dict[str, Any], name: str, ckpt: dict[str, Any] | None = None) -> float:
    """Train or resume a run. Saves best.pt and last.pt. Returns best validation score."""
    t, stride = cfg["training"], cfg["data"]["stride"]
    device = torch.device(t["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("training.device is 'cuda' but torch.cuda.is_available() is False. Set training.device: cpu")

    files = init_run_files(name)
    model = build_model(cfg["model"])
    target = getattr(model, "target", cfg["model"].get("target", "dT"))
    train_set, val_runs, stats, bins, mean, std = setup_data_and_stats(
        stride, target, t["noise"], t["augment"], ckpt
    )
    set_stats_fn = getattr(model, "set_stats", None)
    if callable(set_stats_fn):
        set_stats_fn(mean, std)

    predictor = TorchPredictor(model, stats, stride, target, str(device), name=name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0, t))

    if ckpt is None:
        set_all_seeds(t["seed"])
        init_fresh_run(os.path.join(CKPT_DIR, name), files["config.yaml"], files["log.csv"], cfg)
        step, best, best_step, elapsed0 = 0, math.inf, None, 0.0
    else:
        model.load_state_dict(ckpt["model"])
        step, best, best_step, elapsed0 = restore_from_checkpoint(ckpt, optimizer, files["log.csv"])

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] model: {cfg['model']['name']}, {n_params:,} parameters, target: {target}, stride: {stride}, "
          f"samples: {len(train_set)}, noise: {t['noise']:g}, lr: {t['lr']:g} -> {t['lr_end']:g}, "
          f"steps: {t['steps']}, device: {device}, precision: {t['precision']}, augment: {t['augment']}"
          + (f", resumed from step {step}" if ckpt is not None else ""), flush=True)

    if step >= t["steps"]:
        print(f"[{name}] already finished {step}/{t['steps']} steps.")
        return best

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()
    interval = Interval(device)
    stream = batch_stream(train_set, step, t)
    timing: dict[str, list[Any]] = {"train": [], "val": []}
    log = Log(files["log.csv"])

    while step < t["steps"]:
        t_data = time.perf_counter()
        batch = next(stream)
        interval.data_s += time.perf_counter() - t_data

        lr = learning_rate(step, t)
        loss, gn = execute_train_step(model, optimizer, batch, target, mean, std, t["precision"], device, lr)
        step += 1
        interval.add(step, loss, gn)

        validating = (step % t["val_every"] == 0 or step == t["steps"])
        if step % t["log_every"] and not validating:
            continue

        stats_row, n_steps = interval.flush()
        row = {"step": step, "lr": lr, **stats_row}
        timing["train"].append((n_steps, row["ms_per_step"]))

        score: float = float("nan")
        improved: bool = False
        if validating:
            score, note, val_s, improved = evaluate_validation(predictor, model, val_runs, stride, best)
            timing["val"].append(val_s)
            if improved:
                best, best_step = score, step
            row.update(val_score=score, val_s=val_s, note=note)

        row["time_s"] = elapsed0 + time.perf_counter() - t_start
        log.append(row)
        print(fmt_row(row, t["steps"]), flush=True)

        if validating:
            assert_weights_finite(model, step)
            current_time = float(row["time_s"])
            if improved:
                best_ckpt = build_checkpoint(
                    name, model, cfg, target, stride, stats, bins,
                    step, score, best, best_step, current_time, t["precision"]
                )
                save_atomic(best_ckpt, files["best.pt"])
            last_ckpt = build_checkpoint(
                name, model, cfg, target, stride, stats, bins,
                step, score, best, best_step, current_time, t["precision"],
                optimizer, rng_state()
            )
            save_atomic(last_ckpt, files["last.pt"])
            interval.t0 = time.perf_counter()

    log.flush()
    steps_timed = sum(n for n, _ in timing["train"])
    ms = sum(n * m for n, m in timing["train"]) / max(steps_timed, 1)
    val_s = float(np.mean(timing["val"])) if timing["val"] else float("nan")
    val_every = max(t.get("val_every", 2000), 1)
    val_s_safe = val_s if not math.isnan(val_s) else 0.0
    tune_h = (20_000 * ms / 1e3 + (20_000 / val_every) * val_s_safe) / 3600.0
    full_h = (100_000 * ms / 1e3 + (100_000 / val_every) * val_s_safe) / 3600.0
    mem = (f", peak GPU memory {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GiB"
           if device.type == "cuda" else "")
    print(f"\n[{name}] Timing profile: {ms:.1f} ms/step, {val_s:.1f} s/val{mem}.")
    print(f"[{name}] Projected: 20k-step trial ~ {tune_h:.2f} h | 100k-step run ~ {full_h:.2f} h on this machine.\n")

    if best_step is not None:
        print(f"[{name}] Best validation score: {best:.5g} at step {best_step} -> {files['best.pt']}")
    else:
        print(f"[{name}] Warning: No valid finite score obtained.")
    return best


# ---------------------------------------------------------------- sweep runner

def generate_sweep_candidates(trials: int) -> list[tuple[float, float]]:
    """Generate fixed deterministic hyperparameter grid for tuning."""
    grid = {
        "lr": [1e-4, 3e-4],
        "noise": [0.003, 0.01, 0.03],
    }
    combinations = list(itertools.product(grid["lr"], grid["noise"]))
    return combinations[:trials] if trials < len(combinations) else combinations


def build_trial_config(base_cfg: dict[str, Any], lr: float, noise: float, steps: int) -> dict[str, Any]:
    """Clone base config with specific trial hyperparameter overrides."""
    trial_cfg = {
        "model": dict(base_cfg["model"]),
        "data": dict(base_cfg["data"]),
        "training": dict(base_cfg["training"]),
    }
    trial_cfg["training"]["lr"] = lr
    trial_cfg["training"]["noise"] = noise
    trial_cfg["training"]["steps"] = steps
    trial_cfg["training"]["seed"] = 0
    return trial_cfg


def save_and_display_sweep(model_name: str, results: list[dict[str, Any]]) -> str:
    """Save ranked sweep results to CSV and print summary table."""
    results.sort(key=lambda x: x["val_score"])
    csv_path = os.path.join(SWEEP_DIR, f"{model_name}_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "trial", "name", "val_score", "lr", "noise", "steps"])
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            writer.writerow({"rank": rank, **r})

    print("\n============================================================")
    print(f"Sweep Completed for {model_name}. Summary saved to {csv_path}:")
    for rank, r in enumerate(results, 1):
        print(f"  #{rank}: score {r['val_score']:.5g} | lr={r['lr']:g}, noise={r['noise']:g} ({r['name']})")
    print("============================================================\n")
    return csv_path


def run_sweep(base_cfg: dict[str, Any], trials: int = 6, override_steps: int | None = None) -> list[dict[str, Any]]:
    """Run a fixed trial budget hyperparameter sweep for fair model comparison."""
    os.makedirs(SWEEP_DIR, exist_ok=True)
    model_name = base_cfg["model"]["name"]
    candidates = generate_sweep_candidates(trials)
    steps_to_use = override_steps or min(base_cfg["training"]["steps"], 20000)

    print("\n============================================================")
    print(f"Starting {len(candidates)}-trial sweep for model: {model_name}")
    print("============================================================")

    results = []
    for i, (lr, noise) in enumerate(candidates):
        trial_cfg = build_trial_config(base_cfg, lr, noise, steps_to_use)
        trial_name = f"{model_name}_sweep_trial{i+1}_lr{lr:g}_noise{noise:g}"
        print(f"\n--- [Trial {i+1}/{len(candidates)}] {trial_name} ---")

        best_score = train(trial_cfg, trial_name)
        results.append({
            "trial": i + 1,
            "name": trial_name,
            "lr": lr,
            "noise": noise,
            "steps": steps_to_use,
            "val_score": best_score,
        })

    save_and_display_sweep(model_name, results)
    return results


# ---------------------------------------------------------------- multi-seed runner

def build_seed_config(base_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Clone config and set specific random seed."""
    seed_cfg = {
        "model": dict(base_cfg["model"]),
        "data": dict(base_cfg["data"]),
        "training": dict(base_cfg["training"]),
    }
    seed_cfg["training"]["seed"] = seed
    return seed_cfg


def run_multi_seed(cfg: dict[str, Any], seeds: list[int], name_prefix: str | None = None) -> dict[str, Any]:
    """Train the selected config across multiple seeds (e.g. 3 seeds) for reporting mean ± std."""
    model_name = cfg["model"]["name"]
    prefix = name_prefix or f"{model_name}_eval"
    scores = []

    print(f"\nTraining {len(seeds)} seeds for {model_name}: {seeds}")
    for s in seeds:
        seed_cfg = build_seed_config(cfg, s)
        run_name = f"{prefix}_seed{s}"
        score = train(seed_cfg, run_name)
        scores.append(score)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    print(f"\nMulti-seed result for {prefix}: {mean_score:.5g} ± {std_score:.5g} over seeds {seeds}\n")
    return {"seeds": seeds, "scores": scores, "mean": mean_score, "std": std_score}


# ---------------------------------------------------------------- CLI

def build_cli_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    ap = argparse.ArgumentParser(description="Unified training harness for melt-pool GNNs and baselines.")
    ap.add_argument("--config", help="Path to yaml config, e.g. configs/model_a.yaml")
    ap.add_argument("--name", help="Run name (saved under checkpoints/<name>/)")
    ap.add_argument("--resume", action="store_true", help="Resume run from checkpoints/<name>/last.pt")
    ap.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep with fixed trial budget")
    ap.add_argument("--trials", type=int, default=6, help="Number of trials for the sweep (default: 6)")
    ap.add_argument("--multi-seed", dest="multi_seed", action="store_true", help="Train multiple seeds for final evaluation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of seeds for multi-seed run")

    ap.add_argument("--lr", type=float, help="Override initial learning rate")
    ap.add_argument("--noise", type=float, help="Override noise standard deviation multiplier")
    ap.add_argument("--steps", type=int, help="Override number of training steps")
    ap.add_argument("--seed", type=int, help="Override single run seed")
    ap.add_argument("--precision", choices=PRECISIONS, help="Override precision (fp32 or bf16)")
    ap.add_argument("--device", help="Override compute device (e.g. cuda or cpu)")
    ap.add_argument("--batch-size", type=int, dest="batch_size", help="Override training batch size")
    ap.add_argument("--val-every", "--val_every", type=int, dest="val_every", help="Override validation interval")
    ap.add_argument("--log-every", "--log_every", type=int, dest="log_every", help="Override log interval")
    return ap


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI argument overrides to training configuration."""
    for k in OVERRIDES:
        v = getattr(args, k, None)
        if v is not None:
            cfg["training"][k] = v
    return check_config(cfg)


def dispatch_resume(run_name: str | None):
    """Resume execution from a checkpoint."""
    if not run_name:
        raise SystemExit("--resume requires --name")
    ckpt_path = os.path.join(CKPT_DIR, run_name, "last.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = check_config(ckpt["config"])
    train(cfg, run_name, ckpt)


def main(argv=None):
    """Parse arguments and dispatch to resume, sweep, multi-seed, or single train run."""
    ap = build_cli_parser()
    args = ap.parse_args(argv)

    if args.resume:
        dispatch_resume(args.name)
        return

    if not args.config:
        raise SystemExit("Missing --config argument (or --resume with --name)")

    base_cfg = apply_cli_overrides(load_config(args.config), args)

    if args.sweep:
        run_sweep(base_cfg, trials=args.trials, override_steps=args.steps)
        return

    if args.multi_seed:
        run_multi_seed(base_cfg, seeds=args.seeds, name_prefix=args.name)
        return

    t = base_cfg["training"]
    run_name = args.name or f"{base_cfg['model']['name']}_n{t['noise']:g}_lr{t['lr']:g}_seed{t['seed']}_{t['steps']}_{t['precision']}"
    train(base_cfg, run_name)


if __name__ == "__main__":
    main()
