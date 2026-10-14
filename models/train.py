"""Training (plan step 4): one-step training on the train split, checkpoints picked on the validation rollout error.

Usage (from the repo root, after python -m graphs.preprocess, python -m graphs.dataset --stride 5 and
python -m evaluation.rollout --size-bins):
  python -m models.train --config configs/model_a.yaml --steps 2000 --name smoke          smoke run: cost per step
  python -m models.train --config configs/model_a.yaml --noise 0.01 --lr 1e-4 --steps 20000 --name a_tune_n0.01_lr1e-4
  python -m models.train --config configs/model_a.yaml --seed 1 --name a_seed1
  python -m models.train --name a_seed1 --resume                                          continue after a stop
Overrides of the configuration: --noise (a fraction of the train spread of T), --lr, --steps, --seed, --precision,
--name.
--resume continues checkpoints/<name>/last.pt with its saved configuration; --config and overrides may be given
only if they agree with it. Without --resume a run name that already has a checkpoint is refused; a folder without
one (a run that stopped before its first validation) is started over.

What a run does
  loss        mean squared error of the model output against the normalised target (dT - mean) / std over every
              node of the batch, each node weighted equally (statistics: data/graphs/norm_stride<s>.json)
  noise       Gaussian noise with std = noise x the train spread of T on the input temperature. The features and the
              target use the noisy state (graphs/dataset.py, MeltPoolGraphs noise_std)
  augment     true: each training sample is rotated by a uniform random angle and reflected with probability 0.5 (its
              edge vectors and positions; graphs/dataset.py, MeltPoolGraphs augment). Required in the configuration;
              runs saved before it existed resume as false. Validation is never augmented
  precision   fp32, or bf16: the training forward pass runs under torch.autocast(bfloat16), so the hidden matrix
              products of the encoders and the processor use bf16 (the GPU's matrix units), while the weights, Adam
              state, the encoders' first layers, LayerNorm, residual streams, edge sums, decoder and loss stay fp32
              (models/backbone.py). Validation and every evaluation run in fp32. The checkpoint records the precision.
              Required in the configuration; runs saved before it existed resume as fp32
  optimiser   Adam with learning rate lr x (lr_end / lr)^(s / (steps - 1)) for update s = 0 .. steps - 1, an
              exponential decay from lr at the first update to lr_end at the last, over the run's own length;
              batch_size graphs per step
  data order  each epoch is a permutation of the train samples drawn from (seed, epoch); the incomplete last batch
              of an epoch is dropped. So the step number fixes the position in the data
  validation  every val_every steps and at the last step: evaluation.rollout.selection_score, the mean rollout nRMSE
              over the val runs at the model's stride, through a TorchPredictor around the model being trained (same
              weights, same device). The model is put back in training mode afterwards, also when the validation
              fails. A rollout that becomes non-finite (FloatingPointError from the harness) scores inf: it is
              logged with the run and frame, never selected, and training continues. Any other error stops the run
  stops       on a non-finite training loss or gradient norm (checked every log_every steps and before each save;
              the message names the first such step), on a code error, and when the statistics, the size bins or a
              resumed run do not fit the current processed train split
  transfers   on a GPU, every read that a result depends on is verified (evaluation/torch_predictor.py): the training
              statistics of each log row, the finite-weights check, the weights and Adam state copied into a
              checkpoint, and the validation rollouts; so are the weights and Adam state uploaded at the start or on
              --resume. A transfer that stays wrong after a few tries stops the run with StaleReadError (--resume
              continues). The training batches are uploaded without a check

Output in checkpoints/<name>/ (git-ignored)
  best.pt      the checkpoint with the lowest validation score
  last.pt      the latest validation step, with the optimiser state and the random generator states for --resume
  log.csv      every log_every steps: step, learning rate, mean train loss and mean and largest gradient norm since
               the previous row, time per step (and the part spent loading data), elapsed time; at validation steps
               also the score, the validation time and a note if the validation failed
  config.yaml  the resolved configuration
A checkpoint holds the model weights and configuration, the training statistics, the stride, the target, the train
fingerprint, the size bins, the step, its validation score and the best score so far. load_checkpoint() rebuilds the
model (evaluation.rollout --checkpoint uses it). Checkpoints load with torch.load(weights_only=True).

Resume: weights, optimiser, step, best score and the random generator states (torch, CUDA, numpy, random) come from
last.pt. The learning rate and the data order are functions of the step, so the run continues where last.pt was
saved: on a CPU with num_workers 0 the weights come out bit for bit the same as without the stop. With num_workers > 0
the noise draws after a resume differ, and on a GPU the scatter-sum is not bit-reproducible anyway. log.csv rows after
the resumed step are dropped.
"""
import argparse
import csv
import math
import os
import random
import time

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
from train_harness import build_model as harness_build_model, get_model_class

MODELS = {
    "model_a": get_model_class("model_a"),
    "model_b": get_model_class("model_b"),
    "gcn": get_model_class("gcn"),
    "model_d": get_model_class("model_d"),
}
CKPT_DIR = "checkpoints"
LOG_FIELDS = ("step", "lr", "train_loss", "grad_norm", "grad_norm_max", "ms_per_step", "data_ms_per_step",
              "val_score", "val_s", "time_s", "note")
TRAINING_KEYS = dict(steps=int, batch_size=int, lr=float, lr_end=float, noise=float, seed=int, val_every=int,
                     log_every=int, num_workers=int, device=str, precision=str)
OVERRIDES = ("noise", "lr", "steps", "seed", "precision")
FLAGS = ("augment",)                                  # true or false in the configuration (yaml booleans only)
PRECISIONS = ("fp32", "bf16")


# ---------------------------------------------------------------- configuration and model

def check_config(cfg):
    """Fill types and check values. Returns the configuration with numbers as numbers (yaml reads 1e-4 as text)."""
    for section in ("model", "data", "training"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"configuration has no '{section}' section")
    if cfg["model"].get("name") not in MODELS:
        raise ValueError(f"model name must be one of {sorted(MODELS)}, not {cfg['model'].get('name')!r}")
    cfg["data"]["stride"] = check_stride(cfg["data"].get("stride"))
    t = cfg["training"]
    missing = [k for k in (*TRAINING_KEYS, *FLAGS) if k not in t]
    if missing:
        raise ValueError(f"training section misses {', '.join(missing)}")
    for k in FLAGS:
        if not isinstance(t[k], bool):
            raise ValueError(f"training.{k} must be true or false, not {t[k]!r}")
    for k, typ in TRAINING_KEYS.items():
        t[k] = typ(t[k])
    positive = [k for k in ("steps", "batch_size", "val_every", "log_every") if t[k] < 1]
    if positive or t["num_workers"] < 0 or t["seed"] < 0:
        raise ValueError("steps, batch_size, val_every, log_every must be at least 1 and num_workers, seed at least 0")
    if not (t["lr"] > 0 and t["lr_end"] > 0 and math.isfinite(t["lr"]) and math.isfinite(t["lr_end"])):
        raise ValueError("lr and lr_end must be positive")
    if not (t["noise"] >= 0 and math.isfinite(t["noise"])):
        raise ValueError("noise must be at least 0")
    if t["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, not {t['precision']!r}")
    return cfg


def load_config(path):
    with open(path) as f:
        return check_config(yaml.safe_load(f))


def build_model(model_cfg, stats=None):
    return harness_build_model(model_cfg, stats=stats)


def load_checkpoint(path, device="cpu"):
    """The model of a checkpoint in evaluation mode on device, and the checkpoint dict (statistics, stride, target,
    size bins, step, scores)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"], ckpt["stats"])
    model.load_state_dict(ckpt["model"])
    return move_checked(model, device, f"weights of {path}").eval(), ckpt       # on a GPU: checked on arrival


def learning_rate(step, t):
    """Learning rate of the update after `step` finished steps: exponential decay from lr at the first update to
    lr_end at the last one."""
    return t["lr"] * (t["lr_end"] / t["lr"]) ** (step / max(t["steps"] - 1, 1))


def target_of(batch, target):
    return batch.T_next - batch.T if target == "dT" else batch.H_next - batch.H


def grad_norm(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if hasattr(torch.nn.utils, "get_total_norm"):
        return torch.nn.utils.get_total_norm(grads)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


# ---------------------------------------------------------------- data order and random state

def epoch_batches(n, batch_size, seed, epoch):
    """The batches of one epoch: a permutation drawn from (seed, epoch), cut into full batches."""
    k = n // batch_size
    return np.random.default_rng([seed, epoch]).permutation(n)[:k * batch_size].reshape(k, batch_size).tolist()


def batch_stream(dataset, step, t):
    """Batches from the given step on (step = batches already used), across epochs."""
    per_epoch = len(dataset) // t["batch_size"]
    if per_epoch < 1:
        raise ValueError(f"{len(dataset)} train samples are fewer than one batch of {t['batch_size']}")
    epoch, first = divmod(step, per_epoch)
    gen = torch.Generator().manual_seed(t["seed"])          # DataLoader's worker seeds; keeps the global RNG for noise
    while True:
        batches = epoch_batches(len(dataset), t["batch_size"], t["seed"], epoch)[first:]
        yield from DataLoader(dataset, batch_sampler=batches, num_workers=t["num_workers"], generator=gen)
        epoch, first = epoch + 1, 0


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    st = dict(torch=torch.get_rng_state(), python=random.getstate(),
              numpy=(name, keys.tolist(), int(pos), int(has_gauss), float(cached)))
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    name, keys, pos, has_gauss, cached = st["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if "cuda" in st and torch.cuda.is_available() and len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(st["cuda"])


# ---------------------------------------------------------------- logging and checkpoints

class Interval:
    """Train loss and gradient norm since the last log row, accumulated on the device (no sync per step)."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        def z():
            return torch.zeros((), dtype=torch.float64, device=self.device)

        self.loss, self.gn, self.gn_max, self.n = z(), z(), z(), 0
        self.bad = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.t0, self.data_s = time.perf_counter(), 0.0

    def add(self, step, loss, gn):
        finite = torch.isfinite(loss) & torch.isfinite(gn)
        self.bad = torch.where(~finite & (self.bad < 0), step, self.bad)
        self.loss += loss.detach().double()
        self.gn += gn.detach().double()
        self.gn_max = torch.maximum(self.gn_max, gn.detach().double())
        self.n += 1

    def flush(self):
        """Mean loss, mean and largest gradient norm, ms per step and data ms per step. Raises on a non-finite step.
        The four device values come in one verified read (a stale read once gave a false stop here)."""
        bad, loss, gn, gn_max = to_host(torch.stack([self.bad.double(), self.loss, self.gn, self.gn_max])).tolist()
        if bad >= 0:
            raise FloatingPointError(f"non-finite training loss or gradient norm at step {int(bad)}; the run stops "
                                     f"(--resume continues from the last validation step, if one was saved)")
        n = max(self.n, 1)
        ms = 1e3 * (time.perf_counter() - self.t0) / n
        out = dict(train_loss=loss / n, grad_norm=gn / n, grad_norm_max=gn_max,
                   ms_per_step=ms, data_ms_per_step=1e3 * self.data_s / n)
        steps = self.n
        self.reset()
        return out, steps


def save(obj, fn, tries=10):
    """Write through a temporary file. Retries the final rename for a few seconds, since on Windows another program
    (a virus scanner, a file viewer) can briefly hold the old file open."""
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


def write_log(fn, rows, mode):
    with open(fn, mode, newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS, restval="")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


class Log:
    """Appends rows to log.csv. If the file is locked (on Windows, while it is open in Excel), the rows are kept and
    written with the next row, and a warning is printed, so the run goes on."""

    def __init__(self, fn):
        self.fn, self.pending = fn, []

    def append(self, row):
        self.pending.append(row)
        self.flush()

    def flush(self):
        try:
            write_log(self.fn, self.pending, "a")
            self.pending = []
        except OSError as e:
            print(f"WARNING: cannot write {self.fn} ({e}); {len(self.pending)} rows kept for the next try")


def fmt_row(r, steps):
    s = (f"step {r['step']:>7d}/{steps}  lr {r['lr']:.2e}  loss {r['train_loss']:.3e}  |grad| {r['grad_norm']:.2e} "
         f"(max {r['grad_norm_max']:.2e})  {r['ms_per_step']:.1f} ms/step (data {r['data_ms_per_step']:.1f})")
    if r.get("val_score", "") != "":
        s += f"  val {r['val_score']:.5g} in {r['val_s']:.1f} s"
    return s + (f"  {r['note']}" if r.get("note") else "")


# ---------------------------------------------------------------- training

def validate(predictor, model, val_runs, stride):
    """Selection score through the predictor; inf for a non-finite rollout. The model is in training mode after."""
    t0 = time.perf_counter()
    try:
        score, note = selection_score(val_runs, predictor, stride), ""
    except FloatingPointError as e:
        score, note = math.inf, f"validation rollout not finite: {e}"
    finally:
        model.train()
    return score, note, time.perf_counter() - t0


def train(cfg, name, ckpt=None):
    """Train (or, with ckpt = the loaded last.pt, resume) one run. Returns the best score."""
    t, stride = cfg["training"], cfg["data"]["stride"]
    device = torch.device(t["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("training.device is cuda, but torch.cuda.is_available() is False: install the CUDA build of "
                         "torch (https://pytorch.org) or set training.device: cpu")
    run_dir = os.path.join(CKPT_DIR, name)
    files = {k: os.path.join(run_dir, k) for k in ("best.pt", "last.pt", "log.csv", "config.yaml")}

    stats = load_stats(stride)                  # checked against the current processed train split
    bins = load_size_bins()                     # the same
    if bins.get("train_fingerprint") != stats["train_fingerprint"]:
        raise ValueError("data/graphs/size_bins.json and the statistics come from different training data")
    if ckpt is not None and (ckpt["stats"] != stats or ckpt["size_bins"] != bins):
        raise ValueError(f"{files['last.pt']} was trained with other statistics or size bins than the current "
                         f"processed train split; it cannot be resumed on this data")

    if ckpt is None:
        random.seed(t["seed"])
        np.random.seed(t["seed"])
        torch.manual_seed(t["seed"])
    model = build_model(cfg["model"], stats)
    target = model.target
    if target not in stats["target"]:
        raise ValueError(f"no statistics for target {target}")
    noise_std = t["noise"] * stats["node"]["T"]["std"]
    train_set = MeltPoolGraphs("train", stride, stats=stats, noise_std=noise_std, augment=t["augment"])
    val_runs = load_runs("val", 0)
    check_stats(stats, stride, val_runs, "split val")      # fail now, not at the first validation
    mean, std = stats["target"][target]["mean"], stats["target"][target]["std"]

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    predictor = TorchPredictor(model, stats, stride, target, device, name=name)   # moves the model to the device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0, t))
    if ckpt is None:
        step, best, best_step, elapsed0 = 0, math.inf, None, 0.0
        os.makedirs(run_dir, exist_ok=True)
        with open(files["config.yaml"], "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        write_log(files["log.csv"], [], "w")
    else:
        optimizer.load_state_dict(ckpt["optimizer"])

        def adam_state(sd):
            return {
                f"{i}.{k}": v
                for i, s in sd["state"].items()
                for k, v in s.items()
                if torch.is_tensor(v)
            }

        check_on_device(adam_state(optimizer.state_dict()), adam_state(ckpt["optimizer"]),
                        f"Adam state of {files['last.pt']}")       # on a GPU: arrived intact
        step, best, best_step, elapsed0 = ckpt["step"], ckpt["best_score"], ckpt["best_step"], ckpt["time_s"]
        with open(files["log.csv"], newline="") as f:
            kept = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
        write_log(files["log.csv"], kept, "w")
        set_rng_state(ckpt["rng"])
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {cfg['model']['name']}, {n_params:,} parameters, target {target}, stride {stride}, "
          f"{len(train_set)} train samples ({len(train_set) // t['batch_size']} batches of {t['batch_size']} per "
          f"epoch), {len(val_runs)} val runs, noise {t['noise']:g} x {stats['node']['T']['std']:.1f} K = "
          f"{noise_std:.2f} K, lr {t['lr']:g} -> {t['lr_end']:g}, {t['steps']} steps, device {device}, "
          f"precision {t['precision']} (validation fp32), augment "
          f"{'rotations and reflections' if t['augment'] else 'off'}"
          + (f", resumed at step {step}" if ckpt is not None else ""), flush=True)
    if step >= t["steps"]:
        print(f"{name} already ran {step} of {t['steps']} steps")
        return best

    def checkpoint(score, elapsed, rng=None):
        out = {"format": 1, "name": name, "model": {k: to_host(v) for k, v in model.state_dict().items()},
                   "model_config": cfg["model"], "config": cfg, "target": target, "stride": stride, "stats": stats,
                   "train_fingerprint": stats["train_fingerprint"], "size_bins": bins, "step": step, "score": score,
                   "best_score": best, "best_step": best_step, "time_s": elapsed, "precision": t["precision"],
                   "torch_version": str(torch.__version__)}
        if rng is not None:
            out.update(optimizer=tree_to_host(optimizer.state_dict()), rng=rng)
        return out

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    interval, stream, timing = Interval(device), batch_stream(train_set, step, t), dict(train=[], val=[])
    log = Log(files["log.csv"])
    while step < t["steps"]:
        t_data = time.perf_counter()
        batch = next(stream)                                # building the samples on the CPU (data time)
        interval.data_s += time.perf_counter() - t_data
        batch = batch.to(device)
        lr = learning_rate(step, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=t["precision"] == "bf16"):
            out = model(batch)
        out, goal = out.float(), (target_of(batch, target) - mean) / std     # the loss in fp32
        if out.shape != goal.shape:
            raise ValueError(f"{cfg['model']['name']} returned shape {tuple(out.shape)}, the target has "
                             f"{tuple(goal.shape)}")
        loss = torch.mean((out - goal) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norm(model)
        optimizer.step()
        step += 1
        interval.add(step, loss, gn)

        validating = step % t["val_every"] == 0 or step == t["steps"]
        if step % t["log_every"] and not validating:
            continue
        stats_row, n_steps = interval.flush()             # raises on a non-finite step, before anything is saved
        row = dict(step=step, lr=lr, **stats_row)
        timing["train"].append((n_steps, row["ms_per_step"]))
        if validating:
            score, note, val_s = validate(predictor, model, val_runs, stride)
            timing["val"].append(val_s)
            improved = score < best
            if improved:
                best, best_step = score, step
                note = (note + " " if note else "") + "best"
            row.update(val_score=score, val_s=val_s, note=note)
        row["time_s"] = elapsed0 + time.perf_counter() - t_start
        log.append(row)                                     # the row first, so a resume never lacks it
        print(fmt_row(row, t["steps"]), flush=True)
        if validating:
            finite = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
            if not bool(to_host(finite)):
                raise FloatingPointError(f"non-finite model weights at step {step}; nothing saved")
            rng = rng_state()                               # what the next batch draws from, after validation
            if improved:
                save(checkpoint(score, row["time_s"]), files["best.pt"])
            save(checkpoint(score, row["time_s"], rng), files["last.pt"])
            interval.t0 = time.perf_counter()               # validation and saving are not training time

    log.flush()
    steps_timed = sum(n for n, _ in timing["train"])
    ms = sum(n * m for n, m in timing["train"]) / max(steps_timed, 1)
    val_s = float(np.mean(timing["val"])) if timing["val"] else float("nan")
    full = 100_000 * ms / 1e3 + 100_000 / t["val_every"] * val_s
    mem = (f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB"
           if device.type == "cuda" else "")
    print(f"\n{name}: {ms:.1f} ms per training step, {val_s:.1f} s per validation{mem}. A 100,000-step run with "
          f"validation every {t['val_every']} steps would take about {full / 3600:.1f} h on this machine.")
    if best_step is None:
        print("no finite validation score, so best.pt was not written")
    else:
        print(f"best validation score {best:.5g} at step {best_step}: {files['best.pt']}")
    return best


# ---------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a model (plan step 4); see the module docstring.")
    ap.add_argument("--config", help="yaml configuration, e.g. configs/model_a.yaml")
    ap.add_argument("--name", help="run name: checkpoints/<name>/")
    ap.add_argument("--noise", type=float, help="input noise as a fraction of the train spread of T")
    ap.add_argument("--lr", type=float, help="learning rate at step 0")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--precision", choices=PRECISIONS, help="fp32, or bf16 matrix products in training")
    ap.add_argument("--resume", action="store_true", help="continue checkpoints/<name>/last.pt")
    a = ap.parse_args(argv)

    def with_overrides(cfg):
        for k in OVERRIDES:
            if getattr(a, k) is not None:
                cfg["training"][k] = getattr(a, k)
        return check_config(cfg)

    if a.resume:
        if not a.name:
            raise SystemExit("--resume needs --name")
        fn = os.path.join(CKPT_DIR, a.name, "last.pt")
        if not os.path.exists(fn):
            raise SystemExit(f"{fn} not found, nothing to resume")
        ckpt = torch.load(fn, map_location="cpu", weights_only=True)
        saved = {k: dict(v) for k, v in ckpt["config"].items()}
        saved["training"].setdefault("precision", "fp32")     # runs saved before the setting existed were fp32
        saved["training"].setdefault("augment", False)        # and without augmentation
        cfg = check_config(saved)
        given = with_overrides(load_config(a.config) if a.config else {k: dict(v) for k, v in cfg.items()})
        if given != cfg:
            diff = [f"{sec}.{k}: saved {cfg[sec].get(k)!r}, given {given[sec].get(k)!r}" for sec in cfg
                    for k in sorted(set(cfg[sec]) | set(given.get(sec, {}))) if cfg[sec].get(k) != given[sec].get(k)]
            raise SystemExit(f"--config or overrides differ from the saved configuration of {a.name} ("
                             + "; ".join(diff) + "); a resumed run keeps its configuration (see "
                             f"{os.path.join(CKPT_DIR, a.name, 'config.yaml')})")
        train(cfg, a.name, ckpt)
        return
    if not a.config:
        raise SystemExit("--config is needed (or --resume with --name)")
    cfg = with_overrides(load_config(a.config))
    t = cfg["training"]
    name = a.name or (f"{cfg['model']['name']}_n{t['noise']:g}_lr{t['lr']:g}_seed{t['seed']}_{t['steps']}_"
                      f"{t['precision']}" + ("_aug" if t["augment"] else ""))
    if any(os.path.exists(os.path.join(CKPT_DIR, name, f)) for f in ("best.pt", "last.pt")):
        raise SystemExit(f"{os.path.join(CKPT_DIR, name)} already holds a run: continue it with --resume, or pick "
                         f"another --name")
    if os.path.exists(os.path.join(CKPT_DIR, name, "log.csv")):
        print(f"{os.path.join(CKPT_DIR, name)} holds a run that stopped before its first checkpoint; starting it over")
    train(cfg, name)


if __name__ == "__main__":
    main()
