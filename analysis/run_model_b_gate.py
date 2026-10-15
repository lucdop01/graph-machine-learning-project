"""Prepare, train and score the preregistered A/B/fixed-filter three-seed gate.

Run from the repository root. --check is read-only. --prepare-data generates
the full train/val/test_interp splits. --train runs six equal-budget jobs in
the shared trainer, then evaluates. Without --train, evaluate existing jobs.
Interrupted jobs require --resume; configs/data must match before reuse.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml

from models.train import load_config, load_checkpoint
from graphs.dataset import load_stats, load_runs, MeltPoolGraphs
from evaluation.rollout import load_size_bins, evaluate_split, write_results
from evaluation.torch_predictor import TorchPredictor
from models.linear_filter import LinearFilter
from analysis.model_b_gate import decide

SEEDS = (0, 1, 2)
DOMAINS = {"val": ("val", 0), "new_mesh": ("test_interp", 1)}


def command(*args):
    subprocess.run([sys.executable, "-m", *map(str, args)], check=True)


def gate_configs(steps, device, precision):
    configs = {m: load_config(f"configs/model_{m.lower()}.yaml") for m in ("A", "B")}
    for cfg in configs.values():
        cfg["training"].update(steps=steps, device=device, precision=precision)
    assert configs["A"]["training"] == configs["B"]["training"], "A/B training settings differ"
    assert configs["A"]["data"] == configs["B"]["data"], "A/B data settings differ"
    for key in ("width", "layers", "hidden_layers"):
        assert configs["A"]["model"][key] == configs["B"]["model"][key], f"A/B {key} differs"
    if configs["A"]["data"]["stride"] != 5:
        raise ValueError("the gate is preregistered at stride 5")
    return configs


def check_data():
    stats, bins = load_stats(5), load_size_bins()
    if bins["train_fingerprint"] != stats["train_fingerprint"]:
        raise ValueError("size bins and normalisation have different training data")
    for split, seed, count in (("train", 0, 150), ("val", 0, 20), ("test_interp", 1, 30)):
        runs = load_runs(split, seed, preload=False)
        if len(runs) != count:
            raise ValueError(f"gate requires {count} {split} runs, found {len(runs)}")
    return stats, bins


def validate_checkpoint(ckpt, expected, stats):
    if ckpt["config"] != expected:
        raise ValueError("checkpoint configuration differs from this gate; use a new --prefix")
    if ckpt["stats"] != stats or ckpt["target"] != "dT" or ckpt["stride"] != 5:
        raise ValueError("checkpoint data statistics, target or stride differs from this gate")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--prefix", default="b_gate_20k_aug")
    ap.add_argument("--out", default="results/lei/gate")
    a = ap.parse_args(argv)
    if a.steps < 1 or Path(a.prefix).name != a.prefix or a.prefix in (".", ".."):
        ap.error("steps must be positive and prefix must be a plain run name")
    if a.check and (a.prepare_data or a.train):
        ap.error("--check is read-only; use separately from --prepare-data or --train")
    configs = gate_configs(a.steps, a.device, a.precision)
    if a.prepare_data:
        command("data.generate", "--only", "train", "val", "test_interp")
        command("graphs.preprocess", "--splits", "train", "val", "test_interp")
        command("graphs.preprocess", "--splits", "test_interp", "--mesh-seed", 1)
        # Only computes stats if absent; never silently overwrites existing stats.
        MeltPoolGraphs("train", 5, preload=False)
        command("evaluation.rollout", "--size-bins")
        print("Gate data prepared. Run --check, then --train on the training machine.")
        return
    try:
        stats, bins = check_data()
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Gate data not ready: {e}\nPrepare with: python -m analysis.run_model_b_gate --prepare-data")
    if a.check:
        print(f"Data/config checks pass. CUDA available: {torch.cuda.is_available()}. "
              f"Six jobs, {a.steps} steps each, train-only stats {stats['train_fingerprint']}.")
        return
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Use a GPU training machine; --device cpu --precision fp32 is explicit CPU mode.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    previous_manifest = out / "manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text())
        if previous.get("configs") != configs:
            raise SystemExit("Output directory belongs to another gate configuration; choose a new --out and --prefix")
    manifest = dict(status="pending", rule="see analysis/model_b_gate.py", steps=a.steps,
                    seeds=list(SEEDS), domains=DOMAINS, train_fingerprint=stats["train_fingerprint"],
                    torch_version=str(torch.__version__), checkpoints={})
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    manifest["configs"] = configs
    repo = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {path: sha256(repo / path) for path in
                                ("models/train.py", "models/model_b.py", "graphs/dataset.py",
                                 "analysis/model_b_gate.py", "analysis/run_model_b_gate.py")}
    # No previous verdict should look current while re-scoring is incomplete.
    for filename in ("decision.json", "report.md"):
        (out / filename).unlink(missing_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    runs = {d: load_runs(split, seed) for d, (split, seed) in DOMAINS.items()}
    for model_name in ("A", "B"):
        for seed in SEEDS:
            cfg = copy.deepcopy(configs[model_name])
            cfg["training"]["seed"] = seed
            name = f"{a.prefix}_{model_name.lower()}_seed{seed}"
            config_path = out / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            last = Path("checkpoints") / name / "last.pt"
            if last.exists():
                ckpt = torch.load(last, map_location="cpu", weights_only=True)
                validate_checkpoint(ckpt, cfg, stats)
                if ckpt["step"] < a.steps and not (a.train and a.resume):
                    raise SystemExit(f"{name} incomplete; rerun with --train --resume")
            elif not a.train:
                raise SystemExit(f"{last} missing; run with --train")
            if a.train and (not last.exists() or ckpt["step"] < a.steps):
                args = ["models.train", "--config", config_path, "--name", name]
                if last.exists():
                    args.append("--resume")
                command(*args)
            best = last.with_name("best.pt")
            model, ckpt = load_checkpoint(str(best), a.device)
            validate_checkpoint(ckpt, cfg, stats)
            manifest["checkpoints"][name] = dict(best_step=ckpt["step"], sha256=sha256(best))
            predictor = TorchPredictor(model, stats, 5, device=a.device, name=name)
            for domain, domain_runs in runs.items():
                per_run, summary = evaluate_split(domain_runs, predictor, 5, bins)
                write_results(per_run, summary, str(out), f"{name}_{domain}")
                rows.append(dict(model=model_name, seed=seed, domain=domain,
                                 nrmse=summary["nrmse_mean"], melt_mae=summary["melt_mae"]))
            del predictor, model
    for domain, domain_runs in runs.items():
        per_run, summary = evaluate_split(domain_runs, LinearFilter(), 5, bins)
        write_results(per_run, summary, str(out), f"lf_{domain}")
        rows.extend(dict(model="LF", seed=seed, domain=domain, nrmse=summary["nrmse_mean"],
                         melt_mae=summary["melt_mae"]) for seed in SEEDS)
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "domain", "nrmse", "melt_mae"])
        w.writeheader()
        w.writerows(rows)
    decision = decide(rows)
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest["status"] = "complete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ["# Model B gate", "", f"Decision: **{decision['decision']}**. Primary metric: rollout nRMSE.",
              "", "| Domain | Model | nRMSE mean ± sample std | Melting-range MAE mean ± sample std (K) |",
              "|---|---|---|---|"]
    for domain in DOMAINS:
        for m in ("A", "B", "LF"):
            group = [r for r in rows if r["domain"] == domain and r["model"] == m]
            fmt = lambda key: f"{np.mean([r[key] for r in group]):.6g} ± {np.std([r[key] for r in group], ddof=1):.3g}"
            report.append(f"| {domain} | {m} | {fmt('nrmse')} | {fmt('melt_mae')} |")
    report += ["", "LF is deterministic: repeated rows label the paired comparisons, not independent fits.",
               "Decision uses both domains and the preregistered nRMSE rule; melting MAE is descriptive."]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
