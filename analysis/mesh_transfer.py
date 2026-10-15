"""Score checkpoints on test_interp mesh 0/1 and test_fine, beside the solver gap."""
import argparse
import json
from pathlib import Path
from models.train import load_checkpoint
from models.linear_filter import LinearFilter
from graphs.dataset import load_runs, load_stats
from evaluation.torch_predictor import TorchPredictor
from evaluation.rollout import CrossSolver, evaluate_split, load_size_bins, write_results, _clean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/lei/mesh_transfer")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"mesh0": load_runs("test_interp", 0), "mesh1": load_runs("test_interp", 1),
            "fine": load_runs("test_fine", 0)}
    bins, report = load_size_bins(), {}
    strides = set()

    def score(name, predictor, stride):
        for domain, domain_runs in runs.items():
            rows, summary = evaluate_split(domain_runs, predictor, stride, bins,
                                           runs["mesh0"] if domain == "fine" else None)
            key = f"{name}_{domain}_s{stride}"
            write_results(rows, summary, str(out), key)
            report[key] = summary

    for path in a.checkpoints:
        model, ckpt = load_checkpoint(path, a.device)
        stride = ckpt["stride"]
        if ckpt["stats"] != load_stats(stride) or ckpt["size_bins"] != bins:
            raise ValueError(f"{path}: checkpoint and current training data differ")
        name = Path(path).parent.name
        predictor = TorchPredictor(model, ckpt["stats"], stride, ckpt["target"], a.device, name=name)
        if any(key.startswith(name + "_") for key in report):
            raise ValueError("checkpoint run names must be unique")
        score(name, predictor, stride)
        strides.add(stride)
    for stride in sorted(strides):
        score("fixed_filter", LinearFilter(), stride)
        rows, summary = evaluate_split(runs["fine"], CrossSolver(runs["mesh0"]), stride, bins, runs["mesh0"])
        key = f"solver_gap_s{stride}"
        write_results(rows, summary, str(out), key)
        report[key] = summary
    (out / "summary.json").write_text(json.dumps(_clean(report), indent=2) + "\n")
    (out / "interpretation.txt").write_text(
        "mesh0 vs mesh1: new-mesh transfer. test_fine changes only solver ground truth.\n"
        "Solver-to-solver gap is context for resolution sensitivity, not a model error floor.\n"
        "Solidification on test_fine is diagnostic only. Do not pool different strides.\n")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
