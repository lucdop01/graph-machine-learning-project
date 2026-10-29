"""Audit existing A/B/LF-T validation evidence without rerunning models.

B scores are verified against shared training logs, not independently rerun.
LF-T is read from its remote branch, without overwriting main's fixed filter.
Run: python -m analysis.compare_available_b_results
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
import yaml


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    out = Path("results/lei/available_comparison")
    out.mkdir(parents=True, exist_ok=True)
    a_path = Path("results/eval/model_a_seed0_val_s5.json")
    a = json.loads(a_path.read_text())
    a_rows = list(csv.DictReader(Path("results/eval/model_a_seed0_val_s5.csv").open()))
    a_cfg = Path("results/models/model_a_seed0/config.yaml").read_text()
    if "augment: true" not in a_cfg:
        raise ValueError("A checkpoint no longer matches the augmented reference")
    lf_ref = git("rev-parse", "origin/laplacian-filter-baseline")
    lf_raw = git("show", f"{lf_ref}:results/eval/lf_t_val_s5.json")
    lf = json.loads(lf_raw)
    lf_csv = git("show", f"{lf_ref}:results/eval/lf_t_val_s5.csv")
    lf_rows = list(csv.DictReader(io.StringIO(lf_csv)))
    if {r["run"] for r in lf_rows} != {r["run"] for r in a_rows} or len(a_rows) != 20 or len(lf_rows) != 20:
        raise ValueError("A and LF-T must cover the same 20 validation runs")
    checks = ("run", "split", "mesh", "stride", "step_us", "frames_scored", "horizon", "last_frame", "melt_points")
    indexed = {r["run"]: r for r in lf_rows}
    for r in a_rows:
        if any(r[k] != indexed[r["run"]][k] for k in checks):
            raise ValueError(f"A/LF-T evaluation metadata mismatch for {r['run']}")
    for summary, rows in ((a, a_rows), (lf, lf_rows)):
        np.testing.assert_allclose(summary["nrmse_mean"], np.mean([float(r["nrmse"]) for r in rows]), rtol=1e-12)
        if summary["split"] != "val" or summary["stride"] != 5 or summary["runs"] != 20:
            raise ValueError("unexpected summary domain")
    report_path = Path("MODEL_B_D_EVALUATION_REPORT.md")
    report = report_path.read_text()
    block = report.split("MODEL B (PHYSICS-GUIDED DIFFUSION RESIDUAL) RESULTS", 1)[1].split("3-SEED SCORE", 1)[0]
    matches = re.findall(r"Seed\s+([012])\s*\|\s*Step\s+(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)", block)
    if len(matches) != 3 or {m[0] for m in matches} != {"0", "1", "2"}:
        raise ValueError("expected exactly three B seed results in report")
    b_values = []
    b_audit = []
    a_config = yaml.safe_load(a_cfg)
    for seed, reported_step, reported_score, _ in sorted(matches):
        root = Path(f"results/models/model_b_seed{seed}")
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        with (root / "log.csv").open() as f:
            rows = list(csv.DictReader(f))
        scored = [r for r in rows if r.get("val_score")]
        best = min(scored, key=lambda r: float(r["val_score"]))
        score = float(best["val_score"])
        if int(best["step"]) != int(reported_step) or abs(score - float(reported_score)) > 1e-8:
            raise ValueError(f"B seed {seed} log does not match report")
        if not (root / "best.pt").is_file():
            raise ValueError(f"B seed {seed} checkpoint missing")
        for key in ("steps", "batch_size", "lr", "lr_end", "noise", "augment", "precision"):
            if cfg["training"][key] != a_config["training"][key]:
                raise ValueError(f"A/B config mismatch: {key}")
        b_values.append(score)
        b_audit.append(dict(seed=int(seed), best_step=int(best["step"]), val_score=score,
                           checkpoint=str(root / "best.pt"),
                           checkpoint_sha256=hashlib.sha256((root / "best.pt").read_bytes()).hexdigest()))
    b_mean, b_std = float(np.mean(b_values)), float(np.std(b_values, ddof=1))
    unaug = [json.loads(Path(f"results/eval/a_seed{s}_val_s5.json").read_text()) for s in range(3)]
    records = [
        dict(model="LF-T K5", nrmse=lf["nrmse_mean"], seed_std=None, seeds=0,
             melt_mae_K=lf["melt_mae"], iou=lf["iou"]["iou"], evidence="raw per-run CSV and summary; deterministic fitted baseline"),
        dict(model="A augmented", nrmse=a["nrmse_mean"], seed_std=None, seeds=1,
             melt_mae_K=a["melt_mae"], iou=a["iou"]["iou"], evidence="raw per-run CSV and summary; only seed 0 available"),
        dict(model="B training logs", nrmse=b_mean, seed_std=b_std, seeds=3,
             melt_mae_K=None, iou=None, evidence="best validation scores verified from shared training logs and configs; checkpoints present; no independent rollout"),
        dict(model="A unaugmented reference", nrmse=float(np.mean([d["nrmse_mean"] for d in unaug])),
             seed_std=float(np.std([d["nrmse_mean"] for d in unaug], ddof=1)), seeds=3,
             melt_mae_K=float(np.mean([d["melt_mae"] for d in unaug])),
             iou=float(np.mean([d["iou"]["iou"] for d in unaug])),
             evidence="raw three-seed summaries; augmentation differs, so not a matched B control")]
    result = dict(scope="Existing validation evidence, 20 runs, stride 5; no new model execution",
                  records=records, b_seeds=b_values, b_training_log_audit=b_audit,
                  b_relative_nrmse_reduction_vs_augmented_a_seed0=1 - b_mean / a["nrmse_mean"],
                  formal_gate="NOT DETERMINED",
                  gate_missing=["independent per-run B rollout results", "matched augmented A seeds 1 and 2",
                                "B and baseline scores on test_interp mesh seed 1", "matched short-run training budget"],
                  provenance=dict(main_snapshot=git("rev-parse", "origin/main"), lf_branch_commit=lf_ref,
                                  b_report_sha256=hashlib.sha256(report.encode()).hexdigest(),
                                  a_source=str(a_path), lf_source="results/eval/lf_t_val_s5.json"),
                  a_lf_run_metadata_checked=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "lf_t_val_s5_source.json").write_text(lf_raw + "\n")
    (out / "lf_t_val_s5_source.csv").write_text(lf_csv + "\n")
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lei-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    colors = ("#555555", "#2864a5", "#c8702a", "#8997a6")
    labels = ["LF-T K5 (deterministic; raw)", "A augmented (1 seed; raw)",
              "B (3 seeds; training logs)", "A unaugmented (3 seeds; reference)"]
    for i, r in enumerate(records):
        ax.errorbar(r["nrmse"], i, xerr=r["seed_std"], fmt="D" if i == 2 else "o",
                    color=colors[i], capsize=4, markersize=7, markerfacecolor="white" if i == 2 else colors[i])
        ax.annotate(f"{r['nrmse']:.6f}" + (f" ± {r['seed_std']:.6f}" if r["seed_std"] else ""),
                    (r["nrmse"], i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(.0018, .6)
    ax.set_ylim(3.65, -.65)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Validation rollout nRMSE (log scale; lower is better)")
    ax.set_title("Available A / B / LF-T evidence | val, stride 5")
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, -.015, "B scores verified from training logs; rollouts not independently reproduced. This is not the preregistered gate.",
             ha="center", fontsize=9)
    fig.savefig(out / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
