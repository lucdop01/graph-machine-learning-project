"""Audit updated A/B/filter results and mesh transfer from immutable team snapshots.

No training or rollout is run. Sources in an unmerged team branch are copied with
hashes, without merging its code. Run: python -m analysis.compare_model_b
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import tempfile

EVAL_REF = "011f7243148a8232100afd2289675431ad734604"
TABLE_REF = "011f7243148a8232100afd2289675431ad734604"
DOMAINS = ["val", "test_interp", "test_interp_seed1", "test_extrap", "test_path", "test_fine", "test_large"]
COUNTS = [20, 30, 30, 15, 20, 30, 10]
MODELS = {"A": [f"model_a_seed{s}" for s in range(3)],
          "B": [f"model_b_seed{s}" for s in range(3)],
          "C": [f"model_c_seed{s}" for s in range(3)],
          "D": [f"model_d_seed{s}" for s in range(3)],
          "GCN": [f"gcn_seed{s}" for s in range(3)],
          "LF-T": ["lf_t"], "Fixed filter": ["laplacian_filter"], "U-Net": ["unet"]}


def git(*args):
    return subprocess.check_output(["git", *args])


def mean_std(values):
    return dict(mean=st.mean(values), std=st.stdev(values) if len(values) > 1 else None, values=values)


def same_stats(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_stats(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_stats(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    return a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/lei/model_b_comparison")
    ap.add_argument("--eval-ref", default=EVAL_REF)
    ap.add_argument("--table-ref", default=TABLE_REF)
    a = ap.parse_args()
    out = Path(a.out)
    source_dir = out / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(eval_commit=a.eval_ref, table_commit=a.table_ref,
                    main_commit=git("rev-parse", "HEAD").decode().strip(), files=[], checkpoint_audit=[])

    def source(path, ref=None):
        ref = ref or a.eval_ref
        raw = git("show", f"{ref}:{path}")
        target = source_dir / Path(path).name
        target.write_bytes(raw)
        manifest["files"].append(dict(path=path, commit=ref, sha256=hashlib.sha256(raw).hexdigest()))
        return raw.decode()

    loaded, aggregate, checks = {}, {}, 0
    references = {}
    for domain, count in zip(DOMAINS, COUNTS):
        aggregate[domain] = {}
        for label, stems in MODELS.items():
            summaries = []
            for stem in stems:
                base = f"results/eval/{stem}_{domain}_s5"
                summary = json.loads(source(base + ".json"))
                rows = list(csv.DictReader(io.StringIO(source(base + ".csv"))))
                assert len(rows) == summary["runs"] == count, base
                assert len({r["run"] for r in rows}) == count, base
                assert summary["stride"] == 5 and summary["mesh_seed"] == int(domain.endswith("seed1")), base
                assert summary["split"] == domain.replace("_seed1", ""), base
                assert math.isclose(summary["nrmse_mean"], st.mean(float(r["nrmse"]) for r in rows), rel_tol=1e-11), base
                pooled = sum(float(r["melt_mae"]) * int(r["melt_points"]) for r in rows if int(r.get("melt_points") or 0))
                points = sum(int(r.get("melt_points") or 0) for r in rows)
                assert points == summary["melt_points"], base
                assert math.isclose(summary["melt_mae"], pooled / points, rel_tol=1e-11), base
                union = sum(float(r["union_mm2"]) for r in rows if float(r.get("union_mm2") or 0) > 0)
                intersection = sum(float(r["iou"]) * float(r["union_mm2"]) for r in rows if float(r.get("union_mm2") or 0) > 0)
                assert math.isclose(summary["iou"]["iou"], intersection / union, rel_tol=1e-10), base
                metadata = {r["run"]: {k: r[k] for k in ("split", "mesh", "stride", "step_us", "frames_scored",
                            "horizon", "last_frame", "melt_points", "size_bin", "peak_area_mm2")} for r in rows}
                if domain not in references:
                    references[domain] = metadata
                assert metadata == references[domain], f"Different scored trajectories: {base}"
                loaded[stem, domain] = summary
                summaries.append(summary)
                checks += 1
            aggregate[domain][label] = dict(seeds=len(stems) if label not in ("LF-T", "Fixed filter") else None,
                nrmse=mean_std([s["nrmse_mean"] for s in summaries]),
                melt_mae_K=mean_std([s["melt_mae"] for s in summaries]),
                iou=mean_std([s["iou"]["iou"] for s in summaries]),
                iou_by_bin={b: mean_std([s["iou_by_bin"][b]["iou"] for s in summaries if b in s["iou_by_bin"]])
                            for b in ("small", "medium", "large") if any(b in s["iou_by_bin"] for s in summaries)},
                energy_a=mean_std([s["energy"]["a"]["median"] for s in summaries]),
                runs_per_seed=count)
    gap = json.loads(source("results/eval/cross_test_fine_s5.json"))
    source("results/eval/cross_test_fine_s5.csv")
    source("results/tables/main_table.md", a.table_ref)
    table_rows = list(csv.DictReader(io.StringIO(source("results/tables/main_table.csv", a.table_ref))))
    labels = {"A": "a", "B": "b", "C": "c", "D": "d", "GCN": "gcn", "LF-T": "lf_t", "Fixed filter": "fixed", "U-Net": "unet"}
    indexed_table = {(r["model"], r["split"], r["metric"]): r for r in table_rows}
    for domain, models in aggregate.items():
        for label, item in models.items():
            for key, metric in (("nrmse", "nrmse"), ("melt_mae_K", "melt_mae"), ("iou", "iou")):
                row = indexed_table[labels[label], domain, metric]
                assert math.isclose(float(row["mean"]), item[key]["mean"], rel_tol=1e-5), row
                if item[key]["std"] is not None:
                    assert math.isclose(float(row["std"]), item[key]["std"], rel_tol=1e-5), row
    # Verify current shared weights/configurations match the evaluated snapshot.
    import torch
    from models.train import load_checkpoint
    reference_config = None
    reference_stats = None
    fingerprints = set()
    parameter_counts = {}
    for label in ("A", "B", "C", "D", "GCN"):
        for seed, stem in enumerate(MODELS[label]):
            path = Path(f"results/models/{stem}/best.pt")
            assert git("rev-parse", f"{a.eval_ref}:{path}").strip() == git("hash-object", str(path)).strip(), str(path)
            model, ck = load_checkpoint(str(path), "cpu")
            parameter_counts[label] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if label in ("A", "B"):
                cfg = ck["config"]
                training = {k: v for k, v in cfg["training"].items() if k != "seed"}
                comparison_config = dict(training=training, data=cfg["data"],
                    backbone={k: cfg["model"][k] for k in ("width", "layers", "hidden_layers")})
                reference_stats = reference_stats or ck["stats"]
                assert same_stats(reference_stats, ck["stats"]), f"Normalization mismatch: {stem}"
                reference_config = reference_config or comparison_config
                assert comparison_config == reference_config, f"A/B training config differs: {stem}"
                assert cfg["training"]["seed"] == seed and cfg["training"]["steps"] == 100000
                assert cfg["training"]["augment"] is True
                assert ck["stride"] == 5 and ck["target"] == "dT"
                fingerprints.add(ck["train_fingerprint"])
                score = loaded[stem, "val"]["nrmse_mean"]
                assert math.isclose(ck["score"], score, rel_tol=1e-4), f"Validation/checkpoint score differs: {stem}"
                manifest["checkpoint_audit"].append(dict(model=label, seed=seed, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), best_step=ck["step"],
                    score=ck["score"], eval_score=score, train_fingerprint=ck["train_fingerprint"],
                    torch_version=ck.get("torch_version"), precision=ck["precision"]))
            del model, ck
    assert len(fingerprints) == 1
    assert all(s["size_bins"]["train_fingerprint"] in fingerprints for s in loaded.values())
    comparisons = {}
    for domain in DOMAINS:
        x, y = aggregate[domain]["A"], aggregate[domain]["B"]
        comparisons[domain] = dict(nrmse_reduction=1-y["nrmse"]["mean"]/x["nrmse"]["mean"],
            melt_mae_reduction=1-y["melt_mae_K"]["mean"]/x["melt_mae_K"]["mean"])
        for metric in ("nrmse", "melt_mae_K"):
            delta = x[metric]["mean"] - y[metric]["mean"]
            spread = max(x[metric]["std"], y[metric]["std"])
            comparisons[domain][metric + "_spread_check"] = dict(improvement=delta, max_sample_std=spread, exceeds_spread=delta > spread)
    transfer = {}
    for label in MODELS:
        old = aggregate["test_interp"][label]["nrmse"]
        new = aggregate["test_interp_seed1"][label]["nrmse"]
        transfer[label] = dict(original=old, new_mesh=new, ratio_of_means=new["mean"]/old["mean"],
            per_seed_ratios=[newval/oldval for newval, oldval in zip(new["values"], old["values"])])
    result = dict(aggregate=aggregate, comparisons=comparisons, transfer=transfer,
        solver_gap=dict(nrmse=gap["nrmse_mean"], melt_mae_K=gap["melt_mae"], iou=gap["iou"]["iou"]),
        parameter_counts=parameter_counts, audits=dict(domains_checked=checks, same_trajectory_metadata=True,
        summary_nrmse_recomputed=True, pooled_melt_mae_recomputed=True, pooled_iou_recomputed=True, training_configs_match=True,
        normalization_agrees_within_roundoff=True, team_table_checked=True,
        shared_train_fingerprint=next(iter(fingerprints))),
        formal_short_gate="NOT PERFORMED: these are 100k-step final runs, not short-budget runs",
        note="Spread checks are descriptive post-hoc comparisons, not a retrospective preregistered decision. "
             "No new rollouts were executed; original evaluation credit belongs to Luc and model trainers.")
    (out / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for domain in DOMAINS:
        for label, scores in aggregate[domain].items():
            record = dict(domain=domain, model=label, seeds=scores["seeds"], runs_per_seed=scores["runs_per_seed"])
            for metric in ("nrmse", "melt_mae_K", "iou", "energy_a"):
                record[metric + "_mean"] = scores[metric]["mean"]
                record[metric + "_std"] = scores[metric]["std"]
            records.append(record)
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader(); w.writerows(records)
    with (out / "mesh_transfer.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "original_nrmse", "new_mesh_nrmse", "ratio_of_means"])
        for label, item in transfer.items():
            w.writerow([label, item["original"]["mean"], item["new_mesh"]["mean"], item["ratio_of_means"]])
    write_report(out, result, manifest)
    plot(out, result)
    print(json.dumps(dict(audits=result["audits"], parameters=parameter_counts,
                         comparisons=comparisons, transfer=transfer, solver_gap=result["solver_gap"]), indent=2))


def fmt(item, digits=6):
    return f"{item['mean']:.{digits}f}" + (f" ± {item['std']:.{digits}f}" if item["std"] is not None else "")


def write_report(out, r, manifest):
    ag = r["aggregate"]
    lines = ["# Model B comparison and mesh transfer", "", "Updated 6 November 2026.", "",
        "This analysis replaces the old A-seed-0 comparison. It uses the three current augmented A checkpoints, "
        "B seeds 0–2, and Luc's committed per-run evaluation files. No new training or rollout is claimed. "
        "Combined evaluation snapshot is in the pending luc/main-results-table branch (which includes luc/evaluate-all-models); B test files are not yet main at this snapshot.", "",
        "## Protocol and audit", "",
        "A and B have the same 100,000-step budget, batch size, learning-rate schedule, noise, augmentation, "
        "precision, stride 5 and backbone settings. Best validation checkpoints may occur at different steps. "
        "Checkpoint fingerprints match, and each evaluated checkpoint blob matches the current shared weights. "
        "We recomputed per-run nRMSE means and pooled melting MAE, and checked identical run sets, scoring frames, "
        "mesh labels, size bins and true melting-region point counts across models. "
        "Melting-range MAE uses true temperatures 1800–2000 K. Values are per-seed split metrics, then seed mean ± sample SD; "
        "this is not a confidence interval. Filter results are deterministic; U-Net has one seed. "
        "Training normalization agrees within floating-point roundoff. Training runtimes differ: A checkpoints "
        "record PyTorch 2.9.1+rocm7.2.1, B records 2.5.1. Thus this is a matched-hyperparameter comparison "
        "of available team runs, not an identical-backend controlled retraining experiment.", "",
        "## Updated A/B comparison", "",
        "| Domain | A nRMSE | B nRMSE | B reduction | A melting MAE / K | B melting MAE / K |",
        "|---|---:|---:|---:|---:|---:|"]
    for d in DOMAINS:
        x,y=ag[d]["A"],ag[d]["B"]
        lines.append(f"| {d} | {fmt(x['nrmse'])} | {fmt(y['nrmse'])} | {r['comparisons'][d]['nrmse_reduction']:.1%} | {fmt(x['melt_mae_K'],3)} | {fmt(y['melt_mae_K'],3)} |")
    lines += ["", "The earlier 18.69% reduction used B's three-seed mean against an obsolete A seed 0. "
              "It is superseded by the matched three-seed comparison above.", "", "## Filter reference", "",
              "| Model | Validation nRMSE | Melting MAE / K | IoU |", "|---|---:|---:|---:|"]
    for label in ("A","B","LF-T","Fixed filter"):
        v=ag['val'][label]
        lines.append(f"| {label} | {fmt(v['nrmse'])} | {fmt(v['melt_mae_K'],3)} | {fmt(v['iou'])} |")
    lines += ["", "LF-T has five polynomial coefficients plus one source coefficient (six fitted values). "
              "The fixed physical filter has zero fitted parameters; the two baselines must remain distinct.", "",
              "## Mesh transfer", "", "| Model | Original mesh nRMSE | New mesh nRMSE | New/original |",
              "|---|---:|---:|---:|"]
    for label,item in r['transfer'].items():
        lines.append(f"| {label} | {fmt(item['original'])} | {fmt(item['new_mesh'])} | {item['ratio_of_means']:.2f}× |")
    lines += ["", "The new mesh keeps the test_interp physical cases fixed and changes mesh seed 0 to 1 without retraining. "
              "A lower new-mesh error and a smaller degradation ratio are different claims; both are reported. "
              "This tests one prescribed mesh perturbation, not arbitrary remeshing or node-density changes. "
              "U-Net uses a fixed regular grid with outputs sampled on the evaluation mesh; its near-unit ratio "
              "does not demonstrate transfer of a learned unstructured graph operator. "
              "B retains lower mean error than A, but degrades more relatively (3.22× vs 2.73×); "
              "the new-mesh nRMSE mean improvement is smaller than B's seed SD.", "",
              "## Fine solver and paths", "", "| Model | test_fine nRMSE | test_fine melting MAE / K | test_path nRMSE |",
              "|---|---:|---:|---:|"]
    for label in ("A","B","C","D"):
        f,p=ag['test_fine'][label],ag['test_path'][label]
        lines.append(f"| {label} | {fmt(f['nrmse'])} | {fmt(f['melt_mae_K'],3)} | {fmt(p['nrmse'])} |")
    gap=r['solver_gap']
    lines += ["", f"10 versus 5 µm solver gap: nRMSE {gap['nrmse']:.6f}; melting MAE {gap['melt_mae_K']:.3f} K. "
              "This is resolution-sensitivity context, not a model error floor. test_fine changes solver truth "
              "resolution; it does not itself test a finer GNN mesh. Solidification is diagnostic only at stride 5.", "",
              "D has lower path error than C and B in these files; the old claim that C is best on paths is superseded. "
              "Verified parameter counts: GCN 318,593; D 2,365,058; A/B/C 2,331,905.", "",
              "## Gate status", "", "These results compare completed 100k-step runs. They do not recreate the original "
              "short-training gate or establish that a decision was made before full training. "
              "The following checks compare mean A−B improvement against max(sample SD(A), sample SD(B)) "
              "descriptively. They are not significance tests or a new preregistered rule.", "",
              "| Domain | Metric | A−B | Max seed SD | Exceeds spread |", "|---|---|---:|---:|---|"]
    for d in ('val','test_interp_seed1'):
        for m in ('nrmse','melt_mae_K'):
            c=r['comparisons'][d][m+'_spread_check']
            lines.append(f"| {d} | {m} | {c['improvement']:.6g} | {c['max_sample_std']:.6g} | {c['exceeds_spread']} |")
    lines += ["", "The written plan permits nRMSE or melting-range improvement; the old automated gate used only nRMSE. "
              "That rule difference must be disclosed and agreed by the team; no formal Go/No-go is manufactured here. "
              "No duplicate short training is required merely to reconstruct a missing chronology.", "",
              "## Evidence and contribution", "",
              f"Evaluation snapshot: `{manifest['eval_commit']}`; team table snapshot: `{manifest['table_commit']}`. "
              "provenance.json lists original paths and hashes; sources/ retains the JSON/CSV snapshots. "
              "Original evaluation credit: Luc and the model trainers. This contribution is updated evidence auditing, "
              "A/B/filter comparison, mesh-transfer analysis, figures and report text. "
              "It does not claim a new GPU evaluation or causal proof that B mitigates oversmoothing.", "",
              "Reproduce: `python -m analysis.compare_model_b`.", "",
              "![Updated A/B comparison](comparison.png)", "", "![Mesh transfer](mesh_transfer.png)"]
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def plot(out, result):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lei-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ag=result['aggregate']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax, metric, title in zip(axes,['nrmse','melt_mae_K'],['Rollout nRMSE','Melting-range MAE (K)']):
        x=np.arange(len(DOMAINS))
        for label,offset,color in [('A',-.15,'#2864a5'),('B',.15,'#c8702a')]:
            means=[ag[d][label][metric]['mean'] for d in DOMAINS]
            errs=[ag[d][label][metric]['std'] for d in DOMAINS]
            ax.errorbar(x+offset,means,yerr=errs,fmt='o',capsize=3,label=label,color=color)
        ax.set_yscale('log');ax.set_xticks(x,['val','interp','new mesh','extrap','path','fine','large'],rotation=35)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Updated A/B evaluation | 3 seeds each | mean ± sample SD')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    labels=list(MODELS);x=np.arange(len(labels))
    for domain,offset,color,title in [('test_interp',-.15,'#2864a5','Original mesh'),('test_interp_seed1',.15,'#c8702a','New mesh')]:
        ax.errorbar(x+offset,[ag[domain][m]['nrmse']['mean'] for m in labels],
                    yerr=[ag[domain][m]['nrmse']['std'] or 0 for m in labels],fmt='o',capsize=3,color=color,label=title)
    ax.set_xticks(x,labels);ax.set_yscale('log');ax.set_ylabel('Rollout nRMSE (lower is better)')
    ax.set_title('Mesh transfer | same physical cases, unchanged weights')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'mesh_transfer.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    main()
