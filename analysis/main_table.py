"""Main results table and the proposal's hypothesis test, from the files in results/eval/ (plan: main comparison A-D).

    python -m analysis.main_table      results/tables/main_table.md, main_table.csv and hypothesis.csv

Every value comes from the split summaries that evaluation.rollout writes; nothing is re-run. Each metric is taken per
training seed as the file pools it, then reported as the mean and sample standard deviation over seeds. Parameter counts
are read from the checkpoints when torch is installed.

Hypothesis (proposal): the enthalpy models C and D cut the melting-range error by at least 20% relative to the best
temperature-predicting GNN, with a gain under 10% on overall error (nRMSE). The reference is chosen once, as the GCN, A or
B with the lowest mean val melting-range MAE, so test results do not pick it. A difference counts as beyond the seed
spread when it exceeds the larger of the two seed standard deviations (the rule of the Model B gate).
"""
import csv
import json
import statistics as st
from pathlib import Path

EVAL = Path("results/eval")
OUT = Path("results/tables")
SEEDS = (0, 1, 2)
DOMAINS = [("val", "val"), ("test_interp", "test_interp"), ("test_interp_seed1", "test_interp, new mesh (seed 1)"),
           ("test_extrap", "test_extrap"), ("test_path", "test_path"), ("test_fine", "test_fine (5 um truth)"),
           ("test_large", "test_large")]
SHORT = {"test_interp_seed1": "new mesh", "test_interp": "interp", "test_extrap": "extrap", "test_path": "path",
         "test_fine": "fine", "test_large": "large", "val": "val"}

# key, label, file stems, target, built-in physics, checkpoint for the parameter count
MODELS = [
    ("gcn", "GCN", ["gcn_seed{}"], "dT", "none; ignores edge geometry", "results/models/gcn_seed0/best.pt"),
    ("a", "A: MeshGraphNet", ["model_a_seed{}"], "dT", "none", "results/models/model_a_seed0/best.pt"),
    ("b", "B: + linear diffusion", ["model_b_seed{}"], "dT", "fixed linear heat diffusion branch",
     "results/models/model_b_seed0/best.pt"),
    ("c", "C: enthalpy output", ["model_c_seed{}"], "dH", "exact material table T(H)",
     "results/models/model_c_seed0/best.pt"),
    ("d", "D: conservative enthalpy", ["model_d_seed{}"], "dH (edge fluxes)", "table and exact heat conservation",
     "results/models/model_d_seed0/best.pt"),
    ("unet", "U-Net (grid)", ["unet"], "dT", "none; regular 5 um grid", "checkpoints/unet_early_stop_retry_group_s5_seed0.pt"),
    ("lf_t", "LF-T (fitted linear filter)", ["lf_t"], "T", "linear Laplacian polynomial, fitted on train", None),
    ("fixed", "Fixed linear filter", ["laplacian_filter"], "T", "linear diffusion, constant k and cp, nothing fitted", None),
    ("a_noaug", "A without augmentation (ablation)", ["a_seed{}"], "dT", "none", "results/models/model_a_seed0/best.pt"),
]
TEMPERATURE_GNNS = ("gcn", "a", "b")
OURS = ("c", "d")
LF_T_PARAMETERS = 6          # five filter orders and the laser coefficient (models/artifacts/lf_t_s5_k5.json)


def stems(entry):
    names = entry[2]
    return [n.format(s) for n in names for s in SEEDS] if "{}" in names[0] else list(names)


def load(stem, domain):
    path = EVAL / f"{stem}_{domain}_s5.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    bins = s["iou_by_bin"]                       # a split may have no run in a bin
    iou = {b: bins.get(b, {}).get("iou") for b in ("small", "medium", "large")}
    none = bins.get("none", {})
    return dict(nrmse=s["nrmse_mean"], melt_mae=s["melt_mae"], iou=s["iou"]["iou"],
                iou_small=iou["small"], iou_medium=iou["medium"], iou_large=iou["large"],
                spurious=none.get("runs_with_predicted_pool", 0), no_pool_runs=none.get("runs", 0),
                energy_a=s["energy"]["a"]["median"], energy_end=s["energy"]["end"]["median"], runs=s["runs"])


def parameters(path):
    if path is None:
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if path.startswith("checkpoints/unet"):
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
        return sum(v.numel() for v in state.values())
    from models.train import load_checkpoint
    model, _ = load_checkpoint(path, "cpu")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return st.mean(values), (st.stdev(values) if len(values) > 1 else None), len(values)


def fmt(mean, std, digits, pct=False):
    if mean is None:
        return "-"
    k = 100 if pct else 1
    s = f"{mean * k:.{digits}f}"
    if std is not None:
        s += f" ± {std * k:.{digits}f}"
    return s + ("%" if pct else "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}                                  # (model key, domain) -> list of per-seed metric dicts
    missing = []
    for entry in MODELS:
        for domain, _ in DOMAINS:
            per_seed = [load(stem, domain) for stem in stems(entry)]
            missing += [f"{stem}_{domain}" for stem, r in zip(stems(entry), per_seed) if r is None]
            results[entry[0], domain] = [r for r in per_seed if r is not None]
    cross = load("cross", "test_fine")
    summary = {}
    for (key, domain), rows in results.items():
        for metric in ("nrmse", "melt_mae", "iou", "iou_small", "iou_medium", "iou_large", "energy_a", "energy_end"):
            summary[key, domain, metric] = stats([r[metric] for r in rows])

    with (OUT / "main_table.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "label", "split", "metric", "mean", "std", "seeds"])
        for (key, domain, metric), (m, s, n) in summary.items():
            w.writerow([key, dict((e[0], e[1]) for e in MODELS)[key], domain, metric,
                        "" if m is None else f"{m:.6g}", "" if s is None else f"{s:.6g}", n])

    # hypothesis test
    reference = min(TEMPERATURE_GNNS, key=lambda k: summary[k, "val", "melt_mae"][0])
    hyp = []
    for domain, _ in DOMAINS:
        rm, rs, _ = summary[reference, domain, "melt_mae"]
        rn, rns, _ = summary[reference, domain, "nrmse"]
        for key in OURS:
            mm, ms, _ = summary[key, domain, "melt_mae"]
            nm, ns, _ = summary[key, domain, "nrmse"]
            melt_change, nrmse_change = mm / rm - 1, nm / rn - 1
            hyp.append(dict(split=domain, model=key, reference=reference,
                            melt_mae=mm, melt_mae_std=ms, reference_melt_mae=rm, reference_melt_mae_std=rs,
                            melt_change=melt_change, melt_beyond_spread=abs(mm - rm) > max(ms, rs),
                            nrmse=nm, reference_nrmse=rn, nrmse_change=nrmse_change,
                            nrmse_beyond_spread=abs(nm - rn) > max(ns, rns),
                            melt_target_met=melt_change <= -0.20, specific_to_melting=-nrmse_change < 0.10))
    with (OUT / "hypothesis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(hyp[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in h.items()} for h in hyp)

    label = {e[0]: e[1] for e in MODELS}
    seeds = {e[0]: len(stems(e)) for e in MODELS}
    lines = ["# Main results", "",
             "Generated by `python -m analysis.main_table` from the split summaries in `results/eval/`; do not edit by "
             "hand. Closed-loop rollouts at stride 5 (66.7 us) from the true initial state. Each metric is pooled per "
             "training seed as its file pools it, then shown as mean ± sample standard deviation over seeds.", "",
             "Melting-range MAE: mean absolute error where the true temperature is 1,800-2,000 K. IoU: area-weighted "
             "overlap of melted fraction above 0.5, pooled over runs, overall and per melt-pool size bin (terciles of "
             "the train split's peak pool area). Spurious pools: runs without a true pool in which the model predicts "
             "one, summed over seeds. Energy error (a): median over runs of the largest absolute energy error relative "
             "to the laser input so far, after 5% of the input. Final energy error: median signed error at the end, "
             "relative to the total laser input. Solidification durations are not reported: at stride 5 they are a "
             "diagnostic only (README, evaluation notes).", ""]
    if missing:
        lines += [f"**Missing files:** {', '.join(missing)}", ""]

    lines += ["## Models", "", "| Model | Target | Built-in physics | Parameters | Seeds |", "|---|---|---|---|---|"]
    for e in MODELS:
        p = LF_T_PARAMETERS if e[0] == "lf_t" else 0 if e[0] == "fixed" else parameters(e[5])
        n = "- (deterministic)" if e[0] in ("lf_t", "fixed") else seeds[e[0]]
        lines.append(f"| {e[1]} | {e[3]} | {e[4]} | {'-' if p is None else f'{p:,}'} | {n} |")

    for metric, title, digits in (("nrmse", "nRMSE", 4), ("melt_mae", "Melting-range MAE (K)", 1)):
        lines += ["", f"## Overview: {title}", "", "| Model | " + " | ".join(SHORT[d] for d, _ in DOMAINS) + " |",
                  "|---" * (len(DOMAINS) + 1) + "|"]
        best = {d: min(summary[e[0], d, metric][0] for e in MODELS if e[0] != "a_noaug") for d, _ in DOMAINS}
        for e in MODELS:
            cells = []
            for d, _ in DOMAINS:
                m = summary[e[0], d, metric][0]
                s = fmt(m, None, digits)
                cells.append(f"**{s}**" if e[0] != "a_noaug" and m == best[d] else s)
            lines.append(f"| {e[1]} | " + " | ".join(cells) + " |")
        lines += ["", "Bold: lowest mean, excluding the ablation."]

    lines += ["", "## Per split"]
    for domain, title in DOMAINS:
        lines += ["", f"### {title}", "",
                  "| Model | nRMSE | Melting-range MAE (K) | IoU | IoU small | IoU medium | IoU large | Spurious pools "
                  "| Energy error (a) | Final energy error |", "|---|---|---|---|---|---|---|---|---|---|"]
        for e in MODELS:
            rows = results[e[0], domain]
            if not rows:
                continue
            g = lambda m, d, pct=False: fmt(*summary[e[0], domain, m][:2], d, pct)
            spurious = f"{sum(r['spurious'] for r in rows)} of {sum(r['no_pool_runs'] for r in rows)}"
            lines.append(f"| {e[1]} | {g('nrmse', 5)} | {g('melt_mae', 1)} | {g('iou', 3)} | {g('iou_small', 3)} | "
                         f"{g('iou_medium', 3)} | {g('iou_large', 3)} | {spurious} | {g('energy_a', 2, True)} | "
                         f"{g('energy_end', 2, True)} |")
        if domain == "test_fine" and cross:
            lines.append(f"| Solver, 10 vs 5 um grid (reference gap) | {cross['nrmse']:.5f} | {cross['melt_mae']:.1f} | "
                         f"{cross['iou']:.3f} | - | - | - | - | {cross['energy_a'] * 100:.2f}% | "
                         f"{cross['energy_end'] * 100:.2f}% |")

    lines += ["", "## Hypothesis test", "",
              f"Reference: the temperature-predicting GNN with the lowest val melting-range MAE, "
              f"**{label[reference]}** ({summary[reference, 'val', 'melt_mae'][0]:.2f} K). The proposal expects C and "
              "D to be at least 20% lower in melting-range MAE with a gain under 10% in nRMSE, which would show that "
              "the gain comes from the phase change. Changes are relative to the reference's mean; \"beyond spread\" "
              "means the difference exceeds the larger of the two seed standard deviations.", "",
              "| Split | Model | Melting-range MAE (K) | vs reference | Beyond spread | nRMSE vs reference | Beyond spread "
              "| Expected pattern |", "|---|---|---|---|---|---|---|---|"]
    for h in hyp:
        pattern = ("yes" if h["melt_target_met"] and h["specific_to_melting"]
                   else "melting target met, but overall gain is larger" if h["melt_target_met"] else "no")
        lines.append(f"| {SHORT[h['split']]} | {label[h['model']][:1]} | {fmt(h['melt_mae'], h['melt_mae_std'], 1)} "
                     f"vs {fmt(h['reference_melt_mae'], h['reference_melt_mae_std'], 1)} | "
                     f"{h['melt_change'] * 100:+.0f}% | {'yes' if h['melt_beyond_spread'] else 'no'} | "
                     f"{h['nrmse_change'] * 100:+.0f}% | {'yes' if h['nrmse_beyond_spread'] else 'no'} | {pattern} |")
    met = [h for h in hyp if h["melt_target_met"]]
    lines += ["", f"The melting-range target is met in {len(met)} of {len(hyp)} model and split pairs"
              + (": " + ", ".join(f"{label[h['model']][:1]} on {SHORT[h['split']]}" for h in met) if met else "") + "."]
    (OUT / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'main_table.md'}, main_table.csv and hypothesis.csv; reference {reference}; "
          f"missing files: {len(missing)}")


if __name__ == "__main__":
    main()
