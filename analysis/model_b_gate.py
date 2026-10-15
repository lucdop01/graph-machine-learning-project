"""Apply a preregistered Model B gate to shared-harness rollout results.

Input CSV: model,seed,domain,nrmse. model is A/B/LF; domain is val/new_mesh.
Each group must contain exactly the same three seeds and finite, nonnegative
mean rollout nRMSE values. 'new_mesh' means test_interp on mesh seed 1, with stride 5. Fixed LF is repeated under the same seeds for provenance.
"""
import argparse
import csv
import json
import numpy as np


def decide(rows):
    groups = {(m, d): {} for m in ("A", "B", "LF") for d in ("val", "new_mesh")}
    for row in rows:
        key = row["model"], row["domain"]
        if key not in groups:
            raise ValueError(f"unknown group {key}")
        seed, value = int(row["seed"]), float(row["nrmse"])
        if seed in groups[key] or not np.isfinite(value) or value < 0:
            raise ValueError(f"duplicate seed or invalid nrmse in {key}")
        groups[key][seed] = value
    seeds = set(groups[("A", "val")])
    if len(seeds) != 3 or any(set(g) != seeds for g in groups.values()):
        raise ValueError("all six groups need the same exactly three seeds")
    summary = {}
    for domain in ("val", "new_mesh"):
        values = {m: np.array([groups[m, domain][s] for s in sorted(seeds)]) for m in ("A", "B", "LF")}
        means = {m: float(v.mean()) for m, v in values.items()}
        stds = {m: float(v.std(ddof=1)) for m, v in values.items()}
        margin = max(stds["A"], stds["B"])
        summary[domain] = dict(mean=means, std=stds, margin=margin,
                               improvement=means["A"] - means["B"],
                               passed=bool(means["A"] - means["B"] > margin))
    return dict(decision="go" if all(s["passed"] for s in summary.values()) else "no-go",
                seeds=sorted(seeds), domains=summary,
                rule="B improves on A nRMSE by more than max sample std(A,B) in both domains; LF reported as reference")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default="results/model_b_gate.json")
    a = ap.parse_args()
    with open(a.csv, newline="") as f:
        result = decide(list(csv.DictReader(f)))
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
