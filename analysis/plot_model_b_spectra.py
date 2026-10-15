"""Create report figures and numerical summaries from analysis.spectra JSON.

python -m analysis.plot_model_b_spectra
No checkpoints or GPU required. The comparison measures temporal integration
of the same linear graph operator, not agreement with the nonlinear solver.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lei-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create(source, out):
    records = json.loads(Path(source).read_text())
    if len(records) < 2:
        raise ValueError("provide at least two independently sampled meshes")
    if len({(r["mesh"], r["step_us"], r["alpha_m2_s"]) for r in records}) != 1:
        raise ValueError("compare the same domain, time step and diffusivity")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    rows = []
    colors = ("#2864a5", "#c8702a", "#4a8650")
    for j, r in enumerate(records):
        f = r["frequency_response"]
        lam = np.asarray(f["lambda_m_inv2"])
        exact, discrete = np.asarray(f["exact_heat_kernel"]), np.asarray(f["fixed_filter"])
        z = r["alpha_m2_s"] * r["step_us"] * 1e-6 * lam
        if not np.allclose(exact, np.exp(-z)):
            raise ValueError("stored exact response does not match the operator and time step")
        if not np.allclose(discrete, (1 - z / r["substeps"]) ** r["substeps"]):
            raise ValueError("stored filter response does not match its substep count")
        color, label = colors[j % len(colors)], f"Mesh seed {r['seed']}"
        if j == 0:
            zz = np.geomspace(min(z[1:]), max(z), 1000)
            axes[0].semilogx(zz, np.exp(-zz), color="#222222", lw=2, label="Exact graph heat kernel")
        axes[0].semilogx(z[1:], discrete[1:], color=color, ls="--", label=f"Filter, seed {r['seed']}")
        error = np.abs(discrete - exact)
        axes[1].semilogx(z[1:], error[1:], color=color, label=label)
        values = np.asarray(r["eigenvalues_m_inv2"])
        axes[2].plot(np.arange(1, len(values)), values[1:] / 1e6, "o-", ms=4, color=color, label=label)
        k = int(error.argmax())
        rows.append(dict(mesh_seed=r["seed"], nodes=r["nodes"], substeps=r["substeps"],
                         lambda_max_m_inv2=r["lambda_max_m_inv2"],
                         max_sampled_abs_gain_error=float(error[k]), z_at_max_sampled_error=float(z[k]),
                         zero_mode_m_inv2=float(values[0])))
    axes[0].set(title="A  Homogeneous diffusion response", xlabel=r"$z=\alpha\,\Delta t\,\lambda$", ylabel="Mode amplitude gain", ylim=(-.03, 1.04))
    axes[1].set(title="B  Time-integration discrepancy", xlabel=r"$z=\alpha\,\Delta t\,\lambda$", ylabel="Absolute gain difference")
    axes[2].set(title="C  Low-frequency mesh spectrum", xlabel="Sorted nonzero mode index", ylabel=r"Eigenvalue ($10^6$ m$^{-2}$)")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Model B fixed branch | {records[0]['mesh']} | step {records[0]['step_us']:g} µs", fontsize=13)
    fig.savefig(out / "model_b_spectral_analysis.png")
    fig.savefig(out / "model_b_spectral_analysis.svg")
    svg = out / "model_b_spectral_analysis.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    with (out / "spectral_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    first = np.asarray(records[0]["eigenvalues_m_inv2"])[1:]
    differences = []
    for r in records[1:]:
        values = np.asarray(r["eigenvalues_m_inv2"])[1:]
        if values.shape != first.shape:
            raise ValueError("meshes must report the same number of low modes")
        differences.append(dict(reference_seed=records[0]["seed"], compared_seed=r["seed"],
                                max_relative_sorted_low_mode_difference=float(np.max(abs(values - first) / first))))
    result = dict(source=str(source), step_us=records[0]["step_us"],
                  alpha_m2_s=records[0]["alpha_m2_s"], meshes=rows, low_mode_comparison=differences,
                  limitation="Same linear graph operator, homogeneous source-free diffusion; not nonlinear solver accuracy or trained-model mesh transfer. Max gain error is over the stored frequency samples.")
    (out / "spectral_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="results/lei/spectral_response.json")
    ap.add_argument("--out", default="results/lei/spectral_report")
    a = ap.parse_args()
    create(a.source, a.out)


if __name__ == "__main__":
    main()
