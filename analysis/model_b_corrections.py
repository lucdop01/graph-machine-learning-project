"""Teacher-forced correction maps and melting-band concentration for trained B.

Uses sampled ground-truth input states for interpretation, not rollout scores.
Writes per-node NPZ maps and an area-weighted summary; undefined ratios are null.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.train import load_checkpoint
from models.model_b import ModelB
from graphs.dataset import SampleBuilder, load_runs, load_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--frames-per-run", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/lei/corrections")
    a = ap.parse_args()
    if a.frames_per_run < 1:
        ap.error("frames-per-run must be positive")
    model, ckpt = load_checkpoint(a.checkpoint, a.device)
    if not isinstance(model, ModelB):
        ap.error("correction maps require a Model B checkpoint")
    stats, stride = ckpt["stats"], ckpt["stride"]
    if stats != load_stats(stride):
        raise ValueError("checkpoint statistics differ from the current training data")
    builder = SampleBuilder(stats, stride, a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    band_mass = total_mass = band_area = total_area = 0.
    rows = []
    for run in load_runs(a.split, a.mesh_seed):
        for k in np.unique(np.linspace(0, run.last - stride, a.frames_per_run, dtype=int)):
            g = builder.sample(run, int(k), run.T(int(k)))
            with torch.no_grad():
                fixed = model.diffusion(g).cpu().numpy()
                correction = (model.correction(g).reshape(-1) * model.target_std).cpu().numpy()
            T, area = run.T(int(k)), run.mesh.area
            band = (T >= 1800) & (T <= 2000)
            mass = area * np.abs(correction)
            band_mass += float(mass[band].sum())
            total_mass += float(mass.sum())
            band_area += float(area[band].sum())
            total_area += float(area.sum())
            np.savez_compressed(out / f"{run.name}_k{k}.npz", pos=run.mesh.pos, T=T,
                                fixed_dT=fixed, correction_dT=correction, area=area, band=band)
            rows.append(dict(run=run.name, k=int(k), correction_abs_area_integral=float(mass.sum())))
    fraction = band_mass / total_mass if total_mass else None
    share = band_area / total_area if total_area else None
    result = dict(checkpoint=a.checkpoint, split=a.split, mesh_seed=a.mesh_seed,
                  diagnostic="teacher-forced, uniformly sampled frames; area-weighted absolute correction",
                  band_K=[1800, 2000], correction_fraction_in_band=fraction, area_fraction_in_band=share,
                  concentration_ratio=fraction / share if fraction is not None and share else None, samples=rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
