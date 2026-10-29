"""Independent complete-rollout evaluation on a small, explicitly labelled subset."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=int, default=2, help='0 means the entire split')
    ap.add_argument('--split', choices=['val', 'test_interp'], default='val')
    ap.add_argument('--mesh-seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/lei/quick_eval')
    ap.add_argument('--checkpoints', nargs='+', default=[
        'results/models/model_a_seed0/best.pt', 'results/models/model_b_seed0/best.pt'])
    a = ap.parse_args()
    if a.runs < 0:
        ap.error('--runs must be nonnegative')
    print('START: loading libraries; no training will be performed', flush=True)
    import numpy as np
    import torch
    from graphs.dataset import load_runs
    from evaluation.rollout import checkpoint_predictor, evaluate_run, summarise, write_results
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: submit this script through the GPU batch job')
    runs = load_runs(a.split, a.mesh_seed, preload=False)
    total = len(runs)
    runs = runs[:a.runs] if a.runs else runs
    out = Path(a.out) / f'{a.split}_mesh{a.mesh_seed}_n{len(runs)}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'DATA READY: {len(runs)}/{total} complete trajectories, mesh {a.mesh_seed}', flush=True)
    started = time.monotonic()
    comparison = {}
    for path in a.checkpoints:
        predictor, stride, bins = checkpoint_predictor(path, a.device)
        rows, parts = [], []
        edges = None if bins is None else np.asarray(bins['edges_m2'])
        for run in runs:
            print(f'RUN: {predictor.name} {run.name}', flush=True)
            row, part = evaluate_run(run, predictor, stride, edges)
            rows.append(row)
            parts.append(part)
            summary = summarise(rows, parts, bins)
            summary.update(checkpoint=path, checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                           mesh_seed=a.mesh_seed, planned_runs=len(runs), full_split_runs=total,
                           subset=len(runs) != total, completed=len(rows) == len(runs),
                           independent_rollout=True, formal_gate=False)
            write_results(rows, summary, str(out), f'{predictor.name}_s{stride}')
            print(f'SAVED {len(rows)}/{len(runs)}: nRMSE={summary["nrmse_mean"]:.7f}', flush=True)
        comparison[predictor.name] = summary
        del predictor
        if a.device == 'cuda':
            torch.cuda.empty_cache()
    (out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(f'COMPLETE in {time.monotonic() - started:.1f}s: {out}', flush=True)
    print('Subset results are preliminary; they are not full-split or three-seed gate results.', flush=True)


if __name__ == '__main__':
    main()
