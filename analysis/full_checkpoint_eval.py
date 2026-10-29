"""Score one shared checkpoint (or baselines) on all of Lei's evaluation domains."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', required=True, type=int, choices=range(5))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    print(f'START full evaluation task {a.task}: loading libraries', flush=True)
    import numpy as np
    from graphs.dataset import load_runs, load_stats, current_train_fingerprint
    from evaluation.rollout import (checkpoint_predictor, evaluate_run, summarise, write_results,
                                    load_size_bins, CrossSolver, find_pair)
    from models.linear_filter import LinearFilter
    from analysis.team_lf_filter import load_predictor
    out = Path('results/lei/full_eval')
    out.mkdir(parents=True, exist_ok=True)
    stats, bins = load_stats(5), load_size_bins()
    fingerprint = current_train_fingerprint()
    if stats['train_fingerprint'] != fingerprint or bins['train_fingerprint'] != fingerprint:
        raise ValueError('Current training data/statistics/size bins do not match')
    paths = ['results/models/model_a_seed0/best.pt'] + [
        f'results/models/model_b_seed{s}/best.pt' for s in range(3)]
    if a.task < 4:
        path = paths[a.task]
        predictor, stride, ck_bins = checkpoint_predictor(path, a.device)
        from models.train import load_checkpoint
        # Fail instead of silently scoring weights against unrelated training data.
        _, ck = load_checkpoint(path, 'cpu')
        if ck['train_fingerprint'] != fingerprint or ck_bins != bins:
            raise ValueError('Checkpoint and current data fingerprint/size bins differ')
        del ck
        predictors = [(predictor, stride, dict(checkpoint=path,
            checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))]
    else:
        predictors = [(load_predictor(), 5, dict(baseline='team fitted LF-T K5')),
                      (LinearFilter(), 5, dict(baseline='fixed physical diffusion'))]
    domains = [('val', 0), ('test_interp', 0), ('test_interp', 1), ('test_fine', 0)]
    for split, mesh in domains:
        runs = load_runs(split, mesh, preload=False)
        partners = load_runs('test_interp', 0, preload=False) if split == 'test_fine' else None
        active = list(predictors)
        if a.task == 4 and split == 'test_fine':
            active.append((CrossSolver(partners), 5, dict(baseline='solver gap, not model error floor')))
        for predictor, stride, provenance in active:
            rows, parts = [], []
            stem = f'{predictor.name}_{split}_mesh{mesh}_s{stride}'
            for run in runs:
                print(f'RUN {stem}: {run.name}', flush=True)
                row, part = evaluate_run(run, predictor, stride, np.asarray(bins['edges_m2']),
                                         find_pair(run, partners) if partners is not None else None)
                rows.append(row)
                parts.append(part)
                summary = summarise(rows, parts, bins)
                summary.update(provenance, mesh_seed=mesh, train_fingerprint=fingerprint,
                               planned_runs=len(runs), completed=len(rows) == len(runs), formal_gate=False,
                               solidification_diagnostic_only=split == 'test_fine')
                write_results(rows, summary, str(out), stem)
                print(f'SAVED {stem} {len(rows)}/{len(runs)} nRMSE={summary["nrmse_mean"]:.7f}', flush=True)
    (out / f'task_{a.task}_complete.json').write_text(json.dumps(dict(task=a.task, completed=True)) + '\n')
    print(f'COMPLETE full evaluation task {a.task}', flush=True)


if __name__ == '__main__':
    main()
