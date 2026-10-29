"""Paired A/B 20k-step seed-0 pilot with the team's fitted LF-T reference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

OUT = Path('results/lei/short_gate_seed0')


def run_name(model):
    return f'lei_short20k_{model.lower()}_seed0'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-model', choices=['A', 'B'])
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    ap.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'])
    a = ap.parse_args()
    if bool(a.train_model) == a.evaluate:
        ap.error('Choose --train-model or --evaluate')
    print('START short gate: 20,000 training steps, seed 0 only', flush=True)
    import numpy as np
    import torch
    import yaml
    from analysis.run_model_b_gate import gate_configs, check_data, validate_checkpoint, sha256
    from models.train import load_checkpoint
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on a GPU compute node')
    cfgs = gate_configs(20000, a.device, a.precision)
    stats, bins = check_data()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.train_model:
        cfg = cfgs[a.train_model]
        cfg['training']['seed'] = 0
        name = run_name(a.train_model)
        config = OUT / f'{name}.yaml'
        last = Path('checkpoints') / name / 'last.pt'
        args = [sys.executable, '-u', '-m', 'models.train', '--name', name]
        if last.exists():
            ck = torch.load(last, map_location='cpu', weights_only=True)
            validate_checkpoint(ck, cfg, stats)
            if ck['step'] >= 20000:
                print(f'ALREADY COMPLETE: {name}', flush=True)
                return
            args.append('--resume')
        else:
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            args.extend(['--config', str(config)])
        subprocess.run(args, check=True)
        print(f'COMPLETE training: {name}', flush=True)
        return

    from graphs.dataset import load_runs
    from evaluation.rollout import evaluate_run, summarise, write_results
    from evaluation.torch_predictor import TorchPredictor
    from analysis.team_lf_filter import load_predictor
    rows = []
    manifest = dict(status='running', seeds=[0], steps=20000, formal_gate=False,
                    interpretation='Single-seed pilot; seed spread cannot be estimated.', checkpoints={})
    (OUT / 'decision.json').unlink(missing_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for label in ['A', 'B', 'LF-T']:
        if label == 'LF-T':
            predictor = load_predictor()
            model = None
        else:
            path = Path('checkpoints') / run_name(label) / 'best.pt'
            model, ck = load_checkpoint(str(path), a.device)
            cfg = cfgs[label]
            cfg['training']['seed'] = 0
            validate_checkpoint(ck, cfg, stats)
            if ck['size_bins'] != bins:
                raise ValueError('Checkpoint and evaluation size bins differ')
            manifest['checkpoints'][label] = dict(path=str(path), sha256=sha256(path), best_step=ck['step'])
            predictor = TorchPredictor(model, ck['stats'], 5, ck['target'], a.device, name=run_name(label))
        for domain, split, mesh in [('val', 'val', 0), ('new_mesh', 'test_interp', 1)]:
            runs = load_runs(split, mesh, preload=False)
            per_run, parts = [], []
            for run in runs:
                print(f'RUN {label} {domain}: {run.name}', flush=True)
                r, p = evaluate_run(run, predictor, 5, np.asarray(bins['edges_m2']))
                per_run.append(r)
                parts.append(p)
                summary = summarise(per_run, parts, bins)
                summary.update(mesh_seed=mesh, planned_runs=len(runs), completed=len(per_run) == len(runs),
                               training_seed=0 if label != 'LF-T' else None, formal_gate=False)
                write_results(per_run, summary, str(OUT), f'{label}_{domain}')
                print(f'SAVED {label} {domain}: {len(per_run)}/{len(runs)}', flush=True)
            rows.append(dict(model=label, domain=domain, nrmse=summary['nrmse_mean'], melt_mae=summary['melt_mae']))
        del predictor, model
    with (OUT / 'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(formal_gate='NOT DETERMINED', seeds=[0], note=manifest['interpretation'], domains={})
    for domain in ['val', 'new_mesh']:
        values = {r['model']: r for r in rows if r['domain'] == domain}
        result['domains'][domain] = dict(values=values,
            b_lower_nrmse_than_a=values['B']['nrmse'] < values['A']['nrmse'],
            b_lower_melt_mae_than_a=values['B']['melt_mae'] < values['A']['melt_mae'])
    (OUT / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Short gate: seed 0', '', 'A and B: 20,000 steps each. LF-T: existing train-fitted baseline.', '',
             '| Domain | Model | nRMSE | Melt MAE (K) |', '|---|---|---:|---:|']
    lines += [f'| {r["domain"]} | {r["model"]} | {r["nrmse"]:.7f} | {r["melt_mae"]:.4f} |' for r in rows]
    lines += ['', 'Single-seed pilot: no seed standard deviation or formal three-seed verdict.']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    manifest['status'] = 'complete'
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'COMPLETE: {OUT / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
