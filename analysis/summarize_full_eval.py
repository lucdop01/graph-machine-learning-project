"""Summarise complete domains while explicitly reporting missing tasks and controls."""
import csv
import json
from pathlib import Path


def main():
    root = Path('results/lei/full_eval')
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.glob('*_s5.json')):
        s = json.loads(path.read_text())
        if not s.get('completed'):
            continue
        rows.append(dict(model=s['predictor'], split=s['split'], mesh_seed=s['mesh_seed'],
                         runs=s['runs'], nrmse=s['nrmse_mean'], melt_mae_K=s.get('melt_mae'),
                         iou=s.get('iou', {}).get('iou'), source=str(path)))
    if rows:
        with (root / 'scores.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    complete = [i for i in range(5) if (root / f'task_{i}_complete.json').exists()]
    status = dict(completed_tasks=complete, missing_tasks=sorted(set(range(5))-set(complete)),
                  formal_gate='NOT DETERMINED',
                  missing_controls=['matched augmented Model A seeds 1 and 2'],
                  note='Existing 100k-step checkpoints; not the original short-budget gate. '
                       'LF-T and fixed diffusion are distinct baselines. '
                       'Solver gap is context, not a model error floor. '
                       'test_fine solidification is diagnostic only.')
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    lines = ['# Existing-checkpoint full evaluation', '',
             f'Completed tasks: {complete}; missing tasks: {status["missing_tasks"]}.', '',
             '| Model | Split | Mesh seed | Runs | nRMSE |', '|---|---|---:|---:|---:|']
    lines += [f'| {r["model"]} | {r["split"]} | {r["mesh_seed"]} | {r["runs"]} | {r["nrmse"]:.7f} |' for r in rows]
    lines += ['', 'Formal gate: NOT DETERMINED. Missing matched augmented A seeds 1/2.', '', status['note']]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
