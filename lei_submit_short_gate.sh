#!/bin/bash
# Optional argument: the existing data preparation job ID.
set -euo pipefail
dependency=()
if [[ -n "${1:-}" ]]; then
  [[ "$1" =~ ^[0-9]+$ ]] || { echo 'Expected numeric data preparation job ID'; exit 1; }
  dependency=("--dependency=afterok:$1")
fi
active=$(squeue -h -u "$USER" -o '%j' | awk '$1 == "lei_short_train" || $1 == "lei_short_eval" {print}')
if [[ -n "$active" ]]; then echo "Short gate already queued/running: $active"; exit 1; fi
train_job=$(sbatch --parsable "${dependency[@]}" lei_short_train.sbatch)
train_job=${train_job%%;*}
echo "A/B seed-0 training array: $train_job (20,000 steps each)"
if eval_job=$(sbatch --parsable --dependency="afterok:$train_job" lei_short_eval.sbatch); then
  echo "Evaluation job: ${eval_job%%;*}"
else
  echo "Training remains queued. Retry only: sbatch --dependency=afterok:$train_job lei_short_eval.sbatch"
  exit 1
fi
echo 'Outputs: results/lei/short_gate_seed0/report.md; this is a single-seed pilot.'
