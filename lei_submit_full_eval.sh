#!/bin/bash
# Submit from the repository root. No model training is started.
set -euo pipefail
if [[ ! -d .venv ]]; then echo 'Missing .venv: use the DelftBlue repository directory'; exit 1; fi
if compgen -G 'results/lei/full_eval/*_s5.json' >/dev/null; then
  echo 'Existing full-evaluation outputs found. Archive them before starting another campaign.'
  exit 1
fi
for run in model_a_seed0 model_b_seed0 model_b_seed1 model_b_seed2; do
  test -f "results/models/$run/best.pt" || { echo "Missing checkpoint: $run"; exit 1; }
done
# Refuse duplicate preparation/evaluation jobs, including an older preparation.
active=$(squeue -h -u "$USER" -o '%j' | awk '$1 == "lei_full_prepare" || $1 == "lei_full_eval" || $1 == "lei_prepare_eval" {print}')
if [[ -n "$active" ]]; then
  echo "Preparation/evaluation already queued or running: $active"
  echo 'Wait for it to finish before submitting this pipeline.'
  exit 1
fi
prep=$(sbatch --parsable lei_full_prepare.sbatch)
prep=${prep%%;*}
echo "Data preparation job: $prep"
if evaluation=$(sbatch --parsable --dependency="afterok:$prep" lei_full_eval.sbatch); then
  echo "Evaluation array: ${evaluation%%;*} (starts only after preparation succeeds)"
  summary=$(sbatch --parsable --dependency="afterany:${evaluation%%;*}" lei_full_summary.sbatch)
  echo "Summary job: ${summary%%;*} (reports incomplete tasks even if evaluation fails)"
else
  echo "Evaluation submission failed. Preparation $prep remains queued; do not resubmit preparation."
  echo "Retry: sbatch --dependency=afterok:$prep lei_full_eval.sbatch"
  exit 1
fi
echo 'Outputs: results/lei/full_eval. This evaluates existing weights; formal gate still needs matched A seeds 1/2.'
