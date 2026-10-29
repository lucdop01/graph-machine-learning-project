#!/bin/bash -l
#SBATCH --job-name=lei_short_train
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=lei_short_train_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load 2025
module load python/3.11.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
models=(A B)
srun python -u -m analysis.short_gate_seed0 --train-model "${models[$SLURM_ARRAY_TASK_ID]}"
