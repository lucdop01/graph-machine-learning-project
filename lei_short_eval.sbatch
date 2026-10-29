#!/bin/bash -l
#SBATCH --job-name=lei_short_eval
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=04:00:00
#SBATCH --output=lei_short_eval_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load 2025
module load python/3.11.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
srun python -u -m analysis.short_gate_seed0 --evaluate
