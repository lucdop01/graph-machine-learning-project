#!/bin/bash -l
#SBATCH --job-name=lei_quick_eval
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:30:00
#SBATCH --output=lei_quick_eval_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module purge
module load 2025
module load python/3.11.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
date
hostname
srun python -u -m analysis.quick_checkpoint_eval "$@"
