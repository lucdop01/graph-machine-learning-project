#!/bin/bash -l
#SBATCH --job-name=lei_gpu_probe
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=lei_gpu_probe_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module purge
module load 2025
module load python/3.11.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

hostname
nvidia-smi
srun python -m verify.gpu_probe --device cuda
