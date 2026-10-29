#!/bin/bash -l
#SBATCH --job-name=lei_full_summary
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --output=lei_full_summary_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load 2025
module load python/3.11.9
python -m analysis.summarize_full_eval
