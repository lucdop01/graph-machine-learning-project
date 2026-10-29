#!/bin/bash -l
#SBATCH --job-name=lei_full_prepare
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=08:00:00
#SBATCH --output=lei_full_prepare_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load 2025
module load python/3.11.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
python -m data.generate --only train val test_interp test_fine
python -m graphs.preprocess --splits train val test_interp test_fine
python -m graphs.preprocess --splits test_interp --mesh-seed 1
python -m graphs.dataset --stride 5 --splits train val test_interp test_fine
python -m evaluation.rollout --size-bins
echo 'COMPLETE: full evaluation data ready'
