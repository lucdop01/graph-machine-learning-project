#!/bin/bash
#SBATCH --job-name=train_model_d
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -o pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment & Logging Initialization
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/smiyyapuram/graph-machine-learning-project"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-train_model_d}"
MASTER_LOG="logs/${JOB_NAME}_${JOB_ID}.log"

# Duplicate all stdout and stderr into a unified timestamped master log file
exec > >(tee -a "$MASTER_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Exit & Failure Trap Handler
# ─────────────────────────────────────────────────────────────────────────────
cleanup_and_report() {
    EXIT_CODE=$?
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    log "================================================================="
    log "JOB SUMMARY"
    log "Job Name:    $JOB_NAME"
    log "Job ID:      $JOB_ID"
    log "Node:        $(hostname)"
    log "End Time:    $END_TIME"
    log "Exit Code:   $EXIT_CODE"
    log "Master Log:  $MASTER_LOG"
    log "Slurm Out:   logs/${JOB_NAME}_${JOB_ID}.out"
    log "Slurm Err:   logs/${JOB_NAME}_${JOB_ID}.err"

    if [ $EXIT_CODE -eq 0 ]; then
        log "STATUS: SUCCESS - Model training finished normally."
    else
        log "STATUS: FAILED (Exit Code: $EXIT_CODE)"
        log "-------------------------------------------------------------"
        log "Troubleshooting & Debug Information:"
        log "GPU State at failure:"
        nvidia-smi || true
        log "Memory usage at failure:"
        free -h || true
        log "Saved checkpoints:"
        ls -la checkpoints/ 2>/dev/null || true
        log "-------------------------------------------------------------"
        log "Check the last lines of $MASTER_LOG for the exact Python traceback."
    fi
    log "================================================================="
}
trap cleanup_and_report EXIT

# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment & Module Setup
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log "STARTING JOB: $JOB_NAME (ID: $JOB_ID)"
log "Node Hostname: $(hostname)"
log "Working Dir:   $PROJECT_DIR"
log "Master Log:    $MASTER_LOG"
log "================================================================="

log "Loading cluster modules (2025, miniconda3, gcc/12.4.0, cuda/12.5)..."
module purge
module load 2025
module load miniconda3
module load gcc/12.4.0
module load cuda/12.5

ENV_DIR="$PROJECT_DIR/.env"
if [ ! -d "$ENV_DIR" ]; then
    log "ERROR: Local environment $ENV_DIR does not exist!"
    log "Please install it with: conda env create --prefix ./.env -f environment.yml"
    exit 1
fi

log "Activating local environment from $ENV_DIR..."
source "$ENV_DIR/bin/activate" || conda activate "$ENV_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-Flight Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
log "--- System & GPU Diagnostics ---"
log "Python: $(which python) ($(python --version 2>&1))"

python - << 'EOF'
import sys, torch
print(f"  PyTorch Version:       {torch.__version__}")
print(f"  CUDA Available:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Device Count:     {torch.cuda.device_count()}")
    print(f"  CUDA Device 0:         {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Device Capability:{torch.cuda.get_device_capability(0)}")
    print(f"  Allocated Memory:      {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
else:
    print("  WARNING: CUDA is NOT available to PyTorch!")
    sys.exit(1)
EOF

log "--- NVIDIA-SMI Snapshot ---"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu --format=csv,noheader || true

log "--- Code Version ---"
log "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
log "Git Branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
DIRTY=$(git status -s 2>/dev/null)
if [ -n "$DIRTY" ]; then
    log "Git Status: Uncommitted changes detected:"
    git status -s | while IFS= read -r line; do log "  $line"; done
else
    log "Git Status: Clean working tree."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Data Readiness Verification
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "data/graphs/norm_stride5.json" ] || [ ! -f "data/graphs/size_bins.json" ]; then
    log "WARNING: Preprocessed dataset (norm_stride5.json / size_bins.json) not found in data/graphs/!"
    log "Beginning on-demand data preparation pipeline..."

    if [ ! -d "data/runs" ]; then
        log "[Data Prep 1/4] Generating simulation runs (python -m data.generate)..."
        python -m data.generate
    else
        log "[Data Prep 1/4] data/runs/ already exists, skipping generation."
    fi

    log "[Data Prep 2/4] Preprocessing graphs (python -m graphs.preprocess)..."
    python -m graphs.preprocess
    python -m graphs.preprocess --splits test_interp --mesh-seed 1

    log "[Data Prep 3/4] Computing normalization statistics (python -m graphs.dataset --stride 5)..."
    python -m graphs.dataset --stride 5

    log "[Data Prep 4/4] Computing melt pool size bins (python -m evaluation.rollout --size-bins)..."
    python -m evaluation.rollout --size-bins
    log "Data preparation completed successfully."
else
    log "Dataset check: data/graphs/norm_stride5.json and size_bins.json verified."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Model D Training Execution
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log "LAUNCHING MODEL D TRAINING"
log "Model Config: configs/model_d.yaml"
log "Device:       cuda (A100)"
log "Precision:    bf16"
log "CLI Extra:    $*"
log "================================================================="

TRAIN_START=$(date +%s)

python train_harness.py \
    --config configs/model_d.yaml \
    --device cuda \
    --precision bf16 \
    --name model_d_seed0 \
    "$@"

TRAIN_END=$(date +%s)
DURATION=$((TRAIN_END - TRAIN_START))
log "Model D training finished in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s."
