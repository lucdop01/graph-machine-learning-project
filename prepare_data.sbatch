#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=compute-p1
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -o pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment & Logging Initialization
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/smiyyapuram/graph-machine-learning-project"
cd "$PROJECT_DIR"
mkdir -p logs data/runs data/graphs

JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-prep_data}"
MASTER_LOG="logs/${JOB_NAME}_${JOB_ID}.log"

# Duplicate all stdout and stderr to a timestamped master log file
exec > >(tee -a "$MASTER_LOG") 2>&1

export PYTHONUNBUFFERED=1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Exit Trap Handler
# ─────────────────────────────────────────────────────────────────────────────
cleanup_and_report() {
    EXIT_CODE=$?
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    log "================================================================="
    log "DATA PREPARATION SUMMARY"
    log "Job Name:    $JOB_NAME"
    log "Job ID:      $JOB_ID"
    log "Node:        $(hostname)"
    log "End Time:    $END_TIME"
    log "Exit Code:   $EXIT_CODE"
    log "Master Log:  $MASTER_LOG"
    log "Slurm Out:   logs/${JOB_NAME}_${JOB_ID}.out"
    log "Slurm Err:   logs/${JOB_NAME}_${JOB_ID}.err"

    if [ $EXIT_CODE -eq 0 ]; then
        log "STATUS: SUCCESS - All simulation runs and graphs preprocessed."
        log "Files created:"
        ls -lh data/graphs/norm_stride5.json data/graphs/size_bins.json 2>/dev/null || true
    else
        log "STATUS: FAILED (Exit Code: $EXIT_CODE)"
        log "-------------------------------------------------------------"
        log "Troubleshooting:"
        log "Check the last lines of $MASTER_LOG for details."
        log "Disk space on scratch:"
        df -h "$PROJECT_DIR" || true
    fi
    log "================================================================="
}
trap cleanup_and_report EXIT

# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment & Module Setup
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log "STARTING DATA PREPARATION: $JOB_NAME (ID: $JOB_ID)"
log "Node Hostname: $(hostname)"
log "Master Log:    $MASTER_LOG"
log "================================================================="

log "Loading cluster modules (2025, miniconda3, gcc/12.4.0)..."
module purge
module load 2025
module load miniconda3
module load gcc/12.4.0

ENV_DIR="$PROJECT_DIR/.env"
if [ ! -d "$ENV_DIR" ]; then
    log "ERROR: Local environment $ENV_DIR does not exist!"
    log "Please install it with: conda env create --prefix ./.env -f environment.yml"
    exit 1
fi

log "Activating local environment from $ENV_DIR..."
source "$ENV_DIR/bin/activate" || conda activate "$ENV_DIR"

log "Python: $(which python) ($(python --version 2>&1))"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────
run_step() {
    local step_name="$1"
    shift
    local step_start
    step_start=$(date +%s)
    log ">>> [STEP START] $step_name: $*"
    "$@"
    local step_end
    step_end=$(date +%s)
    local step_dur=$((step_end - step_start))
    log "<<< [STEP DONE] $step_name (took ${step_dur}s)"
}

# Step 1: Generate simulation runs (~275 runs)
run_step "Generate Simulation Runs" python -m data.generate

# Step 2: Preprocess graphs onto meshes
run_step "Preprocess Standard Graphs" python -m graphs.preprocess
run_step "Preprocess New-Mesh Test Graphs" python -m graphs.preprocess --splits test_interp --mesh-seed 1

# Step 3: Compute normalization stats
run_step "Compute Normalization Statistics (stride 5)" python -m graphs.dataset --stride 5

# Step 4: Melt pool size bins for rollout scoring
run_step "Compute Melt Pool Size Bins" python -m evaluation.rollout --size-bins

log "All data preparation steps finished successfully."
