#!/bin/bash
#SBATCH --job-name=model_b_all_seeds
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -o pipefail

PROJECT_DIR="/scratch/smiyyapuram/graph-machine-learning-project"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-model_b_all_seeds}"
MASTER_LOG="logs/${JOB_NAME}_${JOB_ID}.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "================================================================="
log "STARTING MULTI-SEED RUN: MODEL B (Job ID: $JOB_ID)"
log "Node Hostname: $(hostname)"
log "Allocated Resources: 1 A100 GPU, 12 CPUs, 48 GB RAM"
log "================================================================="

# Load cluster modules
module purge
module load 2025 miniconda3 gcc/12.4.0 cuda/12.5

ENV_DIR="$PROJECT_DIR/.env"
source "$ENV_DIR/bin/activate" || conda activate "$ENV_DIR"

# Diagnostics
log "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Determine which seeds need running
SEEDS_TO_RUN=()
for s in 0 1 2; do
    CKPT="checkpoints/model_b_seed${s}/best.pt"
    if [ -f "$CKPT" ]; then
        log "Seed $s: Checkpoint $CKPT already exists (skipping re-training)."
    else
        log "Seed $s: Needs training."
        SEEDS_TO_RUN+=("$s")
    fi
done

if [ ${#SEEDS_TO_RUN[@]} -eq 0 ]; then
    log "All 3 seeds (0, 1, 2) already have completed checkpoints!"
else
    log "Launching seeds in parallel on the same A100 GPU: ${SEEDS_TO_RUN[*]}"
    PIDS=()
    for s in "${SEEDS_TO_RUN[@]}"; do
        SEED_LOG="logs/model_b_seed${s}_${JOB_ID}.log"
        CKPT_DIR="checkpoints/model_b_seed${s}"
        mkdir -p "$CKPT_DIR"
        log "Starting Model B Seed $s (Checkpoints saved to: $CKPT_DIR/ | Log: $SEED_LOG)..."
        python train_harness.py \
            --config configs/model_b.yaml \
            --device cuda \
            --precision bf16 \
            --seed "$s" \
            --name "model_b_seed${s}" \
            > "$SEED_LOG" 2>&1 &
        PIDS+=("$!")
    done

    # Monitor parallel training
    log "All seed processes spawned (PIDs: ${PIDS[*]}). Waiting for completion..."
    FAIL=0
    for idx in "${!PIDS[@]}"; do
        p="${PIDS[$idx]}"
        s="${SEEDS_TO_RUN[$idx]}"
        if wait "$p"; then
            log "Seed $s (PID $p) finished successfully."
        else
            log "Seed $s (PID $p) FAILED with exit code $?."
            FAIL=1
        fi
    done

    if [ $FAIL -ne 0 ]; then
        log "ERROR: One or more seed runs failed. Check individual seed logs."
        exit 1
    fi
fi

# Multi-seed aggregate evaluation
log "================================================================="
log "AGGREGATING MULTI-SEED BENCHMARKS FOR MODEL B (SEEDS 0, 1, 2)"
log "================================================================="

python - << 'EOF'
import os, torch, numpy as np

seeds = [0, 1, 2]
scores = []
print(f"{'Seed':<8} | {'Best Step':<10} | {'Selection Score (nRMSE)':<25} | {'Checkpoint'}")
print("-" * 75)

for s in seeds:
    path = f"checkpoints/model_b_seed{s}/best.pt"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        score = float(ckpt.get("best_score", ckpt.get("score", float("nan"))))
        step = ckpt.get("best_step", ckpt.get("step", "N/A"))
        scores.append(score)
        print(f"Seed {s:<3} | Step {step:<5} | {score:<25.6g} | {path}")
    else:
        print(f"Seed {s:<3} | MISSING    | {'N/A':<25} | {path}")

if scores:
    mean_val = float(np.mean(scores))
    std_val = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    print("-" * 75)
    print(f"FINAL 3-SEED RESULT FOR MODEL B: {mean_val:.5g} ± {std_val:.5g} nRMSE")
print("=================================================================")
EOF

log "Multi-seed Model B execution finished successfully."
