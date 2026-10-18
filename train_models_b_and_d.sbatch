#!/bin/bash
#SBATCH --job-name=train_models_b_d
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
JOB_NAME="${SLURM_JOB_NAME:-train_models_b_d}"
MASTER_LOG="logs/${JOB_NAME}_${JOB_ID}.log"

# Unified master log
exec > >(tee -a "$MASTER_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "================================================================="
log "STARTING SEQUENTIAL MULTI-SEED PIPELINE (SINGLE A100 GPU)"
log "Job ID:        $JOB_ID"
log "Host Node:     $(hostname)"
log "Allocation:    1 A100 GPU (80GB), 12 CPUs, 48 GB RAM"
log "Plan:          Phase 1: Model D (Seeds 1, 2 in parallel on GPU)"
log "               Phase 2: Model B (Seeds 0, 1, 2 in parallel on GPU)"
log "Master Log:    $MASTER_LOG"
log "================================================================="

# Load cluster modules
module purge
module load 2025 miniconda3 gcc/12.4.0 cuda/12.5

ENV_DIR="$PROJECT_DIR/.env"
source "$ENV_DIR/bin/activate" || conda activate "$ENV_DIR"

log "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: MODEL D (Strict Heat Conservation)
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log ">>> PHASE 1: MODEL D MULTI-SEED TRAINING"
log "================================================================="

# Seed 0 is already completed (checkpoints/model_d_seed0/best.pt)
D_SEEDS_TO_RUN=()
for s in 0 1 2; do
    CKPT="checkpoints/model_d_seed${s}/best.pt"
    if [ -f "$CKPT" ]; then
        log "[Model D] Seed $s: Checkpoint $CKPT already completed (skipping re-training)."
    else
        log "[Model D] Seed $s: Needs training."
        D_SEEDS_TO_RUN+=("$s")
    fi
done

if [ ${#D_SEEDS_TO_RUN[@]} -gt 0 ]; then
    log "[Model D] Launching seeds ${D_SEEDS_TO_RUN[*]} concurrently on the A100 GPU..."
    D_PIDS=()
    for s in "${D_SEEDS_TO_RUN[@]}"; do
        SEED_LOG="logs/model_d_seed${s}_${JOB_ID}.log"
        CKPT_DIR="checkpoints/model_d_seed${s}"
        mkdir -p "$CKPT_DIR"
        log "[Model D] Starting Seed $s -> Checkpoints in $CKPT_DIR/ | Log in $SEED_LOG"
        python train_harness.py \
            --config configs/model_d.yaml \
            --device cuda \
            --precision bf16 \
            --seed "$s" \
            --name "model_d_seed${s}" \
            > "$SEED_LOG" 2>&1 &
        D_PIDS+=("$!")
    done

    log "[Model D] Processes running (PIDs: ${D_PIDS[*]}). Waiting for Model D completion..."
    D_FAIL=0
    for idx in "${!D_PIDS[@]}"; do
        p="${D_PIDS[$idx]}"
        s="${D_SEEDS_TO_RUN[$idx]}"
        if wait "$p"; then
            log "[Model D] Seed $s (PID $p) finished successfully."
        else
            log "[Model D] Seed $s (PID $p) FAILED with exit code $?."
            D_FAIL=1
        fi
    done

    if [ $D_FAIL -ne 0 ]; then
        log "ERROR: One or more Model D runs failed. Inspect individual seed logs."
        exit 1
    fi
fi
log "[Model D] Phase 1 completed successfully."

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: MODEL B (Physics-Guided Linear Diffusion Residual)
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log ">>> PHASE 2: MODEL B MULTI-SEED TRAINING"
log "================================================================="

B_SEEDS_TO_RUN=()
for s in 0 1 2; do
    CKPT="checkpoints/model_b_seed${s}/best.pt"
    if [ -f "$CKPT" ]; then
        log "[Model B] Seed $s: Checkpoint $CKPT already exists (skipping)."
    else
        log "[Model B] Seed $s: Needs training."
        B_SEEDS_TO_RUN+=("$s")
    fi
done

if [ ${#B_SEEDS_TO_RUN[@]} -gt 0 ]; then
    log "[Model B] Launching seeds ${B_SEEDS_TO_RUN[*]} concurrently on the A100 GPU..."
    B_PIDS=()
    for s in "${B_SEEDS_TO_RUN[@]}"; do
        SEED_LOG="logs/model_b_seed${s}_${JOB_ID}.log"
        CKPT_DIR="checkpoints/model_b_seed${s}"
        mkdir -p "$CKPT_DIR"
        log "[Model B] Starting Seed $s -> Checkpoints in $CKPT_DIR/ | Log in $SEED_LOG"
        python train_harness.py \
            --config configs/model_b.yaml \
            --device cuda \
            --precision bf16 \
            --seed "$s" \
            --name "model_b_seed${s}" \
            > "$SEED_LOG" 2>&1 &
        B_PIDS+=("$!")
    done

    log "[Model B] Processes running (PIDs: ${B_PIDS[*]}). Waiting for Model B completion..."
    B_FAIL=0
    for idx in "${!B_PIDS[@]}"; do
        p="${B_PIDS[$idx]}"
        s="${B_SEEDS_TO_RUN[$idx]}"
        if wait "$p"; then
            log "[Model B] Seed $s (PID $p) finished successfully."
        else
            log "[Model B] Seed $s (PID $p) FAILED with exit code $?."
            B_FAIL=1
        fi
    done

    if [ $B_FAIL -ne 0 ]; then
        log "ERROR: One or more Model B runs failed. Inspect individual seed logs."
        exit 1
    fi
fi
log "[Model B] Phase 2 completed successfully."

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT: MULTI-SEED BENCHMARKS ACROSS ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
log "================================================================="
log "FINAL MULTI-SEED SCIENTIFIC BENCHMARK SUMMARY"
log "================================================================="

python - << 'EOF'
import os, torch, numpy as np

def summarize_model(model_name, prefix):
    print(f"\n====================== {model_name.upper()} RESULTS ======================")
    print(f"{'Seed':<8} | {'Best Step':<10} | {'Selection Score (nRMSE)':<25} | {'Checkpoint'}")
    print("-" * 75)
    scores = []
    for s in [0, 1, 2]:
        path = f"checkpoints/{prefix}_seed{s}/best.pt"
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
        print(f"3-SEED SCORE FOR {model_name}: {mean_val:.5g} ± {std_val:.5g} nRMSE")
    print("=" * 75)

summarize_model("Model D (Exact Heat Conservation)", "model_d")
summarize_model("Model B (Physics-Guided Diffusion Residual)", "model_b")
EOF

log "Job completed at $(date). All logs and checkpoints are saved per seed."

