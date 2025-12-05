#!/bin/bash
# Submit all SLURM jobs (orchestrator, trainer, sampler) with shared run_id
#
# Usage:
#   # Use default settings (partition=gpu4, GPU=a6000)
#   bash scripts/slurm_submit_all.sh
#
#   # Override partition and GPU settings via environment variables:
#   SLURM_PARTITION=gpu3 \
#   SLURM_GPU_COUNT_TRAINER=4 \
#   SLURM_GPU_COUNT_SAMPLER=4 \
#   bash scripts/slurm_submit_all.sh
#
# Environment variables:
#   SLURM_PARTITION: SLURM partition name (default: gpu4)
#   SLURM_GPU_COUNT_TRAINER: Number of GPUs for trainer (default: 4)
#   SLURM_GPU_COUNT_SAMPLER: Number of GPUs for sampler (default: 4)
#   Note: Orchestrator runs on CPU only, no GPU required
#   Note: GPU type specification in --gres is not supported on all clusters,
#         so we use --gres=gpu:COUNT format instead of --gres=gpu:TYPE:COUNT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
CONFIG_FILE="${CONFIG_FILE:-my_exp.yaml}"
LOG_DIR="${LOG_DIR:-}"
RUN_ID="${RUN_ID:-}"
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-}"
ORCH_PORT="${ORCH_PORT:-59888}"
ORCH_HOST="${ORCH_HOST:-0.0.0.0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# SLURM resource configuration (can be overridden via environment variables)
SLURM_PARTITION="${SLURM_PARTITION:-gpu4}"
# Note: Orchestrator runs on CPU only, no GPU required
# Note: GPU type is not specified in --gres to ensure compatibility across clusters
SLURM_GPU_COUNT_TRAINER="${SLURM_GPU_COUNT_TRAINER:-4}"
SLURM_GPU_COUNT_SAMPLER="${SLURM_GPU_COUNT_SAMPLER:-4}"

# Generate run_id if not provided
if [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +"%Y%m%d_%H%M%S")_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
fi

# Determine log directory (use absolute path to ensure it's saved in project directory)
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$PROJECT_DIR/logs/$RUN_ID"
elif [[ "$LOG_DIR" != /* ]]; then
    # If relative path, make it absolute relative to project directory
    LOG_DIR="$PROJECT_DIR/$LOG_DIR"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/slurm_logs" || {
    echo "Error: Failed to create slurm_logs directory at $PROJECT_DIR/slurm_logs" >&2
    echo "Please ensure you have write permissions to the project directory." >&2
    exit 1
}

# Save run_id
echo "$RUN_ID" > "$LOG_DIR/run_id.txt"

# Determine orchestrator URL if not provided
if [ -z "$ORCHESTRATOR_URL" ]; then
    # Try to get from config file
    if [ -f "$CONFIG_FILE" ]; then
        ORCH_PORT_FROM_CONFIG=$(grep -E "^orchestrator_port:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' || echo "")
        if [ -n "$ORCH_PORT_FROM_CONFIG" ]; then
            ORCH_PORT="$ORCH_PORT_FROM_CONFIG"
        fi
    fi
    # Default: use localhost with port
    ORCHESTRATOR_URL="http://localhost:$ORCH_PORT"
fi

echo "========================================="
echo "Submitting SLURM jobs with run_id: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Config: $CONFIG_FILE"
echo "Orchestrator: $ORCHESTRATOR_URL"
echo ""
echo "SLURM Resource Configuration:"
echo "  Partition: $SLURM_PARTITION"
echo "  Orchestrator: CPU only (no GPU required)"
echo "  Trainer GPUs: $SLURM_GPU_COUNT_TRAINER"
echo "  Sampler GPUs: $SLURM_GPU_COUNT_SAMPLER"
echo "========================================="

# Submit orchestrator job (CPU only, no GPU required)
echo ""
echo "Submitting orchestrator job (CPU only)..."
ORCH_JOB_ID=$(sbatch \
    --partition="$SLURM_PARTITION" \
    --output="$PROJECT_DIR/slurm_logs/orch.%N.%j.out" \
    --error="$PROJECT_DIR/slurm_logs/orch.%N.%j.err" \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE",LOG_DIR="$LOG_DIR",RUN_ID="$RUN_ID",ORCH_PORT="$ORCH_PORT",ORCH_HOST="$ORCH_HOST" \
    scripts/slurm_orch.sh | awk '{print $4}')
echo "Orchestrator job ID: $ORCH_JOB_ID"
echo "$ORCH_JOB_ID" > "$LOG_DIR/orchestrator_slurm_jobid.txt"

# Wait a bit for orchestrator to start
echo "Waiting 10 seconds for orchestrator to start..."
sleep 10

# Submit trainer job
echo ""
echo "Submitting trainer job..."
TRAINER_JOB_ID=$(sbatch \
    --partition="$SLURM_PARTITION" \
    --gres="gpu:$SLURM_GPU_COUNT_TRAINER" \
    --output="$PROJECT_DIR/slurm_logs/trainer.%N.%j.out" \
    --error="$PROJECT_DIR/slurm_logs/trainer.%N.%j.err" \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE",LOG_DIR="$LOG_DIR",RUN_ID="$RUN_ID",ORCHESTRATOR_URL="$ORCHESTRATOR_URL",NPROC_PER_NODE="$NPROC_PER_NODE",MASTER_PORT="$MASTER_PORT" \
    scripts/slurm_trainer.sh | awk '{print $4}')
echo "Trainer job ID: $TRAINER_JOB_ID"
echo "$TRAINER_JOB_ID" > "$LOG_DIR/trainer_slurm_jobid.txt"

# Submit sampler job
echo ""
echo "Submitting sampler job..."
SAMPLER_JOB_ID=$(sbatch \
    --partition="$SLURM_PARTITION" \
    --gres="gpu:$SLURM_GPU_COUNT_SAMPLER" \
    --output="$PROJECT_DIR/slurm_logs/sampler.%N.%j.out" \
    --error="$PROJECT_DIR/slurm_logs/sampler.%N.%j.err" \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE",LOG_DIR="$LOG_DIR",RUN_ID="$RUN_ID",ORCHESTRATOR_URL="$ORCHESTRATOR_URL" \
    scripts/slurm_sampler.sh | awk '{print $4}')
echo "Sampler job ID: $SAMPLER_JOB_ID"
echo "$SAMPLER_JOB_ID" > "$LOG_DIR/sampler_slurm_jobid.txt"

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "Job IDs:"
echo "  Orchestrator: $ORCH_JOB_ID"
echo "  Trainer: $TRAINER_JOB_ID"
echo "  Sampler: $SAMPLER_JOB_ID"
echo ""
echo "To stop all jobs:"
echo "  bash scripts/slurm_stop.sh --log-dir $LOG_DIR"
echo ""
echo "To check job status:"
echo "  squeue -j $ORCH_JOB_ID,$TRAINER_JOB_ID,$SAMPLER_JOB_ID"
echo "========================================="

