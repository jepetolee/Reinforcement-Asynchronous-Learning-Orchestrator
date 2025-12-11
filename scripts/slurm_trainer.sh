#!/bin/bash
# SLURM job script for RALO Trainer
# 
# To override partition and GPU settings, use environment variables when submitting:
#   export CONFIG_FILE=my_exp.yaml
#   export ORCHESTRATOR_URL=http://172.16.162.40:59888
#   export SLURM_PARTITION=gpu4
#   export SLURM_GPU_COUNT_TRAINER=4
#   export NPROC_PER_NODE=4
#   export MASTER_PORT=29501
#   sbatch \
#     --partition="$SLURM_PARTITION" \
#     --gres="gpu:$SLURM_GPU_COUNT_TRAINER" \
#     --export=ALL,CONFIG_FILE="$CONFIG_FILE",ORCHESTRATOR_URL="$ORCHESTRATOR_URL",NPROC_PER_NODE="$NPROC_PER_NODE",MASTER_PORT="$MASTER_PORT" \
#     scripts/slurm_trainer.sh
#   
#   Note: If your cluster doesn't support GPU type specification in --gres, use:
#   --gres="gpu:$SLURM_GPU_COUNT_TRAINER" (without type)
# 
# Or use slurm_submit_all.sh with environment variables:
#   SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_TRAINER=4 bash scripts/slurm_submit_all.sh

#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:4
#SBATCH --job-name=ralo_trainer
#SBATCH --output=./slurm_logs/trainer.%N.%j.out
#SBATCH --error=./slurm_logs/trainer.%N.%j.err
#SBATCH --ntasks-per-node=4

set -e

# Load modules (optional; ignore failures)
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.1.1 gnu9/9.4.0 || true
fi

# Resolve project root (avoid /var/spool/slurmd)
if [ -n "$PROJECT_DIR" ] && [ -d "$PROJECT_DIR" ]; then
  :
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -d "$PWD" ]; then
  PROJECT_DIR="$PWD"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_DIR"

# Create slurm_logs directory only if writable
set +e
if [ -w "$PROJECT_DIR" ]; then
  mkdir -p "$PROJECT_DIR/slurm_logs" 2>/dev/null || \
    echo "Warning: Failed to create slurm_logs directory at $PROJECT_DIR/slurm_logs" >&2
else
  echo "Warning: Project dir not writable ($PROJECT_DIR). Relying on absolute --output/--error." >&2
fi
set -e

# Parse arguments
CONFIG_FILE="${CONFIG_FILE:-my_exp.yaml}"
LOG_DIR="${LOG_DIR:-}"
RUN_ID="${RUN_ID:-}"
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

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

# Save SLURM job ID
echo "$SLURM_JOB_ID" > "$LOG_DIR/trainer_slurm_jobid.txt"
echo "$RUN_ID" > "$LOG_DIR/run_id.txt"

# Set up CUDA devices if not specified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Use all GPUs allocated by SLURM
    CUDA_VISIBLE_DEVICES=$(echo $SLURM_STEP_GPUS | tr ',' ' ')
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # Fallback: use sequential numbering
        CUDA_VISIBLE_DEVICES="0,1,2,3"
    fi
fi

# Export environment variables
export RUN_ID
export CUDA_VISIBLE_DEVICES
export ORCH_SERVER="$ORCHESTRATOR_URL"
export MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job information
echo "========================================="
echo "SLURM Trainer Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Node ID: $SLURM_NODEID"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Config: $CONFIG_FILE"
echo "Orchestrator: $ORCHESTRATOR_URL"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Processes per node: $NPROC_PER_NODE"
echo "========================================="

# Build command
# Use torchrun for distributed training
TORCHRUN_OPTS="--standalone --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT"
CMD="torchrun $TORCHRUN_OPTS ralo_cli.py train --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"

if [ -n "$ORCHESTRATOR_URL" ]; then
    CMD="$CMD --orchestrator $ORCHESTRATOR_URL"
fi

# Run trainer
echo "Starting trainer..."
echo "Command: $CMD"
echo ""

$CMD

