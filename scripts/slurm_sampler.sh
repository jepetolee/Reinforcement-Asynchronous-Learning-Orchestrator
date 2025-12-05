#!/bin/bash
# SLURM job script for RALO Sampler
# 
# To override partition and GPU settings, use environment variables when submitting:
#   export CONFIG_FILE=my_exp.yaml
#   export ORCHESTRATOR_URL=http://172.16.162.40:59888
#   export GEN_DEVICES=0,1,2,3
#   export SLURM_PARTITION=gpu4
#   export SLURM_GPU_COUNT_SAMPLER=4
#   sbatch \
#     --partition="$SLURM_PARTITION" \
#     --gres="gpu:$SLURM_GPU_COUNT_SAMPLER" \
#     --export=ALL,CONFIG_FILE="$CONFIG_FILE",ORCHESTRATOR_URL="$ORCHESTRATOR_URL",GEN_DEVICES="$GEN_DEVICES" \
#     scripts/slurm_sampler.sh
#   
#   Note: If your cluster doesn't support GPU type specification in --gres, use:
#   --gres="gpu:$SLURM_GPU_COUNT_SAMPLER" (without type)
# 
# Or use slurm_submit_all.sh with environment variables:
#   SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_SAMPLER=4 bash scripts/slurm_submit_all.sh

#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:4
#SBATCH --job-name=ralo_sampler
#SBATCH --output=./slurm_logs/sampler.%N.%j.out
#SBATCH --error=./slurm_logs/sampler.%N.%j.err

set -e

# Load modules
module purge
module load cuda/12.1.1 gnu9/9.4.0

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Create slurm_logs directory if it doesn't exist
# Note: If using slurm_submit_all.sh, the directory is created before job submission
# This is a fallback for direct script execution
# Temporarily disable set -e to allow mkdir to fail gracefully
set +e
mkdir -p "$PROJECT_DIR/slurm_logs" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Failed to create slurm_logs directory at $PROJECT_DIR/slurm_logs" >&2
    echo "Warning: SLURM output files may be written elsewhere or job may fail." >&2
fi
set -e

# Parse arguments
CONFIG_FILE="${CONFIG_FILE:-my_exp.yaml}"
LOG_DIR="${LOG_DIR:-}"
RUN_ID="${RUN_ID:-}"
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-}"
GEN_DEVICES="${GEN_DEVICES:-}"

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
echo "$SLURM_JOB_ID" > "$LOG_DIR/sampler_slurm_jobid.txt"
echo "$RUN_ID" > "$LOG_DIR/run_id.txt"

# Set up GPU devices if not specified
if [ -z "$GEN_DEVICES" ]; then
    # Use all GPUs allocated by SLURM
    if [ -n "$SLURM_STEP_GPUS" ]; then
        GEN_DEVICES=$(echo $SLURM_STEP_GPUS | tr ',' ' ')
    else
        # Fallback: use sequential numbering (0,1,2,3)
        GEN_DEVICES="0,1,2,3"
    fi
fi

# Export environment variables
export RUN_ID
export GEN_DEVICES
export ORCH_SERVER="$ORCHESTRATOR_URL"

# Print job information
echo "========================================="
echo "SLURM Sampler Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Node ID: $SLURM_NODEID"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Config: $CONFIG_FILE"
echo "Orchestrator: $ORCHESTRATOR_URL"
echo "GPU Devices: $GEN_DEVICES"
echo "========================================="

# Build command
CMD="python ralo_cli.py gen --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"

if [ -n "$ORCHESTRATOR_URL" ]; then
    CMD="$CMD --orchestrator $ORCHESTRATOR_URL"
fi

# Run sampler
echo "Starting sampler..."
echo "Command: $CMD"
echo ""

$CMD

