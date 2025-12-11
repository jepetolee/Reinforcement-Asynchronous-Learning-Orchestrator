#!/bin/bash
# SLURM job script for RALO Orchestrator
# 
# Note: Orchestrator runs on CPU only, no GPU required
# 
# To override partition, use environment variables when submitting:
#   SLURM_PARTITION=cpu \
#   sbatch --partition=$SLURM_PARTITION scripts/slurm_orch.sh
# 
# Or use slurm_submit_all.sh with environment variables:
#   SLURM_PARTITION=cpu bash scripts/slurm_submit_all.sh

#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# Note: Orchestrator runs on CPU only, no GPU required
#SBATCH --job-name=ralo_orch
#SBATCH --output=./slurm_logs/orch.%N.%j.out
#SBATCH --error=./slurm_logs/orch.%N.%j.err

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
ORCH_PORT="${ORCH_PORT:-59888}"
ORCH_HOST="${ORCH_HOST:-0.0.0.0}"

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
echo "$SLURM_JOB_ID" > "$LOG_DIR/orchestrator_slurm_jobid.txt"
echo "$RUN_ID" > "$LOG_DIR/run_id.txt"

# Export environment variables
export RUN_ID
export ORCH_HOST
export ORCH_PORT

# Print job information
echo "========================================="
echo "SLURM Orchestrator Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Node ID: $SLURM_NODEID"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Config: $CONFIG_FILE"
echo "Orchestrator: $ORCH_HOST:$ORCH_PORT"
echo "========================================="

# Build command
CMD="python ralo_cli.py orch --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR --orch-port $ORCH_PORT"

# Run orchestrator
echo "Starting orchestrator..."
echo "Command: $CMD"
echo ""

$CMD

