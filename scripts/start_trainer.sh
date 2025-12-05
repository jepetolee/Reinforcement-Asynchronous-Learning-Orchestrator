#!/bin/bash
# Start trainer in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Save environment variable RUN_ID before argument parsing
ENV_RUN_ID="${RUN_ID:-}"

# Parse arguments
CONFIG_FILE=""
LOG_DIR=""
RUN_ID=""
ORCHESTRATOR_URL=""
CUDA_VISIBLE_DEVICES=""
NPROC_PER_NODE=""
MASTER_PORT=""
SKIP_TRAINER_REGISTRATION=""
TORCH_NCCL_ASYNC_ERROR_HANDLING=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --orchestrator)
            ORCHESTRATOR_URL="$2"
            shift 2
            ;;
        --cuda-visible-devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --skip-trainer-registration)
            SKIP_TRAINER_REGISTRATION="$2"
            shift 2
            ;;
        --torch-nccl-async-error-handling)
            TORCH_NCCL_ASYNC_ERROR_HANDLING="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: --config is required"
    exit 1
fi

# Use RUN_ID from environment variable if not provided via argument
if [ -z "$RUN_ID" ] && [ -n "$ENV_RUN_ID" ]; then
    RUN_ID="$ENV_RUN_ID"
elif [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +"%Y%m%d_%H%M%S")_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
fi

# Determine log directory
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/$RUN_ID"
fi

mkdir -p "$LOG_DIR"

# Set up environment variables (use argument or inherit from environment)
if [ -n "$ORCHESTRATOR_URL" ]; then
    export ORCH_SERVER="$ORCHESTRATOR_URL"
elif [ -z "${ORCH_SERVER:-}" ]; then
    # ORCH_SERVER not set, will use default
    :
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # CUDA_VISIBLE_DEVICES not set, will use all available
    :
fi

if [ -n "$SKIP_TRAINER_REGISTRATION" ]; then
    export SKIP_TRAINER_REGISTRATION="$SKIP_TRAINER_REGISTRATION"
elif [ -z "${SKIP_TRAINER_REGISTRATION:-}" ]; then
    # SKIP_TRAINER_REGISTRATION not set
    :
fi

if [ -n "$TORCH_NCCL_ASYNC_ERROR_HANDLING" ]; then
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING"
elif [ -z "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}" ]; then
    # TORCH_NCCL_ASYNC_ERROR_HANDLING not set
    :
fi

# Build command
# Use torchrun if nproc_per_node is specified, otherwise use python directly
if [ -n "$NPROC_PER_NODE" ]; then
    # Use torchrun for distributed training
    TORCHRUN_OPTS="--standalone --nproc_per_node=$NPROC_PER_NODE"
    if [ -n "$MASTER_PORT" ]; then
        TORCHRUN_OPTS="$TORCHRUN_OPTS --master_port=$MASTER_PORT"
    else
        TORCHRUN_OPTS="$TORCHRUN_OPTS --master_port=29501"
    fi
    CMD="torchrun $TORCHRUN_OPTS ralo_cli.py train --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
else
    # Use python directly for single process
    CMD="python ralo_cli.py train --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
fi

if [ -n "$ORCHESTRATOR_URL" ]; then
    CMD="$CMD --orchestrator $ORCHESTRATOR_URL"
fi

# Export RUN_ID for child processes
export RUN_ID

# Start in background
echo "Starting trainer with run_id: $RUN_ID"
echo "Log directory: $LOG_DIR"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
if [ -n "$NPROC_PER_NODE" ]; then
    echo "Using torchrun with nproc_per_node=$NPROC_PER_NODE"
fi
nohup bash -c "$CMD" > "$LOG_DIR/trainer_nohup.out" 2>&1 &
PID=$!

# Save PID
echo $PID > "$LOG_DIR/trainer.pid"
echo "Trainer started with PID: $PID"
echo "Log file: $LOG_DIR/trainer.log"
echo "Run ID: $RUN_ID"

