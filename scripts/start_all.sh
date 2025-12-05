#!/bin/bash
# Start all processes (orchestrator, trainer, sampler) in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
CONFIG_FILE=""
LOG_DIR=""
RUN_ID=""
ORCHESTRATOR_URL=""
ORCH_PORT=""
ORCH_HOST=""
GEN_DEVICES=""
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
        --orch-port)
            ORCH_PORT="$2"
            shift 2
            ;;
        --orch-host)
            ORCH_HOST="$2"
            shift 2
            ;;
        --gen-devices)
            GEN_DEVICES="$2"
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

# Generate run_id if not provided
if [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +"%Y%m%d_%H%M%S")_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
fi

# Determine log directory
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/$RUN_ID"
fi

mkdir -p "$LOG_DIR"

# Save run_id
echo "$RUN_ID" > "$LOG_DIR/run_id.txt"

# Export RUN_ID for child processes
export RUN_ID

# Set up common environment variables
if [ -n "$ORCHESTRATOR_URL" ]; then
    export ORCH_SERVER="$ORCHESTRATOR_URL"
fi

echo "========================================="
echo "Starting all processes with run_id: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "========================================="

# Start orchestrator
echo ""
echo "Starting orchestrator..."

# Set up orchestrator environment variables
if [ -n "$ORCH_HOST" ]; then
    export ORCH_HOST="$ORCH_HOST"
else
    export ORCH_HOST="${ORCH_HOST:-0.0.0.0}"
fi
if [ -n "$ORCH_PORT" ]; then
    export ORCH_PORT="$ORCH_PORT"
fi

ORCH_CMD="python ralo_cli.py orch --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
if [ -n "$ORCH_PORT" ]; then
    ORCH_CMD="$ORCH_CMD --orch-port $ORCH_PORT"
fi
nohup bash -c "$ORCH_CMD" > "$LOG_DIR/orchestrator_nohup.out" 2>&1 &
ORCH_PID=$!
echo $ORCH_PID > "$LOG_DIR/orchestrator.pid"
echo "Orchestrator started with PID: $ORCH_PID"

# Wait a bit for orchestrator to start
sleep 3

# Start trainer
echo ""
echo "Starting trainer..."

# Set up trainer environment variables
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
if [ -n "$SKIP_TRAINER_REGISTRATION" ]; then
    export SKIP_TRAINER_REGISTRATION="$SKIP_TRAINER_REGISTRATION"
fi
if [ -n "$TORCH_NCCL_ASYNC_ERROR_HANDLING" ]; then
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING"
fi

# Build trainer command
if [ -n "$NPROC_PER_NODE" ]; then
    # Use torchrun for distributed training
    TORCHRUN_OPTS="--standalone --nproc_per_node=$NPROC_PER_NODE"
    if [ -n "$MASTER_PORT" ]; then
        TORCHRUN_OPTS="$TORCHRUN_OPTS --master_port=$MASTER_PORT"
    else
        TORCHRUN_OPTS="$TORCHRUN_OPTS --master_port=29501"
    fi
    TRAINER_CMD="torchrun $TORCHRUN_OPTS ralo_cli.py train --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
    echo "Using torchrun with nproc_per_node=$NPROC_PER_NODE"
else
    # Use python directly for single process
    TRAINER_CMD="python ralo_cli.py train --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
fi
if [ -n "$ORCHESTRATOR_URL" ]; then
    TRAINER_CMD="$TRAINER_CMD --orchestrator $ORCHESTRATOR_URL"
fi

nohup bash -c "$TRAINER_CMD" > "$LOG_DIR/trainer_nohup.out" 2>&1 &
TRAINER_PID=$!
echo $TRAINER_PID > "$LOG_DIR/trainer.pid"
echo "Trainer started with PID: $TRAINER_PID"

# Start sampler
echo ""
echo "Starting sampler..."

# Set up sampler environment variables
if [ -n "$GEN_DEVICES" ]; then
    export GEN_DEVICES="$GEN_DEVICES"
fi

SAMPLER_CMD="python ralo_cli.py gen --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
if [ -n "$ORCHESTRATOR_URL" ]; then
    SAMPLER_CMD="$SAMPLER_CMD --orchestrator $ORCHESTRATOR_URL"
fi
nohup bash -c "$SAMPLER_CMD" > "$LOG_DIR/sampler_nohup.out" 2>&1 &
SAMPLER_PID=$!
echo $SAMPLER_PID > "$LOG_DIR/sampler.pid"
echo "Sampler started with PID: $SAMPLER_PID"

echo ""
echo "========================================="
echo "All processes started!"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "PIDs:"
echo "  Orchestrator: $ORCH_PID"
echo "  Trainer: $TRAINER_PID"
echo "  Sampler: $SAMPLER_PID"
echo ""
echo "Log files:"
echo "  $LOG_DIR/orchestrator.log"
echo "  $LOG_DIR/trainer.log"
echo "  $LOG_DIR/sampler.log"
echo "========================================="

