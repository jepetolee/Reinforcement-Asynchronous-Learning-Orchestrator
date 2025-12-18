#!/bin/bash
# Start multiple samplers and one trainer on the same local host
# Usage: ./scripts/start_local_multi.sh --config configs/iiv_dapo_qwen3_4b.yaml

set -e

# Default values
CONFIG_FILE=""
ORCHESTRATOR_URL="http://127.0.0.1:59888"
ORCH_PORT=59888
NUM_SAMPLERS=3
TRAINER_GPUS="3"  # Comma-separated GPU IDs for trainer (e.g., "3" or "3,4,5,6")
NPROC_PER_NODE=1  # Number of processes per node for trainer
LOG_DIR=""
RUN_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
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
        --num-samplers)
            NUM_SAMPLERS="$2"
            shift 2
            ;;
        --trainer-gpus)
            TRAINER_GPUS="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --config CONFIG_FILE [--orchestrator URL] [--orch-port PORT] [--num-samplers N] [--trainer-gpus GPU_IDS] [--nproc-per-node N] [--log-dir DIR] [--run-id ID]"
            echo "  --trainer-gpus: Comma-separated GPU IDs (e.g., '3' or '3,4,5,6')"
            echo "  --nproc-per-node: Number of processes per node for trainer (default: 1)"
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

echo "========================================="
echo "Starting Multi-Process Local Training"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Orchestrator: $ORCHESTRATOR_URL"
echo "Number of samplers: $NUM_SAMPLERS"
echo "Trainer GPUs: $TRAINER_GPUS"
echo "Trainer processes per node: $NPROC_PER_NODE"
echo "========================================="
echo ""

# Step 1: Start Orchestrator
echo "Step 1: Starting Orchestrator..."
python ralo_cli.py orch \
    --config "$CONFIG_FILE" \
    --orch-port "$ORCH_PORT" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    --log-file "orchestrator.log" > "$LOG_DIR/orchestrator_nohup.out" 2>&1 &
ORCH_PID=$!
echo $ORCH_PID > "$LOG_DIR/orchestrator.pid"
echo "Orchestrator started with PID: $ORCH_PID"
echo "Waiting 5 seconds for orchestrator to initialize..."
sleep 5
echo ""

# Step 2: Start Trainer
echo "Step 2: Starting Trainer on GPUs $TRAINER_GPUS (nproc_per_node=$NPROC_PER_NODE)..."
export CUDA_VISIBLE_DEVICES=$TRAINER_GPUS
export ORCH_SERVER="$ORCHESTRATOR_URL"
export RUN_ID

# Check if we should use torchrun for multi-GPU training
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    # Use torchrun for distributed training
    MASTER_PORT=${MASTER_PORT:-29501}
    export MASTER_PORT
    TRAINER_CMD="torchrun --standalone --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT ralo_cli.py train --config $CONFIG_FILE --orchestrator $ORCHESTRATOR_URL --run-id $RUN_ID --log-dir $LOG_DIR --log-file trainer.log"
    echo "Using torchrun for distributed training"
else
    # Use python directly for single GPU
    TRAINER_CMD="python ralo_cli.py train --config $CONFIG_FILE --orchestrator $ORCHESTRATOR_URL --run-id $RUN_ID --log-dir $LOG_DIR --log-file trainer.log"
fi

bash -c "$TRAINER_CMD" > "$LOG_DIR/trainer_nohup.out" 2>&1 &
TRAINER_PID=$!
echo $TRAINER_PID > "$LOG_DIR/trainer.pid"
echo "Trainer started with PID: $TRAINER_PID"
echo "Waiting 3 seconds for trainer to initialize..."
sleep 3
echo ""

# Step 3: Start Multiple Samplers
echo "Step 3: Starting $NUM_SAMPLERS Samplers..."
SAMPLER_PIDS=()
for i in $(seq 0 $((NUM_SAMPLERS - 1))); do
    GPU_ID=$i
    echo "Starting Sampler $i on GPU $GPU_ID..."
    export GEN_DEVICES=$GPU_ID
    export ORCH_SERVER="$ORCHESTRATOR_URL"
    export RUN_ID
    python ralo_cli.py gen \
        --config "$CONFIG_FILE" \
        --orchestrator "$ORCHESTRATOR_URL" \
        --run-id "$RUN_ID" \
        --log-dir "$LOG_DIR" \
        --log-file "sampler_${i}.log" > "$LOG_DIR/sampler_${i}_nohup.out" 2>&1 &
    SAMPLER_PID=$!
    SAMPLER_PIDS+=($SAMPLER_PID)
    echo $SAMPLER_PID > "$LOG_DIR/sampler_${i}.pid"
    echo "Sampler $i started with PID: $SAMPLER_PID"
    sleep 1
done

echo ""
echo "========================================="
echo "All processes started!"
echo "========================================="
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "PIDs:"
echo "  Orchestrator: $ORCH_PID"
echo "  Trainer: $TRAINER_PID"
for i in $(seq 0 $((NUM_SAMPLERS - 1))); do
    echo "  Sampler $i: ${SAMPLER_PIDS[$i]}"
done
echo ""
echo "GPU Assignment:"
echo "  Orchestrator: CPU only"
echo "  Trainer: GPUs $TRAINER_GPUS (nproc_per_node=$NPROC_PER_NODE)"
for i in $(seq 0 $((NUM_SAMPLERS - 1))); do
    echo "  Sampler $i: GPU $i"
done
echo ""
echo "Log files:"
echo "  $LOG_DIR/orchestrator.log"
echo "  $LOG_DIR/trainer.log"
for i in $(seq 0 $((NUM_SAMPLERS - 1))); do
    echo "  $LOG_DIR/sampler_${i}.log"
done
echo ""
echo "To stop all processes:"
echo "  ./scripts/stop_all.sh --log-dir $LOG_DIR"
echo "========================================="

