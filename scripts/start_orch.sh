#!/bin/bash
# Start orchestrator in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
CONFIG_FILE=""
LOG_DIR=""
RUN_ID=""
ORCH_PORT=""
ORCH_HOST=""

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
        --orch-port)
            ORCH_PORT="$2"
            shift 2
            ;;
        --orch-host)
            ORCH_HOST="$2"
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

# Set up environment variables (use argument or inherit from environment)
if [ -n "$ORCH_HOST" ]; then
    export ORCH_HOST="$ORCH_HOST"
elif [ -z "${ORCH_HOST:-}" ]; then
    export ORCH_HOST="0.0.0.0"
fi

if [ -n "$ORCH_PORT" ]; then
    export ORCH_PORT="$ORCH_PORT"
elif [ -z "${ORCH_PORT:-}" ]; then
    # Use port from config or default
    :
fi

# Build command
CMD="python ralo_cli.py orch --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
if [ -n "$ORCH_PORT" ]; then
    CMD="$CMD --orch-port $ORCH_PORT"
fi

# Export RUN_ID for child processes
export RUN_ID

# Start in background
echo "Starting orchestrator with run_id: $RUN_ID"
echo "Log directory: $LOG_DIR"
nohup $CMD > "$LOG_DIR/orchestrator_nohup.out" 2>&1 &
PID=$!

# Save PID
echo $PID > "$LOG_DIR/orchestrator.pid"
echo "Orchestrator started with PID: $PID"
echo "Log file: $LOG_DIR/orchestrator.log"
echo "Run ID: $RUN_ID"

