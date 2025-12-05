#!/bin/bash
# Start sampler in background

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
GEN_DEVICES=""

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
        --gen-devices)
            GEN_DEVICES="$2"
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

if [ -n "$GEN_DEVICES" ]; then
    export GEN_DEVICES="$GEN_DEVICES"
elif [ -z "${GEN_DEVICES:-}" ]; then
    # GEN_DEVICES not set, will use default [0]
    :
fi

# Build command
CMD="python ralo_cli.py gen --config $CONFIG_FILE --run-id $RUN_ID --log-dir $LOG_DIR"
if [ -n "$ORCHESTRATOR_URL" ]; then
    CMD="$CMD --orchestrator $ORCHESTRATOR_URL"
fi

# Export RUN_ID for child processes
export RUN_ID

# Start in background
echo "Starting sampler with run_id: $RUN_ID"
echo "Log directory: $LOG_DIR"
nohup $CMD > "$LOG_DIR/sampler_nohup.out" 2>&1 &
PID=$!

# Save PID
echo $PID > "$LOG_DIR/sampler.pid"
echo "Sampler started with PID: $PID"
echo "Log file: $LOG_DIR/sampler.log"
echo "Run ID: $RUN_ID"

