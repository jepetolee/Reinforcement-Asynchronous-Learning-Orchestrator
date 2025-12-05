#!/bin/bash
# Check status of all processes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$LOG_DIR" ]; then
    echo "Error: --log-dir is required"
    echo "Usage: $0 --log-dir <log_directory>"
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory does not exist: $LOG_DIR"
    exit 1
fi

# Function to check process status
check_process() {
    local pid_file="$1"
    local process_name="$2"
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "$process_name: RUNNING (PID: $PID)"
            return 0
        else
            echo "$process_name: NOT RUNNING (PID: $PID)"
            return 1
        fi
    else
        echo "$process_name: NO PID FILE"
        return 1
    fi
}

echo "========================================="
echo "Process Status for: $LOG_DIR"
echo "========================================="

# Check run_id
if [ -f "$LOG_DIR/run_id.txt" ]; then
    RUN_ID=$(cat "$LOG_DIR/run_id.txt")
    echo "Run ID: $RUN_ID"
    echo ""
fi

# Check all processes
ORCH_RUNNING=0
TRAINER_RUNNING=0
SAMPLER_RUNNING=0

check_process "$LOG_DIR/orchestrator.pid" "Orchestrator" && ORCH_RUNNING=1
check_process "$LOG_DIR/trainer.pid" "Trainer" && TRAINER_RUNNING=1
check_process "$LOG_DIR/sampler.pid" "Sampler" && SAMPLER_RUNNING=1

echo ""
echo "Log files:"
[ -f "$LOG_DIR/orchestrator.log" ] && echo "  orchestrator.log: $(wc -l < "$LOG_DIR/orchestrator.log") lines"
[ -f "$LOG_DIR/trainer.log" ] && echo "  trainer.log: $(wc -l < "$LOG_DIR/trainer.log") lines"
[ -f "$LOG_DIR/sampler.log" ] && echo "  sampler.log: $(wc -l < "$LOG_DIR/sampler.log") lines"

echo ""
if [ $ORCH_RUNNING -eq 1 ] && [ $TRAINER_RUNNING -eq 1 ] && [ $SAMPLER_RUNNING -eq 1 ]; then
    echo "Status: ALL RUNNING"
    exit 0
else
    echo "Status: SOME PROCESSES NOT RUNNING"
    exit 1
fi

