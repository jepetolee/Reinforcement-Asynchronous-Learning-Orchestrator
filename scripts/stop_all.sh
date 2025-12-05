#!/bin/bash
# Stop all processes using PID files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
LOG_DIR=""
LOG_DIRS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            # If next argument starts with --, it's another option, not a value
            if [[ "$2" == --* ]] || [ -z "$2" ]; then
                echo "Error: --log-dir requires a value"
                exit 1
            fi
            # Check if it's a wildcard pattern or multiple directories
            if [[ "$2" == *"/*" ]] || [[ "$2" == *"*" ]]; then
                # Wildcard pattern - handle in main logic
                LOG_DIR="$2"
                shift 2
            else
                # Single directory or first of multiple
                LOG_DIR="$2"
                shift 2
                # Collect any additional directories (from wildcard expansion)
                while [[ $# -gt 0 ]] && [[ "$1" != --* ]] && [ -d "$1" ]; do
                    LOG_DIRS+=("$1")
                    shift
                done
            fi
            ;;
        *)
            # If we already have --log-dir, treat remaining args as directories
            if [ -n "$LOG_DIR" ] && [ -d "$1" ]; then
                LOG_DIRS+=("$1")
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

# Function to stop a process by PID file
stop_process() {
    local pid_file="$1"
    local process_name="$2"
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "Stopping $process_name (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            # Wait a bit, then force kill if still running
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing $process_name (PID: $PID)..."
                kill -9 "$PID" 2>/dev/null || true
            fi
            rm -f "$pid_file"
            echo "$process_name stopped"
        else
            echo "$process_name (PID: $PID) is not running"
            rm -f "$pid_file"
        fi
    else
        echo "$process_name PID file not found: $pid_file"
    fi
}

# Stop processes in a log directory
stop_processes_in_dir() {
    local log_dir="$1"
    if [ ! -d "$log_dir" ]; then
        echo "Warning: Log directory does not exist: $log_dir"
        return
    fi
    echo "Stopping processes in $log_dir..."
    stop_process "$log_dir/orchestrator.pid" "orchestrator"
    stop_process "$log_dir/trainer.pid" "trainer"
    stop_process "$log_dir/sampler.pid" "sampler"
    # Also handle multiple samplers (sampler1, sampler2, etc.)
    stop_process "$log_dir/sampler1.pid" "sampler1"
    stop_process "$log_dir/sampler2.pid" "sampler2"
    echo ""
}

# Handle multiple directories from wildcard expansion
if [ ${#LOG_DIRS[@]} -gt 0 ]; then
    # If LOG_DIR is also set, add it to LOG_DIRS
    if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
        LOG_DIRS+=("$LOG_DIR")
    fi
    echo "Stopping processes in ${#LOG_DIRS[@]} log directory(ies)..."
    echo ""
    for dir in "${LOG_DIRS[@]}"; do
        stop_processes_in_dir "$dir"
    done
elif [ -z "$LOG_DIR" ]; then
    echo "Error: --log-dir is required"
    echo "Usage: $0 --log-dir <log_directory>"
    echo "       $0 --log-dir 'logs/*'  # Stop all processes in all log directories (use quotes)"
    echo "       $0 --log-dir logs/20251120_*  # Stop processes in matching directories"
    exit 1
# Check if LOG_DIR contains wildcards (when quoted, shell won't expand)
elif [[ "$LOG_DIR" == *"/*" ]] || [[ "$LOG_DIR" == *"*" ]]; then
    # Remove /* or * and use the parent directory
    if [[ "$LOG_DIR" == *"/*" ]]; then
        parent_dir="${LOG_DIR%/*}"
        pattern="*/"
    else
        # Extract parent and pattern
        parent_dir="$(dirname "$LOG_DIR")"
        pattern="$(basename "$LOG_DIR")"
    fi
    
    if [ ! -d "$parent_dir" ]; then
        echo "Error: Parent directory does not exist: $parent_dir"
        exit 1
    fi
    
    echo "Stopping processes in all log directories matching pattern: $LOG_DIR"
    echo ""
    found=0
    for dir in "$parent_dir"/$pattern; do
        if [ -d "$dir" ]; then
            stop_processes_in_dir "$dir"
            found=1
        fi
    done
    if [ $found -eq 0 ]; then
        echo "No log directories found matching pattern: $LOG_DIR"
    fi
elif [ -d "$LOG_DIR" ]; then
    # Single directory
    stop_processes_in_dir "$LOG_DIR"
else
    echo "Error: Log directory does not exist: $LOG_DIR"
    exit 1
fi

echo "Done!"

