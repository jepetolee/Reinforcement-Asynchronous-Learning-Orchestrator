#!/bin/bash
# Stop SLURM jobs using job ID files

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

# Function to stop a SLURM job by job ID file
stop_slurm_job() {
    local jobid_file="$1"
    local process_name="$2"
    
    if [ -f "$jobid_file" ]; then
        JOB_ID=$(cat "$jobid_file")
        if [ -n "$JOB_ID" ]; then
            echo "Stopping $process_name (SLURM Job ID: $JOB_ID)..."
            scancel "$JOB_ID" 2>/dev/null && echo "$process_name job $JOB_ID cancelled" || echo "Failed to cancel $process_name job $JOB_ID (may already be finished)"
        else
            echo "$process_name job ID file is empty: $jobid_file"
        fi
    else
        echo "$process_name job ID file not found: $jobid_file"
    fi
}

# Stop processes in a log directory
stop_processes_in_dir() {
    local log_dir="$1"
    if [ ! -d "$log_dir" ]; then
        echo "Warning: Log directory does not exist: $log_dir"
        return
    fi
    echo "Stopping SLURM jobs in $log_dir..."
    stop_slurm_job "$log_dir/orchestrator_slurm_jobid.txt" "orchestrator"
    stop_slurm_job "$log_dir/trainer_slurm_jobid.txt" "trainer"
    stop_slurm_job "$log_dir/sampler_slurm_jobid.txt" "sampler"
    echo ""
}

# Handle multiple directories from wildcard expansion
if [ ${#LOG_DIRS[@]} -gt 0 ]; then
    # If LOG_DIR is also set, add it to LOG_DIRS
    if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
        LOG_DIRS+=("$LOG_DIR")
    fi
    echo "Stopping SLURM jobs in ${#LOG_DIRS[@]} log directory(ies)..."
    echo ""
    for dir in "${LOG_DIRS[@]}"; do
        stop_processes_in_dir "$dir"
    done
elif [ -z "$LOG_DIR" ]; then
    echo "Error: --log-dir is required"
    echo "Usage: $0 --log-dir <log_directory>"
    echo "       $0 --log-dir 'logs/*'  # Stop all jobs in all log directories (use quotes)"
    echo "       $0 --log-dir logs/20251120_*  # Stop jobs in matching directories"
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
    
    echo "Stopping SLURM jobs in all log directories matching pattern: $LOG_DIR"
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

