#!/bin/bash
# Monitor SLURM jobs for crashes. Exits non-zero and prints failure info if any job crashes.
# Usage: bash scripts/monitor_jobs.sh <log_dir> <job_id1> [job_id2 ...]

set -euo pipefail

LOG_DIR="$1"
shift
JOB_IDS=("$@")
JOBS_CSV=$(IFS=,; echo "${JOB_IDS[*]}")

echo "=== MGMD JOB MONITOR STARTED $(date) ==="
echo "Tracking ${#JOB_IDS[@]} jobs: $JOBS_CSV"
echo "Log dir: $LOG_DIR"
echo ""

while true; do
    # Check sacct for any failed/cancelled/oom jobs
    failed_lines=$(sacct -j "$JOBS_CSV" --format=JobID,JobName,State,ExitCode --noheader --parsable2 2>/dev/null \
        | grep -vE '\.(batch|extern)' \
        | grep -vE '\|COMPLETED\||\|PENDING\||\|RUNNING\||\|REQUEUED\|' \
        | grep -E '\|FAILED\||\|OUT_OF_MEMORY\||\|NODE_FAIL\||\|TIMEOUT\|' || true)

    if [ -n "$failed_lines" ]; then
        echo ""
        echo "=== CRASH DETECTED at $(date) ==="
        echo "$failed_lines"
        echo ""
        echo "=== FAILED JOB LOG TAILS ==="
        while IFS='|' read -r jobid jobname state exitcode rest; do
            # jobid may have .0 or similar - strip suffix for log lookup
            base_id="${jobid%%.*}"
            out_file="$LOG_DIR/${base_id}.out"
            err_file="$LOG_DIR/${base_id}.err"
            echo ""
            echo "--- Job $jobid ($jobname) state=$state exit=$exitcode ---"
            if [ -f "$out_file" ]; then
                echo "[stdout tail]"
                tail -80 "$out_file"
            fi
            if [ -f "$err_file" ] && [ -s "$err_file" ]; then
                echo "[stderr tail]"
                tail -40 "$err_file"
            fi
        done <<< "$failed_lines"
        exit 1
    fi

    # Count still-live jobs
    live=$(squeue -u "$USER" --noheader -j "$JOBS_CSV" 2>/dev/null | wc -l || echo 0)
    echo "$(date '+%H:%M:%S') -- $live/${#JOB_IDS[@]} jobs still live (R or PD)"

    if [ "$live" -eq 0 ]; then
        echo "All jobs finished without detected failure."
        exit 0
    fi

    sleep 60
done
