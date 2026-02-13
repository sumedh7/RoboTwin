#!/usr/bin/env bash
set -euo pipefail

task_name=${1}
setting=${2}
expert_data_num=${3}
num_workers=${4:-1}   # default: 1 worker (sequential, same as before)

if [ "$num_workers" -le 0 ]; then
    echo "ERROR: num_workers must be >= 1"
    exit 1
fi

if [ "$num_workers" -eq 1 ]; then
    echo "Running sequentially (1 worker) ..."
    python scripts/process_data.py "$task_name" "$setting" "$expert_data_num"
    exit 0
fi

# --- Parallel mode ---
echo "Processing $expert_data_num episodes with $num_workers parallel workers ..."

chunk_size=$(( (expert_data_num + num_workers - 1) / num_workers ))  # ceiling division
pids=()
log_dir=$(mktemp -d)

for (( w=0; w<num_workers; w++ )); do
    start=$(( w * chunk_size ))
    end=$(( start + chunk_size ))
    if [ "$end" -gt "$expert_data_num" ]; then
        end=$expert_data_num
    fi
    if [ "$start" -ge "$expert_data_num" ]; then
        break
    fi

    log_file="${log_dir}/worker_${w}.log"
    echo "  Worker $w: episodes [$start, $end)  -> $log_file"
    python -u scripts/process_data.py "$task_name" "$setting" "$expert_data_num" \
        --start "$start" --end "$end" \
        > "$log_file" 2>&1 &
    pids+=($!)
done

echo "Launched ${#pids[@]} workers. Waiting for completion ..."

# Wait for all workers; track failures
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: Worker $i (PID ${pids[$i]}) failed. See ${log_dir}/worker_${i}.log"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "$failed worker(s) failed. Logs are in $log_dir"
    exit 1
fi

echo "All $num_workers workers finished successfully."
echo "Logs are in $log_dir"
