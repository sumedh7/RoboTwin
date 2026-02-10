#!/bin/bash
# ---------------------------------------------------------------------------
# Parallel data collection across multiple GPUs.
#
# Usage:
#   bash parallel_collect_data.sh <task_name> <task_config> [num_gpus]
#
# Example:
#   bash parallel_collect_data.sh place_anyobject_stand demo_randomized 8
#
# Each GPU runs an independent shard with its own seed range and output
# directory.  After all shards finish, results are merged and episode
# instructions are generated once.
# ---------------------------------------------------------------------------
set -euo pipefail

task_name=${1:?"Usage: $0 <task_name> <task_config> [num_gpus]"}
task_config=${2:?"Usage: $0 <task_name> <task_config> [num_gpus]"}
num_gpus=${3:-8}

# --- init ---
# Activate the RoboTwin conda environment so that all python calls
# (config parsing, worker processes, merge script) use the right env.
eval "$(conda shell.bash hook)"
conda activate RoboTwin

[ -x ./script/.update_path.sh ] && ./script/.update_path.sh > /dev/null 2>&1 || true

# Read episode_num and language_num from the config YAML
episode_num=$(python -c "
import yaml
cfg = yaml.safe_load(open('task_config/${task_config}.yml'))
print(cfg['episode_num'])
")
language_num=$(python -c "
import yaml
cfg = yaml.safe_load(open('task_config/${task_config}.yml'))
print(cfg.get('language_num', 100))
")

episodes_per_shard=$(( (episode_num + num_gpus - 1) / num_gpus ))
seed_gap=100000  # non-overlapping seed ranges per shard

shard_base="./data/.shards/${task_name}/${task_config}"
mkdir -p "${shard_base}"

echo "========================================"
echo "  Parallel Data Collection"
echo "========================================"
echo "  Task:       ${task_name}"
echo "  Config:     ${task_config}"
echo "  Episodes:   ${episode_num}"
echo "  GPUs:       ${num_gpus}"
echo "  Per shard:  ${episodes_per_shard}"
echo "========================================"
echo ""

# --- launch shards ---
pids=()
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    remaining=$((episode_num - gpu_id * episodes_per_shard))
    shard_episodes=${episodes_per_shard}
    if [ "${remaining}" -lt "${shard_episodes}" ]; then
        shard_episodes=${remaining}
    fi
    if [ "${shard_episodes}" -le 0 ]; then
        break
    fi

    seed_start=$((gpu_id * seed_gap))
    shard_path="${shard_base}/shard_${gpu_id}"

    echo "  [GPU ${gpu_id}]  ${shard_episodes} episodes, seeds from ${seed_start}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "${task_name}" "${task_config}" \
        --seed-start "${seed_start}" \
        --num-episodes "${shard_episodes}" \
        --save-path "${shard_path}" \
        --skip-instructions \
        > "${shard_base}/shard_${gpu_id}.log" 2>&1 &

    pids+=($!)
done

num_workers=${#pids[@]}
echo ""
echo "  ${num_workers} workers launched. Progress:"
echo ""

# Helper: count lines matching a pattern across shard logs (pipefail-safe)
count_matches() { (grep -rh "$1" "${shard_base}"/shard_*.log 2>/dev/null || true) | wc -l; }
count_hdf5()    { (find "${shard_base}" -name '*.hdf5' 2>/dev/null || true) | wc -l; }

# --- poll progress until all workers finish ---
# Runs in the foreground; checks every 5 seconds.
while true; do
    # How many workers still alive?
    alive=0
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && ((alive++)) || true
    done

    seeds_done=$(count_matches 'success!')
    data_done=$(count_hdf5)

    # Print progress (overwrite line in place)
    if [ "${data_done}" -gt 0 ]; then
        printf "\r  [data collection]  %d / %d episodes saved  |  %d workers active    " \
            "${data_done}" "${episode_num}" "${alive}"
    else
        printf "\r  [seed collection]  %d / %d seeds found  |  %d workers active    " \
            "${seeds_done}" "${episode_num}" "${alive}"
    fi

    # Stop once every worker has exited
    [ "${alive}" -eq 0 ] && break
    sleep 5
done
echo ""
echo ""

# --- collect exit codes ---
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" 2>/dev/null || ((failed++)) || true
done

# Final summary
total_seeds=$(count_matches 'success!')
total_data=$(count_hdf5)
echo "  Seeds found: ${total_seeds} / ${episode_num}"
echo "  Data saved:  ${total_data} / ${episode_num}"

if [ "${failed}" -gt 0 ]; then
    echo ""
    echo "  ${failed} shard(s) failed. Check logs in ${shard_base}/shard_*.log"
    echo "  Merging whatever succeeded."
fi

# --- merge ---
echo ""
echo "=== Merging shards ==="
python script/merge_shards.py "${task_name}" "${task_config}" "${num_gpus}"

# --- clean up shard dirs ---
rm -rf "${shard_base}"

# --- generate instructions on merged data ---
echo ""
echo "=== Generating episode instructions ==="
cd description && bash gen_episode_instructions.sh "${task_name}" "${task_config}" "${language_num}"
cd ..

# --- final cleanup ---
rm -rf "data/${task_name}/${task_config}/.cache"

echo ""
echo "=== Done ==="
