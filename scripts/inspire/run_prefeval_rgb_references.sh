#!/usr/bin/env bash
set -uo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/dreamlite-prefeval-rgb-20260917"
task_output="$task_root/runs/dreamlite-prefeval-rgb-20260917/references-v1"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_reader=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct
mkdir -p "$task_output"
cd "$task_repo" || exit 1
git rev-parse HEAD > "$task_output/code-commit.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$task_output/gpus-at-dispatch.csv"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    "$task_python" scripts/eval/prefeval_rgb.py \
      --manifest reports/prefeval-rgb-20260917/registered/manifest.json \
      --reader "$task_reader" --output "$task_output/shard-$shard" \
      --device cuda:0 --shards 4 --shard "$shard" \
      > "$task_output/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
printf '%s\n' "$status" > "$task_output/exit-status.txt"
exit "$status"
