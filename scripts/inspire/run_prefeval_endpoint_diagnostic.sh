#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/dreamlite-prefeval-rgb-20260917"
stage="$task_root/runs/dreamlite-prefeval-rgb-20260917/visual-recovery-v1-run"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
cd "$task_repo"
[[ "$(git rev-parse HEAD)" == "${1:?Exact execution commit required}" && -z "$(git status --porcelain)" ]]
[[ "$(cat "$stage/sentinel-native-gray-r2/exit-status.txt")" == 0 ]]
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
output="$stage/endpoint-diagnostic-v1"
mkdir "$output"
git rev-parse HEAD > "$output/code-commit.txt"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$task_python" scripts/probes/prefeval_rgb_endpoint_diagnostic.py \
    --source "$stage/sentinel-native-gray-r2" --registration reports/prefeval-rgb-20260917/endpoint-diagnostic-v1.json \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --base "$task_models/DreamLite-base-a9a0f15-20260907" \
    --device cuda:0 --shards 4 --shard "$shard" --output "$output" > "$output/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf '%s\n' "$status" > "$output/exit-status.txt"
if [[ "$status" == 0 ]]; then
  "$task_python" scripts/reporting/verify_prefeval_rgb_endpoint_diagnostic.py --source "$stage/sentinel-native-gray-r2" \
    --output "$output" --result "$output/verified.json" > "$output/verification.log" 2>&1 || status=1
fi
printf '%s\n' "$status" > "$output/verified-exit-status.txt"
exit "$status"
