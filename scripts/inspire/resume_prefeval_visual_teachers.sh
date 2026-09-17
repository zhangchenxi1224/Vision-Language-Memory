#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/dreamlite-prefeval-rgb-20260917"
stage="$task_root/runs/dreamlite-prefeval-rgb-20260917/visual-recovery-v1-run"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
cd "$task_repo"
[[ "$(git rev-parse HEAD)" == "${1:?Exact execution commit required}" && -z "$(git status --porcelain)" ]]
[[ "$(cat "$stage/baseline/exit-status.txt")" == 0 ]]
"$task_python" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); assert sum(json.load(open(p/f"shard-{i}"/"complete.json"))["writes"] for i in range(4))==124' "$stage/baseline"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$task_python" scripts/experiments/prefeval_visual_policy.py --policy reports/prefeval-rgb-20260917/visual-recovery-allocation-v1.json --verification "$stage/references-v2-reconstructed.json"
output="$stage/sentinel-native-gray"
mkdir "$output"
git rev-parse HEAD > "$output/code-commit.txt"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$task_python" scripts/experiments/build_prefeval_rgb_teachers.py \
    --manifest reports/prefeval-rgb-20260917/registered/manifest.json --overlay reports/prefeval-rgb-20260917/reader-format-v2.json \
    --policy reports/prefeval-rgb-20260917/visual-recovery-allocation-v1.json \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --base "$task_models/DreamLite-base-a9a0f15-20260907" \
    --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite" \
    --device cuda:0 --shards 4 --shard "$shard" --output "$output" --sentinel > "$output/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf '%s\n' "$status" > "$output/exit-status.txt"
exit "$status"
