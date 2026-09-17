#!/usr/bin/env bash
set -uo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/dreamlite-prefeval-rgb-20260917"
task_run="$task_root/runs/dreamlite-prefeval-rgb-20260917"
stage="$task_run/visual-recovery-v1"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
cd "$task_repo" || exit 1
expected_commit="${1:?Supply exact execution commit}"
[[ "$(git rev-parse HEAD)" == "$expected_commit" && -z "$(git status --porcelain)" ]] || exit 1
mkdir "$stage" || exit 1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$stage/gpus-at-dispatch.csv"
git rev-parse HEAD > "$stage/code-commit.txt"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$task_python" scripts/reporting/verify_prefeval_rgb_references.py --directory "$task_run/references-v2" \
  --overlay reports/prefeval-rgb-20260917/reader-format-v2.json --output "$stage/references-v2-reconstructed.json" || exit 1
"$task_python" scripts/experiments/prefeval_visual_policy.py --policy reports/prefeval-rgb-20260917/visual-recovery-allocation-v1.json \
  --verification "$stage/references-v2-reconstructed.json" > "$stage/allocation-check.json" || exit 1
for phase in baseline sentinel; do
  output="$stage/$phase"
  mkdir -p "$output"
  git rev-parse HEAD > "$output/code-commit.txt"
  pids=()
  for shard in 0 1 2 3; do
    common=(--manifest reports/prefeval-rgb-20260917/registered/manifest.json --overlay reports/prefeval-rgb-20260917/reader-format-v2.json
      --policy reports/prefeval-rgb-20260917/visual-recovery-allocation-v1.json
      --reader "$task_models/Qwen3-VL-4B-Instruct" --base "$task_models/DreamLite-base-a9a0f15-20260907"
      --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite" --device cuda:0 --shards 4 --shard "$shard")
    if [[ "$phase" == baseline ]]; then
      command=(scripts/eval/prefeval_rgb_baseline.py --output "$output/shard-$shard" --package "$task_root/runs/dreamlite-official-alignment/e372f3c-logical-package")
    else
      command=(scripts/experiments/build_prefeval_rgb_teachers.py --output "$output" --sentinel)
    fi
    CUDA_VISIBLE_DEVICES="$shard" "$task_python" "${command[@]}" "${common[@]}" > "$output/shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  printf '%s\n' "$status" > "$output/exit-status.txt"
  [[ "$status" == 0 ]] || exit "$status"
done
