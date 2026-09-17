#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/dreamlite-prefeval-rgb-20260917"
source="$task_root/runs/dreamlite-prefeval-rgb-20260917/visual-recovery-v1-run/sentinel-native-gray-r2"
output="$task_root/runs/dreamlite-prefeval-rgb-20260917/semantic-transfer-v1-run"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
cd "$task_repo"
[[ "$(git rev-parse HEAD)" == "${1:?Exact execution commit required}" && -z "$(git status --porcelain)" ]]
phase="${2:?feasibility, paired, ranking-calibration, trial or joint}"
[[ "$phase" == feasibility || "$phase" == paired || "$phase" == ranking-calibration || "$phase" == trial || "$phase" == joint ]]
driver=scripts/experiments/prefeval_semantic_transfer.py
if [[ "$phase" == joint ]]; then
  [[ "${3:?Explicit instance required}" == dl-clear-retain-h200x4-20260914 ]]
  source="$task_root/runs/dreamlite-prefeval-rgb-20260917/ranking-learning-trial-v1-run"
  output="$task_root/runs/dreamlite-prefeval-rgb-20260917/joint-consolidation-v1-run"
  driver=scripts/experiments/prefeval_joint_consolidation.py
  [[ ! -e "$output" ]]
fi
if [[ "$phase" == ranking-calibration ]]; then output="$task_root/runs/dreamlite-prefeval-rgb-20260917/semantic-ranking-v1-run"; fi
if [[ "$phase" == trial ]]; then
  [[ "${3:?Explicit trial instance required}" == dl-clear-retain-h200x4-20260914 ]]
  output="$task_root/runs/dreamlite-prefeval-rgb-20260917/ranking-learning-trial-v1-run"
  [[ ! -e "$output" ]]
fi
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$output"
git rev-parse HEAD > "$output/$phase-code-commit.txt"
run_stage() {
  local mode="$1"
  local pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard" "$task_python" "$driver" --mode "$mode" \
      --source "$source" --output "$output" --base "$task_models/DreamLite-base-a9a0f15-20260907" \
      --reader "$task_models/Qwen3-VL-4B-Instruct" --device cuda:0 --shards 4 --shard "$shard" > "$output/$mode-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  printf '%s\n' "$status" > "$output/$mode-exit-status.txt"
  return "$status"
}
if [[ "$phase" == joint ]]; then
  hostname > "$output/hostname.txt"
  nvidia-smi > "$output/gpu-before.txt"
  run_stage train
  run_stage evaluate
  "$task_python" scripts/reporting/verify_prefeval_joint_consolidation.py --output "$output" --source "$source" > "$output/final-verification.log" 2>&1
elif [[ "$phase" == trial ]]; then
  printf '%s\n' "$3" > "$output/instance.txt"
  hostname > "$output/hostname.txt"
  nvidia-smi > "$output/gpu-before.txt"
  run_stage trial-train
  run_stage trial-evaluate
  "$task_python" scripts/reporting/verify_prefeval_ranking_trial.py --output "$output" --source "$source" > "$output/final-verification.log" 2>&1
elif [[ "$phase" == ranking-calibration ]]; then
  run_stage ranking-calibrate
  "$task_python" scripts/reporting/verify_prefeval_semantic_transfer.py --mode ranking-calibration --output "$output" > "$output/calibration-verification.log" 2>&1
elif [[ "$phase" == feasibility ]]; then
  run_stage feasibility
  "$task_python" scripts/reporting/verify_prefeval_semantic_transfer.py --mode feasibility --output "$output" > "$output/feasibility-verification.log" 2>&1
else
  "$task_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$output/feasibility-verified.json"
  run_stage train
  run_stage evaluate
  "$task_python" scripts/reporting/verify_prefeval_semantic_transfer.py --mode final --output "$output" --source "$source" > "$output/final-verification.log" 2>&1
fi
