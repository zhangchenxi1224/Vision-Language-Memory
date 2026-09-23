#!/usr/bin/env bash
set -euo pipefail
phase=${1:?smoke or pilot}
[[ "$phase" == smoke || "$phase" == pilot ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/prefeval-k1-l0-l2-20260924"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924/$phase"
cd "$task_repo"
mkdir -p "$task_run"
exec 9>"$task_run/launcher.lock"
flock -n 9 || { echo 'This phase already has a live launcher'; exit 1; }
git rev-parse HEAD > "$task_run/commit.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
pids=()
for gpu in 0 1 2 3; do
  if ((gpu < 2)); then arm=A; shard=$gpu; else arm=B; shard=$((gpu - 2)); fi
  extra=()
  if [[ "$phase" == smoke ]]; then
    [[ "$shard" == 0 ]] || continue
    extra=(--steps 12 --limit 1)
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$task_python" scripts/experiments/prefeval_k1_teacher.py \
    --arm "$arm" --shard "$shard" --shards 2 \
    --base "$task_models/DreamLite-base-a9a0f15-20260907" \
    --reader "$task_models/Qwen3-VL-4B-Instruct" \
    --prefeval "$task_repo/third_party/prefeval_reference" \
    --output "$task_run/$arm" "${extra[@]}" > "$task_run/$arm-$shard.log" 2>&1 &
  pids+=("$!")
  printf '%s %s %s %s\n' "$arm" "$shard" "$gpu" "$!" >> "$task_run/processes.txt"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
printf '%s\n' "$failed" > "$task_run/exit-status.txt"
exit "$failed"
