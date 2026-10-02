#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen scale730 code root}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-scale730-20260924"
mkdir -p "$task_run"
exec 9>"$task_run/launcher.lock"
flock -n 9 || { echo 'Scale730 launcher already active'; exit 1; }
trap 'printf "%s\n" "$?" > "$task_run/exit-status.txt"' EXIT
cd "$task_repo"
git rev-parse HEAD > "$task_run/code-commit.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python3 scripts/inspire/prepare_prefeval_k1_scale730_bank.py \
  --pilot "$task_root/runs/prefeval-k1-l0-l2-20260924/pilot" --output "$task_run/teachers"
pids=()
for arm in A B; do
  gpu=0
  [[ "$arm" == A ]] || gpu=1
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu "$task_python" scripts/experiments/prefeval_k1_teacher.py \
      --arm "$arm" --split train --exclude-pilot --shard "$shard" --shards 2 \
      --base "$task_models/DreamLite-base-a9a0f15-20260907" \
      --reader "$task_models/Qwen3-VL-4B-Instruct" --prefeval "$task_repo/third_party/prefeval_reference" \
      --output "$task_run/teachers/$arm" > "$task_run/teacher-$arm-$shard.log" 2>&1 &
    pids+=("$!")
    printf '%s %s %s %s\n' "$arm" "$shard" "$gpu" "$!" >> "$task_run/teacher-processes.txt"
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
pids=()
for arm in A B; do
  gpu=0
  [[ "$arm" == A ]] || gpu=1
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/inspire/run_prefeval_k1_scale730_write.sh "$arm" \
    > "$task_run/write-$arm.log" 2>&1 &
  pids+=("$!")
  printf '%s %s %s\n' "$arm" "$gpu" "$!" >> "$task_run/writer-processes.txt"
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
