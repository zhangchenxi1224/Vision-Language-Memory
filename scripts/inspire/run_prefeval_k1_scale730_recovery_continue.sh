#!/usr/bin/env bash
# Continue the original frozen FM stages after the diagnosed teacher shard recovery.
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo="$task_root/repos/prefeval-k1-scale730-331092d"
task_run="$task_root/runs/prefeval-k1-scale730-20260924"
task_recovery="$task_run/recovery-token-boundary"
exec 9>"$task_recovery/continuation.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_recovery/continuation-exit-status.txt"' EXIT
for arm in A B; do
  for shard in 0 1; do
    until [[ -f "$task_run/teachers/$arm/finished-$shard.json" ]]; do sleep 30; done
  done
done
# The original parent waits for its original children, then records the earlier A-0 failure.
until [[ -f "$task_run/exit-status.txt" ]]; do sleep 30; done
[[ $(cat "$task_run/exit-status.txt") == 1 ]]
[[ ! -e "$task_run/writer-processes.txt" ]]
[[ $(git -C "$task_repo" rev-parse HEAD) == 331092d2533c3b7cfb562fd15e69012c84c80f44 ]]
export K1_CODE_ROOT="$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$task_repo"
pids=()
for arm in A B; do
  gpu=0
  [[ "$arm" == A ]] || gpu=1
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/inspire/run_prefeval_k1_scale730_write.sh "$arm" \
    > "$task_run/write-$arm.log" 2>&1 &
  pids+=("$!")
  printf '%s %s %s\n' "$arm" "$gpu" "$!" >> "$task_run/writer-processes.txt"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
