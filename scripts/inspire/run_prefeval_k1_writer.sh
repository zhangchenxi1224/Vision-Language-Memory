#!/usr/bin/env bash
# Start only after the complete paired target bank exists. No score-based filtering.
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:-$task_root/repos/prefeval-k1-l0-l2-20260924}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_arm="$task_run/writer/$arm"
cd "$task_repo"
mkdir -p "$task_arm"
exec 9>"$task_arm/launcher.lock"
flock -n 9 || { echo 'Writer launcher already active'; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for pair_arm in A B; do
  for shard in 0 1; do test -f "$task_run/pilot/$pair_arm/finished-$shard.json"; done
done
common=(--arm "$arm" --base "$task_models/DreamLite-base-a9a0f15-20260907"
  --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
writer=scripts/experiments/prefeval_k1_writer.py
parent="$task_root/runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt"
if [[ ! -f "$task_arm/write/complete.json" ]]; then
  "$task_python" "$writer" train "${common[@]}" --stage write --teachers "$task_run/pilot/$arm" \
    --checkpoint "$parent" --output "$task_arm/write" > "$task_arm/write.log" 2>&1
fi
"$task_python" "$writer" rollout "${common[@]}" --checkpoint "$task_arm/write/checkpoint-final.pt" \
  --noise-chains 1 --output "$task_arm/training-prefixes" > "$task_arm/training-prefixes.log" 2>&1
if [[ ! -f "$task_arm/retain/complete.json" ]]; then
  "$task_python" "$writer" train "${common[@]}" --stage retain --teachers "$task_run/pilot/$arm" \
    --sources "$task_arm/training-prefixes" --checkpoint "$task_arm/write/checkpoint-final.pt" \
    --output "$task_arm/retain" > "$task_arm/retain.log" 2>&1
fi
for split in pilot dev; do
  "$task_python" "$writer" rollout "${common[@]}" --split "$split" \
    --checkpoint "$task_arm/retain/checkpoint-final.pt" --output "$task_arm/final-$split" \
    > "$task_arm/final-$split.log" 2>&1
done
printf '0\n' > "$task_arm/exit-status.txt"
