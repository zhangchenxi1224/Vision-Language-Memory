#!/usr/bin/env bash
set -euo pipefail
shard=${1:?0 or 1}
[[ "$shard" == 0 || "$shard" == 1 ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_output="$task_root/runs/prefeval-b-mcq-20260925/neutral-acks"
mkdir -p "$task_output"
exec 9>"$task_output/shard-$shard.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/shard-$shard-exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$task_root/envs/vlm-r3-ngc2502/bin/python" scripts/experiments/prefeval_k1_neutral_ack.py \
 --reader /inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct \
 --output "$task_output" --shard "$shard" --shards 2
