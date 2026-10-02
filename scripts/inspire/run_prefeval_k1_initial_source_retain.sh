#!/usr/bin/env bash
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen independent checkout required}
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_output="$task_root/runs/prefeval-k1-initial-source-retain-20260924/$arm"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || { echo 'Initial-source retain launcher already active'; exit 1; }
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
writer=scripts/experiments/prefeval_k1_writer.py
common=(--arm "$arm" --base "$task_models/DreamLite-base-a9a0f15-20260907"
  --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
if [[ ! -f "$task_output/train/complete.json" ]]; then
  "$task_python" "$writer" train "${common[@]}" --stage retain --retain-source-mode initial \
    --steps 2048 --teachers "$task_run/pilot/$arm" --sources "$task_run/writer/$arm/training-prefixes" \
    --checkpoint "$task_run/writer/$arm/write/checkpoint-final.pt" --output "$task_output/train" \
    > "$task_output/train.log" 2>&1
fi
for split in pilot dev; do
  "$task_python" "$writer" rollout "${common[@]}" --split "$split" --inter-turns 10 --noise-chains 2 \
    --checkpoint "$task_output/train/checkpoint-final.pt" --output "$task_output/$split" \
    > "$task_output/$split.log" 2>&1
  "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_output/$split" \
    --output "$task_output/readback-$split-T1" --families T1 --prefixes 0,1,5,10 --noise-chains 2 \
    --controls memory,mismatch --tasks mcq > "$task_output/readback-$split-T1.log" 2>&1
done
for split in pilot dev; do
  "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_output/$split" \
    --output "$task_output/readback-$split-T1" --families T1 --prefixes 0,1,5,10 --noise-chains 2 \
    --controls memory,mismatch --tasks free > "$task_output/readback-$split-T1-free.log" 2>&1
done
printf '0\n' > "$task_output/exit-status.txt"
