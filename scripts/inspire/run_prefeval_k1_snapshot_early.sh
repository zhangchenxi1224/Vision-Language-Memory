#!/usr/bin/env bash
# Read the prospectively fixed snapshot on spare GPUs; same destinations as the final driver.
set -euo pipefail
split=${1:?pilot or dev}
[[ "$split" == pilot || "$split" == dev ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_run="$task_root/runs/prefeval-b-mcq-20260925"
task_output="$task_run/robust730"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
exec 9>"$task_output/snapshot-$split.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/snapshot-$split-exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
artifact="$task_run/variants-train.json"
[[ "$split" != dev ]] || artifact="$task_run/variants-dev.json"
label="step-002048-$split-V1"
"$task_python" scripts/experiments/prefeval_k1_writer.py rollout --arm B --split "$split" \
 --base "$task_models/DreamLite-base-a9a0f15-20260907" \
 --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite" \
 --checkpoint "$task_output/train/checkpoint-step-002048.pt" \
 --initial-variants "$artifact" --initial-variant 1 --inter-turns 0 --noise-chains 2 \
 --output "$task_output/$label" > "$task_output/$label-early-rollout.log" 2>&1
"$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
 --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_output/$label" \
 --initial-variants "$artifact" --initial-variant 1 \
 --output "$task_output/readback-$label" --families T1 --prefixes 0 --noise-chains 2 \
 --controls memory,mismatch,blank,text --tasks mcq > "$task_output/$label-early-readback.log" 2>&1
