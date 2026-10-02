#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_output="$task_root/runs/prefeval-b-mcq-20260925/identity64"
task_old="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
common=(--arm B --base "$task_models/DreamLite-base-a9a0f15-20260907"
 --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
writer=scripts/experiments/prefeval_k1_writer.py
if [[ ! -f "$task_output/train/complete.json" ]]; then
 "$task_python" "$writer" train "${common[@]}" --stage retain --retain-source-mode initial \
  --retain-target-mode source --steps 2048 --teachers "$task_old/pilot/B" \
  --sources "$task_old/writer/B/training-prefixes" --checkpoint "$task_old/writer/B/write/checkpoint-final.pt" \
  --output "$task_output/train" > "$task_output/train.log" 2>&1
fi
"$task_python" "$writer" rollout "${common[@]}" --checkpoint "$task_output/train/checkpoint-final.pt" \
 --probe-initial-sources "$task_old/writer/B/training-prefixes" --inter-turns 1 --noise-chains 2 \
 --output "$task_output/seen-source" > "$task_output/seen-source.log" 2>&1
"$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split pilot \
 --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_output/seen-source" \
 --output "$task_output/seen-source-T1" --families T1 --prefixes 0,1 --noise-chains 2 \
 --controls memory,mismatch --tasks mcq > "$task_output/seen-source-T1.log" 2>&1
for split in pilot dev; do
 "$task_python" "$writer" rollout "${common[@]}" --split "$split" --inter-turns 10 --noise-chains 2 \
  --checkpoint "$task_output/train/checkpoint-final.pt" --output "$task_output/$split" > "$task_output/$split.log" 2>&1
 "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
  --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_output/$split" \
  --output "$task_output/readback-$split-T1" --families T1 --prefixes 0,1,5,10 --noise-chains 2 \
  --controls memory,mismatch,blank,text --tasks mcq > "$task_output/readback-$split-T1.log" 2>&1
done
