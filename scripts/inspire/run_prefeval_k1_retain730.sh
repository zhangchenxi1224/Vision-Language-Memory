#!/usr/bin/env bash
set -euo pipefail
task_arm=${1:?Expected C or I}
[[ "$task_arm" == C || "$task_arm" == I ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_run="$task_root/runs/prefeval-b-mcq-20260925"
task_output="$task_run/retain730-$task_arm"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[[ -f "$task_run/sourcebank730/bank.json" ]]
common=(--arm B --base "$task_models/DreamLite-base-a9a0f15-20260907"
 --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
task_target=teacher
[[ "$task_arm" != I ]] || task_target=source
writer=scripts/experiments/prefeval_k1_writer.py
if [[ ! -f "$task_output/train/complete.json" ]]; then
 "$task_python" "$writer" train "${common[@]}" --split train --stage retain --steps 23360 \
  --retain-source-mode four-bank --retain-target-mode "$task_target" \
  --initial-variants "$task_run/variants-train.json" --sources "$task_run/sourcebank730" \
  --teachers "$task_root/runs/prefeval-k1-scale730-20260924/teachers/B" \
  --checkpoint "$task_run/robust730/train/checkpoint-final.pt" \
  --output "$task_output/train" > "$task_output/train.log" 2>&1
fi
# Primary V1 endpoint: full train T1 and pilot/dev same-image 3+2, real RGB chains.
for split in dev pilot train; do
 artifact="$task_run/variants-train.json"
 [[ "$split" != dev ]] || artifact="$task_run/variants-dev.json"
 families=T1,T2,T3,O1,O2
 [[ "$split" != train ]] || families=T1
 label="final-$split-V1"
 images="$task_output/$label"
 "$task_python" "$writer" rollout "${common[@]}" --split "$split" \
  --checkpoint "$task_output/train/checkpoint-final.pt" \
  --initial-variants "$artifact" --initial-variant 1 --inter-turns 10 --noise-chains 2 \
  --output "$images" > "$task_output/$label-rollout.log" 2>&1
 "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
  --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" \
  --initial-variants "$artifact" --initial-variant 1 \
  --output "$task_output/readback-$label" --families "$families" --prefixes 0,1,5,10 \
  --noise-chains 2 --controls memory,mismatch,blank,text --tasks mcq \
  > "$task_output/$label-readback.log" 2>&1
 if [[ "$split" != train ]]; then
  for prefix in 0 10; do
   "$task_python" scripts/experiments/prefeval_k1_position_probe.py --kind student --split "$split" \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" --prefix "$prefix" \
    --order-mode official-cyclic --output "$task_output/positions-$label-$prefix.jsonl" \
    > "$task_output/$label-positions-$prefix.log" 2>&1
  done
 fi
done
