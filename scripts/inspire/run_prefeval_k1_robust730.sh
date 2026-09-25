#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_run="$task_root/runs/prefeval-b-mcq-20260925"
task_output="$task_run/robust730"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[[ -f "$task_run/variants-train.json" && -f "$task_run/variants-dev.json" ]]
common=(--arm B --base "$task_models/DreamLite-base-a9a0f15-20260907"
 --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
writer=scripts/experiments/prefeval_k1_writer.py
if [[ ! -f "$task_output/train/complete.json" ]]; then
 "$task_python" "$writer" train "${common[@]}" --split train --stage write \
  --steps 23360 --snapshot-steps 2048 --initial-variants "$task_run/variants-train.json" \
  --teachers "$task_root/runs/prefeval-k1-scale730-20260924/teachers/B" \
  --checkpoint "$task_root/runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt" \
  --output "$task_output/train" > "$task_output/train.log" 2>&1
fi
# Final endpoint first. OOD/V2 cannot select a checkpoint or alter this fixed budget.
for stage in final step-002048; do
 for variant in 1 0 2; do
  [[ "$stage" == final || "$variant" == 1 ]] || continue
  for split in dev pilot train; do
   [[ "$split" != train || ( "$stage" == final && "$variant" == 1 ) ]] || continue
   artifact="$task_run/variants-train.json"
   [[ "$split" != dev ]] || artifact="$task_run/variants-dev.json"
   label="$stage-$split-V$variant"
   images="$task_output/$label"
   families=T1,T2,T3,O1,O2
   [[ "$split" != train && "$stage" == final ]] || families=T1
   "$task_python" "$writer" rollout "${common[@]}" \
    --checkpoint "$task_output/train/checkpoint-$stage.pt" --split "$split" \
    --initial-variants "$artifact" --initial-variant "$variant" --inter-turns 0 --noise-chains 2 \
    --output "$images" > "$task_output/$label-rollout.log" 2>&1
   "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" \
    --initial-variants "$artifact" --initial-variant "$variant" \
    --output "$task_output/readback-$label" --families "$families" --prefixes 0 --noise-chains 2 \
    --controls memory,mismatch,blank,text --tasks mcq > "$task_output/$label-readback.log" 2>&1
   if [[ "$stage" == final && "$variant" == 1 && "$split" != train ]]; then
    "$task_python" scripts/experiments/prefeval_k1_position_probe.py --kind student --split "$split" \
     --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" --prefix 0 \
     --order-mode official-cyclic --output "$task_output/positions-$label.jsonl" \
     > "$task_output/$label-positions.log" 2>&1
   fi
  done
 done
done
