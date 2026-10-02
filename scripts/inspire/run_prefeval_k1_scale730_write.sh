#!/usr/bin/env bash
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen scale730 code root}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-scale730-20260924"
cd "$task_repo"
controls=memory,mismatch
[[ "$arm" != A ]] || controls=memory,mismatch,blank,text
"$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind teacher --split train \
  --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_run/teachers/$arm" \
  --output "$task_run/teacher-T1-mcq/$arm" --families T1 --tasks mcq --controls "$controls"
common=(--arm "$arm" --base "$task_models/DreamLite-base-a9a0f15-20260907"
  --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite")
writer=scripts/experiments/prefeval_k1_writer.py
trained="$task_run/writer/$arm/write"
"$task_python" "$writer" train "${common[@]}" --split train --stage write \
  --teachers "$task_run/teachers/$arm" --steps 23360 --snapshot-steps 2048 \
  --checkpoint "$task_root/runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt" \
  --output "$trained"
# Both checkpoints are fixed prospectively; neither is selected from OOD scores.
for stage in step-002048 final; do
  checkpoint="$trained/checkpoint-$stage.pt"
  for split in pilot dev; do
    images="$task_run/writer/$arm/$stage-$split"
    "$task_python" "$writer" rollout "${common[@]}" --checkpoint "$checkpoint" \
      --split "$split" --inter-turns 0 --noise-chains 2 --output "$images"
    "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student --split "$split" \
      --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" \
      --output "$task_run/student-T1-mcq/$arm/$stage-$split" --families T1 --tasks mcq \
      --prefixes 0 --noise-chains 2 --controls memory,mismatch
  done
done
