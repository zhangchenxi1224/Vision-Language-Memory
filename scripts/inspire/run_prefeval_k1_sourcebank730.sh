#!/usr/bin/env bash
set -euo pipefail
task_variant=${1:?Expected source variant 0 or 1}
[[ "$task_variant" == 0 || "$task_variant" == 1 ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen code required}
task_run="$task_root/runs/prefeval-b-mcq-20260925"
task_output="$task_run/sourcebank730"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
mkdir -p "$task_output"
exec 9>"$task_output/V$task_variant.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_output/V$task_variant-exit-status.txt"' EXIT
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Do not reserve a GPU while waiting for the final parent. Dispatch after completion.
[[ -f "$task_run/robust730/train/complete.json" ]]
"$task_python" -c 'import json,sys; from pathlib import Path; from scripts.experiments.prefeval_k1_data import sha; p=Path(sys.argv[1]); d=json.loads((p.parent/"complete.json").read_text()); assert d=={"steps":23360,"checkpoint_sha256":sha(p)}' \
 "$task_run/robust730/train/checkpoint-final.pt"
"$task_python" scripts/experiments/prefeval_k1_writer.py rollout --arm B --split train \
 --base "$task_models/DreamLite-base-a9a0f15-20260907" \
 --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite" \
 --checkpoint "$task_run/robust730/train/checkpoint-final.pt" \
 --initial-variants "$task_run/variants-train.json" --initial-variant "$task_variant" \
 --inter-turns 0 --noise-chains 2 --noise-domain source-bank \
 --output "$task_output/V$task_variant" > "$task_output/V$task_variant.log" 2>&1
# After both V0/V1 lanes finish, the coordinator freezes bank.json with
# prefeval_k1_source_bank.py. No training is allowed from a partial bank.
