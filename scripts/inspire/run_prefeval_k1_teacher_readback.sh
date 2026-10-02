#!/usr/bin/env bash
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:-$task_root/repos/prefeval-k1-l0-l2-20260924}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
controls=memory,mismatch
# Baselines do not depend on arm; calculate once on identical PNG/readout config.
if [[ "$arm" == A ]]; then controls=memory,mismatch,blank,text; fi
"$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind teacher \
  --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_run/pilot/$arm" \
  --controls "$controls" --output "$task_run/teacher-readback/$arm"
