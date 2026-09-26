#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_run="$task_root/runs/prefeval-b730-exposure512-20260927"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
mkdir -p "$task_run"
if [[ ${1:-} != --foreground ]]; then
  nohup bash "$0" --foreground > "$task_run/launch.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$task_run/launcher.pid"
  cat "$task_run/launcher.pid"
  exit 0
fi
exec 9>"$task_run/launcher.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_run/launcher-exit-status.txt"' EXIT
cd "$task_repo"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$task_python" -m pytest tests/test_prefeval_k1_budget_extension.py -q
"$task_python" scripts/inspire/preflight_prefeval_b730_exposure512.py
"$task_python" scripts/inspire/run_prefeval_b730_exposure512.py
