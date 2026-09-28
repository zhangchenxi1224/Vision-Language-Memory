#!/usr/bin/env bash
set -euo pipefail
task_role=${1:?Specify primary or reader}
case "$task_role" in
  primary) task_prefix= ;;
  reader) task_prefix=reader- ;;
  *) exit 2 ;;
esac
task_scheduler=$(cd -- "$(dirname -- "$0")" && pwd)
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_run="$task_root/runs/prefeval-b730-exposure512-20260927"
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
if [[ ${2:-} != --foreground ]]; then
  nohup bash "$0" "$task_role" --foreground > "$task_run/${task_prefix}launch.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$task_run/${task_prefix}launcher.pid"
  cat "$task_run/${task_prefix}launcher.pid"
  exit 0
fi
exec 9>"$task_run/${task_prefix}launcher.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$task_run/${task_prefix}launcher-exit-status.txt"' EXIT
cd "$task_run/code"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ $task_role == primary ]]; then
  "$task_python" -m pytest tests/test_prefeval_k1_budget_extension.py -q
  "$task_python" scripts/inspire/preflight_prefeval_b730_exposure512.py
fi
"$task_python" "$task_scheduler/run_prefeval_b730_exposure512_distributed.py" "$task_role"
