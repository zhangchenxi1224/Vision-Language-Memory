#!/usr/bin/env bash
set -euo pipefail
task_code=$(cd "$(dirname "$0")/../.." && pwd)
task_root=/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927
task_python=/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python
exec 9>"$task_root/pipeline.lock"
flock -n 9 || { echo 'Existing multitarget pipeline is active; no evaluation started'; exit 1; }
cd "$task_code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_python" scripts/inspire/run_prefeval_c8_eval730.py --output "$task_root/eval730/fixed-c8-b0-20260928"
