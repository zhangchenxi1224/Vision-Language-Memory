#!/usr/bin/env bash
set -euo pipefail
task_code=$(cd "$(dirname "$0")/../.." && pwd)
task_root=/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927
task_python=/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python
mkdir -p "$task_root"
exec 9>"$task_root/pipeline.lock"
flock -n 9 || { echo 'Existing multitarget pipeline is active'; exit 1; }
cd "$task_code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_python" scripts/inspire/run_prefeval_multitarget.py --output "$task_root/smoke" --smoke --targets 2
"$task_python" scripts/inspire/run_prefeval_multitarget.py --output "$task_root/pilot64"
