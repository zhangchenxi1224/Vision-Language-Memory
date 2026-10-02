#!/usr/bin/env bash
# Read the write-only Writer's real RGB chains before comparing retain-trained endpoints.
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen readback implementation root}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_images="$task_run/writer/$arm/training-prefixes"
task_output="$task_run/student-preretain-T1-mcq/$arm"
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || { echo 'Pre-retain readback already active'; exit 1; }
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
git rev-parse HEAD > "$task_output/code-commit.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
deadline=$((SECONDS+43200))
until [[ -f "$task_run/student-T1-mcq/$arm/write-dev/finished-0.json" ]]; do
  (( SECONDS < deadline )) || { echo 'Timeout waiting for dev readback'; exit 1; }
  sleep 30
done
until python3 - "$task_images" <<'PY'
import sys
from pathlib import Path
from scripts.experiments.prefeval_k1_data import load_records
root=Path(sys.argv[1])
sys.exit(0 if all((root/r['base_pair_id'].replace(':','_')/'seed-0'/'complete.json').exists()
                  for r in load_records('pilot')) else 1)
PY
do
  (( SECONDS < deadline )) || { echo 'Timeout waiting for real RGB prefixes'; exit 1; }
  sleep 30
done
"$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student \
  --reader /inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct \
  --images "$task_images" --split pilot --output "$task_output" \
  --prefixes 0,5,10 --noise-chains 1 --families T1 --tasks mcq --controls memory,mismatch
