#!/usr/bin/env bash
# Fixed T1 MCQ readback as soon as each full final split has finished generation.
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Frozen evaluator root}
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_output="$task_run/student-final-T1-mcq/$arm"
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || { echo 'Final MCQ readback already active'; exit 1; }
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
git rev-parse HEAD > "$task_output/code-commit.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for split in pilot dev; do
  task_images="$task_run/writer/$arm/final-$split"
  deadline=$((SECONDS+86400))
  until python3 - "$task_images" "$split" <<'PY'
import sys
from pathlib import Path
from scripts.experiments.prefeval_k1_data import load_records
root=Path(sys.argv[1])
sys.exit(0 if all((root/r['base_pair_id'].replace(':','_')/f'seed-{c}'/'complete.json').exists()
                  for r in load_records(sys.argv[2]) for c in [0,1]) else 1)
PY
  do
    (( SECONDS < deadline )) || { echo "Timeout waiting for complete $split RGB chains"; exit 1; }
    sleep 30
  done
  "$task_root/envs/vlm-r3-ngc2502/bin/python" scripts/experiments/prefeval_k1_evaluate.py \
    --kind student --split "$split" \
    --reader /inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct \
    --images "$task_images" --output "$task_output/$split" \
    --prefixes 0,5,10 --noise-chains 2 --families T1 --tasks mcq --controls memory,mismatch
done
