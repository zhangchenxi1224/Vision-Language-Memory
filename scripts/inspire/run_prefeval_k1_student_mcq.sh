#!/usr/bin/env bash
# Early T1 readback of the fixed write-stage checkpoint, independent of free generation.
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Set the frozen readback implementation root}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_output="$task_run/student-T1-mcq/$arm"
mkdir -p "$task_output"
exec 9>"$task_output/launcher.lock"
flock -n 9 || { echo 'Student MCQ launcher already active'; exit 1; }
trap 'printf "%s\n" "$?" > "$task_output/exit-status.txt"' EXIT
cd "$task_repo"
git rev-parse HEAD > "$task_output/code-commit.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for split in pilot dev; do
  task_images="$task_run/writer/$arm/write-$split"
  deadline=$((SECONDS+43200))
  until python3 - "$task_images" "$split" <<'PY'
import sys
from pathlib import Path
from scripts.experiments.prefeval_k1_data import load_records
root=Path(sys.argv[1])
ready=all((root/r['base_pair_id'].replace(':','_')/f'seed-{chain}'/'complete.json').exists()
          for r in load_records(sys.argv[2]) for chain in range(2))
sys.exit(0 if ready else 1)
PY
  do
    (( SECONDS < deadline )) || { echo "Timeout waiting for $task_images"; exit 1; }
    sleep 30
  done
  controls=memory,mismatch
  # Pilot blank/text T1 baselines already exist in teacher-T1-mcq/A.
  # Dev baselines are shared between the two arms and generated once here.
  if [[ "$arm" == A && "$split" == dev ]]; then controls=memory,mismatch,blank,text; fi
  "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$task_images" --split "$split" \
    --output "$task_output/write-$split" --prefixes 0 --noise-chains 2 \
    --families T1 --tasks mcq --controls "$controls"
done
