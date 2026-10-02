#!/usr/bin/env bash
# Uses the Reader GPU after teacher readback; never competes with the paired Writer.
set -euo pipefail
arm=${1:?A or B}
[[ "$arm" == A || "$arm" == B ]]
task_root=/inspire/ssd/project/exploration-topic/czxs26210936
task_repo=${K1_CODE_ROOT:?Set the frozen readback implementation root}
task_python="$task_root/envs/vlm-r3-ngc2502/bin/python"
task_models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
task_run="$task_root/runs/prefeval-k1-l0-l2-20260924"
task_arm="$task_run/writer/$arm"
task_readback="$task_run/student-readback/$arm"
mkdir -p "$task_readback"
exec 9>"$task_readback/launcher.lock"
flock -n 9 || { echo 'Student readback launcher already active'; exit 1; }
cd "$task_repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
wait_file() {
  local deadline=$((SECONDS+43200))
  until [[ -f "$1" ]]; do
    (( SECONDS < deadline )) || { echo "Timeout waiting for $1"; exit 1; }
    sleep 30
  done
}
wait_rollout() {
  local directory=$1 split=$2 deadline=$((SECONDS+43200))
  until "$task_python" - "$directory" "$split" <<'PY'
import sys
from pathlib import Path
from scripts.experiments.prefeval_k1_data import load_records
root=Path(sys.argv[1])
ready=all((root/r['base_pair_id'].replace(':','_')/f'seed-{chain}'/'complete.json').exists()
          for r in load_records(sys.argv[2]) for chain in range(2))
sys.exit(0 if ready else 1)
PY
  do
    (( SECONDS < deadline )) || { echo "Timeout waiting for $directory"; exit 1; }
    sleep 30
  done
}
readback() {
  local images=$1 split=$2 output=$3 prefixes=$4 families=$5 controls=$6
  "$task_python" scripts/experiments/prefeval_k1_evaluate.py --kind student \
    --reader "$task_models/Qwen3-VL-4B-Instruct" --images "$images" --split "$split" \
    --output "$output" --prefixes "$prefixes" --families "$families" --controls "$controls"
}
wait_file "$task_run/teacher-readback/$arm/finished-0.json"
controls=memory,mismatch
if [[ "$arm" == A ]]; then controls=memory,mismatch,blank,text; fi
# Internal T1 diagnostics arrive before the full phrasing matrix.
for split in pilot dev; do
  wait_rollout "$task_arm/write-$split" "$split"
  readback "$task_arm/write-$split" "$split" "$task_readback/write-$split" 0 T1 "$controls"
done
for split in pilot dev; do
  wait_rollout "$task_arm/final-$split" "$split"
  readback "$task_arm/final-$split" "$split" "$task_readback/final-$split" 0,5,10 T1 "$controls"
done
# L1 is a registered single-preference overwrite extension, not the official task.
for split in pilot dev; do
  "$task_python" scripts/experiments/prefeval_k1_replacement.py \
    --base "$task_models/DreamLite-base-a9a0f15-20260907" \
    --official-source "$task_root/Vision-Language-Memory/third_party/DreamLite" \
    --checkpoint "$task_arm/retain/checkpoint-final.pt" --sources "$task_arm/final-$split" \
    --output "$task_arm/replacement-$split" --split "$split"
  readback "$task_arm/replacement-$split" "$split" "$task_readback/replacement-$split" 0 T1 memory,mismatch
done
for split in pilot dev; do
  readback "$task_arm/final-$split" "$split" "$task_readback/final-$split" 0,5,10 T2,T3,O1,O2 "$controls"
done
printf '0\n' > "$task_readback/exit-status.txt"
