#!/usr/bin/env bash
set -euo pipefail
# Run inside the immutable checkout on the allocated H200 worker. No resource creation.
# Usage: bash scripts/inspire/launch_r11_open_eos.sh replay /inspire/ssd/.../new-output [extra runner arguments]
MODE=${1:?mode required: audit/replay/train/all}
OUTPUT=${2:?fresh absolute output directory required}
shift 2
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256=1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159
export VLM_READER_SNAPSHOT_MANIFEST_SHA256=159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c
PYTHON=/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python
MODEL_ROOT=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
exec "$PYTHON" -u scripts/experiments/run_r11_open_eos_paired.py \
  --mode "$MODE" --output-dir "$OUTPUT" \
  --old-multistart-root /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02 \
  --dreamlite "$MODEL_ROOT/DreamLite-mobile" --reader "$MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --vae-device cuda:0 --reader-device cuda:1 "$@"
