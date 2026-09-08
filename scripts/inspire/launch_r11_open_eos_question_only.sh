#!/usr/bin/env bash
set -euo pipefail
OUTPUT=${1:?fresh absolute output directory required}
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256=1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159
export VLM_READER_SNAPSHOT_MANIFEST_SHA256=159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c
PYTHON=/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python
MODELS=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
RUNS=/inspire/ssd/project/exploration-topic/czxs26210936/runs
exec "$PYTHON" -u scripts/experiments/run_r11_open_eos_question_only.py \
  --output-dir "$OUTPUT" \
  --parent-root "$RUNS/vision-language-memory-r11-open-eos/all-e6e6c80-20260908" \
  --old-multistart-root "$RUNS/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02" \
  --dreamlite "$MODELS/DreamLite-mobile" --reader "$MODELS/Qwen3-VL-4B-Instruct"
