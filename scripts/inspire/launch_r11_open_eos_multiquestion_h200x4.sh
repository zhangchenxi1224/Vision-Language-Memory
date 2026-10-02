#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
project_root=/inspire/ssd/project/exploration-topic/czxs26210936
export PYTHONPATH="$repo_dir/src:$repo_dir"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256=1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159
export VLM_READER_SNAPSHOT_MANIFEST_SHA256=159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c
exec "$project_root/envs/vlm-r3-ngc2502/bin/python" -u \
  scripts/inspire/run_r11_open_eos_multiquestion_h200x4.py \
  --output-root "${VLM_OPEN_MULTI_OUTPUT:?fresh output root required}" \
  --expected-commit "${VLM_OPEN_MULTI_COMMIT:?exact source commit required}" \
  --max-hours 2.5
