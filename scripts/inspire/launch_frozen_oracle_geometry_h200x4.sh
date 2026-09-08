#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
project_root=/inspire/ssd/project/exploration-topic/czxs26210936
model_root=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
runtime_python="$project_root/envs/vlm-r3-ngc2502/bin/python"
export PYTHONPATH="$repo_dir/src:$repo_dir"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256="$(sha256sum "$model_root/DreamLite-mobile/.snapshot_manifest.json" | cut -d ' ' -f1)"
export VLM_READER_SNAPSHOT_MANIFEST_SHA256="$(sha256sum "$model_root/Qwen3-VL-4B-Instruct/.snapshot_manifest.json" | cut -d ' ' -f1)"
expected_commit="${VLM_GEOMETRY_COMMIT:?exact source commit required}"
output_root="${VLM_GEOMETRY_OUTPUT:?fresh project SSD output root required}"
mkdir -p "$output_root"
"$runtime_python" -u scripts/inspire/preflight_frozen_oracle_geometry_h200x4.py --output "$output_root/runtime_preflight.json"
extra_args=()
if [[ -n "${VLM_OPEN_EOS_AUX_SPEC:-}" ]]; then
  extra_args+=(--aux-command-json "$VLM_OPEN_EOS_AUX_SPEC")
fi
exec "$runtime_python" -u scripts/experiments/run_frozen_oracle_geometry_campaign.py \
  --config "$repo_dir/configs/experiments/frozen_oracle_geometry.json" \
  --manifest "$repo_dir/configs/experiments/frozen_oracle_geometry_manifest.json" \
  --train "$project_root/data/vision-language-memory-r3/formal-v1/train.jsonl" \
  --dev "$project_root/data/vision-language-memory-r3/formal-v1/dev.jsonl" \
  --dreamlite "$model_root/DreamLite-mobile" --reader "$model_root/Qwen3-VL-4B-Instruct" \
  --output-root "$output_root" --expected-commit "$expected_commit" --max-hours 72 "${extra_args[@]}"
