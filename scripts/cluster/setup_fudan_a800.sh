#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="${VLM_CODE_ROOT:-/remote-home1/cxzhang/codex_runs/vision-language-memory}"
VENV_ROOT="${VLM_VENV_ROOT:-/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118}"
MODEL_ROOT="${VLM_MODEL_ROOT:-/remote-home1/cxzhang/codex_models/vision-language-memory}"
HF_CACHE_ROOT="${HF_HOME:-/remote-home1/cxzhang/codex_models/.hf-cache}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/remote-home1/cxzhang/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

echo "== Fudan A800 environment setup =="
date
hostname
echo "CODE_ROOT=$CODE_ROOT"
echo "VENV_ROOT=$VENV_ROOT"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "HF_ENDPOINT=$HF_ENDPOINT"

case "$CODE_ROOT" in
  /remote-home1/cxzhang/codex_runs/*) ;;
  *) echo "Unsafe CODE_ROOT: $CODE_ROOT" >&2; exit 2 ;;
esac
case "$VENV_ROOT" in
  /remote-home1/cxzhang/codex_envs/*) ;;
  *) echo "Unsafe VENV_ROOT: $VENV_ROOT" >&2; exit 2 ;;
esac
case "$MODEL_ROOT" in
  /remote-home1/cxzhang/codex_models/*) ;;
  *) echo "Unsafe MODEL_ROOT: $MODEL_ROOT" >&2; exit 2 ;;
esac

if [[ ! -d "$CODE_ROOT/.git" ]]; then
  echo "Expected a Git checkout at $CODE_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$VENV_ROOT")" "$MODEL_ROOT" "$HF_HOME" "$PIP_CACHE_DIR"

python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected compute-node Python 3.10, got {sys.version}")
print("bootstrap_python", sys.executable, sys.version)
PY

VENV_MARKER="$VENV_ROOT/.vision_memory_environment"
if [[ -e "$VENV_ROOT" && ! -x "$VENV_ROOT/bin/python" ]]; then
  echo "Refusing to reuse an incomplete environment at $VENV_ROOT; choose a fresh VLM_VENV_ROOT." >&2
  exit 2
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  VENV_PROBE="$(dirname "$VENV_ROOT")/.vision-memory-venv-probe-${SLURM_JOB_ID:-$$}"
  if [[ -e "$VENV_PROBE" ]]; then
    echo "Unexpected venv probe collision: $VENV_PROBE" >&2
    exit 2
  fi
  trap 'rm -rf -- "$VENV_PROBE"' EXIT
  python3 -m venv "$VENV_PROBE"
  "$VENV_PROBE/bin/python" -m pip --version
  rm -rf -- "$VENV_PROBE"
  trap - EXIT

  python3 -m venv "$VENV_ROOT"
  touch "$VENV_MARKER"
elif [[ ! -f "$VENV_MARKER" ]]; then
  echo "Refusing to reuse an environment without the project marker: $VENV_ROOT" >&2
  exit 2
fi

PYTHON="$VENV_ROOT/bin/python"
"$PYTHON" - <<'PY'
import sys
import pip

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Environment must use Python 3.10, got {sys.version}")
print("environment_python", sys.executable, sys.version)
print("environment_pip", pip.__version__)
PY
"$PYTHON" -m pip install --upgrade "pip<26" setuptools wheel
"$PYTHON" -m pip install \
  "torch==2.4.1" \
  "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" - <<'PY'
import torch
import torchvision

assert torch.__version__ == "2.4.1+cu118", torch.__version__
assert torchvision.__version__ == "0.19.1+cu118", torchvision.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
print("torch_profile_ok", torch.__version__, torchvision.__version__, torch.version.cuda)
PY
"$PYTHON" -m pip install -r "$CODE_ROOT/requirements/runtime-pinned.txt"
"$PYTHON" -m pip install -e "$CODE_ROOT" --no-deps

"$PYTHON" "$CODE_ROOT/scripts/bootstrap/fetch_sources.py"
"$PYTHON" "$CODE_ROOT/scripts/bootstrap/fetch_models.py" --model-root "$MODEL_ROOT"

"$PYTHON" "$CODE_ROOT/scripts/bootstrap/preflight.py" \
  --mode local \
  --model-root "$MODEL_ROOT" \
  --output "$CODE_ROOT/runs/preflight-setup-local.json"
"$PYTHON" -m unittest discover -s "$CODE_ROOT/tests" -v

"$PYTHON" - <<'PY'
import importlib.metadata
import json
import sys

import torch

print(
    json.dumps(
        {
            "python": sys.version,
            "torch_distribution": importlib.metadata.version("torch"),
            "torch_runtime": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_during_cpu_setup": torch.cuda.is_available(),
        },
        indent=2,
    )
)
PY

echo "SETUP_COMPLETE"
