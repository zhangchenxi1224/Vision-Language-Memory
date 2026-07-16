#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE_PYTHON="${VLM_BASE_PYTHON:-python3}"
VENV_ROOT="${VLM_VENV_ROOT:?Set VLM_VENV_ROOT to an absolute Inspire project path}"
MODEL_ROOT="${VLM_MODEL_ROOT:?Set VLM_MODEL_ROOT to an absolute Inspire project path}"
RUN_ROOT="${VLM_RUN_ROOT:?Set VLM_RUN_ROOT to an absolute Inspire project path}"
HF_ROOT="${HF_HOME:?Set HF_HOME to an absolute Inspire project path}"
TORCH_CACHE="${TORCH_HOME:?Set TORCH_HOME to an absolute Inspire project path}"
EXPECTED_COMMIT="${VLM_EXPECTED_COMMIT:?Set VLM_EXPECTED_COMMIT to the clean 40-character R3 commit}"
PIP_INDEX_URL_VALUE="${VLM_PIP_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple/}"
PIP_TRUSTED_HOST_VALUE="${VLM_PIP_TRUSTED_HOST:-nexus.sii.shaipower.online}"
FETCH_SOURCE=1
RUN_TESTS=1

usage() {
  cat <<'EOF'
Usage: bootstrap_r3_h200.sh [--no-fetch-source] [--no-tests]

Creates an overlay venv with --system-site-packages, installs only non-Torch
dependencies with --no-deps, proves that NGC Torch was not overlaid, and writes
an infrastructure preflight manifest. It never downloads model weights and
never starts a scientific GPU stage.
EOF
}

while (($#)); do
  case "$1" in
    --no-fetch-source) FETCH_SOURCE=0 ;;
    --no-tests) RUN_TESTS=0 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for path in "$VENV_ROOT" "$MODEL_ROOT" "$RUN_ROOT" "$HF_ROOT" "$TORCH_CACHE"; do
  [[ "$path" = /* ]] || { echo "Inspire runtime paths must be absolute: $path" >&2; exit 2; }
  [[ "$path" != *PROJECT_USER* ]] || { echo "Replace PROJECT_USER in configs/inspire.env.example first" >&2; exit 2; }
done
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "VLM_EXPECTED_COMMIT must be a full commit" >&2; exit 2; }

for variable in VLM_INSPIRE_INSTANCE VLM_INSPIRE_WORKSPACE VLM_INSPIRE_PROJECT VLM_INSPIRE_IMAGE VLM_INSPIRE_NODE; do
  [[ -n "${!variable:-}" ]] || { echo "Missing required non-secret environment variable: $variable" >&2; exit 2; }
done

actual_commit="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_commit" = "$EXPECTED_COMMIT" ]] || {
  echo "Commit mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
}
[[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "Refusing to bootstrap from a dirty checkout" >&2
  exit 2
}

mkdir -p "$MODEL_ROOT" "$RUN_ROOT" "$HF_ROOT" "$TORCH_CACHE" "$(dirname "$VENV_ROOT")"

torch_fingerprint() {
  "$1" - <<'PY'
import importlib.metadata
import json
from pathlib import Path
import torch

print(json.dumps({
    "runtime_version": torch.__version__,
    "distribution_version": importlib.metadata.version("torch"),
    "cuda_runtime": torch.version.cuda,
    "file": str(Path(torch.__file__).resolve()),
}, sort_keys=True, separators=(",", ":")))
PY
}

base_fingerprint="$(torch_fingerprint "$BASE_PYTHON")"
"$BASE_PYTHON" - <<'PY'
import os
import sys

expected = os.environ.get("VLM_EXPECTED_PYTHON", "3.12")
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual != expected:
    raise SystemExit(f"Expected NGC Python {expected}, got {actual}")
PY

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  [[ ! -e "$VENV_ROOT" ]] || { echo "Refusing to replace invalid venv path: $VENV_ROOT" >&2; exit 2; }
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_ROOT"
fi

grep -Eq '^include-system-site-packages = true$' "$VENV_ROOT/pyvenv.cfg" || {
  echo "Existing venv does not preserve NGC system site packages: $VENV_ROOT" >&2
  exit 2
}
PYTHON="$VENV_ROOT/bin/python"

before_install="$(torch_fingerprint "$PYTHON")"
[[ "$before_install" = "$base_fingerprint" ]] || {
  echo "Overlay venv does not resolve the original NGC Torch" >&2
  exit 2
}

"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --index-url "$PIP_INDEX_URL_VALUE" \
  --trusted-host "$PIP_TRUSTED_HOST_VALUE" \
  -r "$ROOT/requirements/inspire-ngc2502-pinned.txt"
"$PYTHON" -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e "$ROOT"
"$PYTHON" "$ROOT/scripts/inspire/install_non_torch_dependencies.py" \
  --index-url "$PIP_INDEX_URL_VALUE" \
  --trusted-host "$PIP_TRUSTED_HOST_VALUE" \
  --output "$VENV_ROOT/non_torch_dependency_report.json" >/dev/null

after_install="$(torch_fingerprint "$PYTHON")"
[[ "$after_install" = "$base_fingerprint" ]] || {
  echo "Torch fingerprint changed during non-Torch dependency installation" >&2
  exit 2
}

"$PYTHON" - <<'PY'
import importlib
import importlib.metadata
import json
import os
import sysconfig
from pathlib import Path

expected = {
    "accelerate": "1.12.0",
    "diffusers": "0.39.0",
    "einops": "0.8.1",
    "huggingface-hub": "0.36.0",
    "numpy": "1.26.4",
    "peft": "0.18.1",
    "Pillow": "12.3.0",
    "pytest": "9.1.1",
    "PyYAML": "6.0.2",
    "ruff": "0.14.10",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.1",
    "tqdm": "4.67.1",
    "transformers": "4.57.3",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Pinned package mismatch: expected={expected!r}, actual={actual!r}")
for module in ("accelerate", "diffusers", "peft", "safetensors", "tokenizers", "transformers"):
    importlib.import_module(module)

purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
for pattern in ("torch", "torch-*.dist-info", "torch*.egg-info"):
    matches = list(purelib.glob(pattern))
    if matches:
        raise SystemExit(f"NGC Torch was overlaid inside the venv: {matches}")
print(json.dumps({"packages": actual, "venv_purelib": str(purelib)}, sort_keys=True))
PY

if ((FETCH_SOURCE)); then
  "$PYTHON" "$ROOT/scripts/bootstrap/fetch_sources.py"
fi

if ((RUN_TESTS)); then
  "$PYTHON" -m pytest -q \
    "$ROOT/tests/test_differentiable_mobile.py" \
    "$ROOT/tests/test_qwen_scorer_contract.py" \
    "$ROOT/tests/test_inspire_runtime.py"
fi

AUDIT_ROOT="$RUN_ROOT/bootstrap/$EXPECTED_COMMIT"
mkdir -p "$AUDIT_ROOT"
"$PYTHON" "$ROOT/scripts/bootstrap/freeze_environment.py" --output "$AUDIT_ROOT/environment.json" >/dev/null
cp "$VENV_ROOT/non_torch_dependency_report.json" "$AUDIT_ROOT/non_torch_dependency_report.json"
"$PYTHON" "$ROOT/scripts/inspire/preflight_r3_h200.py" \
  --repo "$ROOT" \
  --model-root "$MODEL_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output "$AUDIT_ROOT/infrastructure_preflight.json" >/dev/null

printf '{"status":"ready","venv":"%s","preflight":"%s"}\n' \
  "$VENV_ROOT" "$AUDIT_ROOT/infrastructure_preflight.json"
