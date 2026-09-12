#!/usr/bin/env bash
# Run only on a fresh idle H200 after the fixed parent endpoint has completed.
# Argument 1 is the explicit platform lease deadline; argument 2, if needed,
# must be --diagnostic to preserve a failed development gate as a diagnostic.
set -euo pipefail
deadline=${1:?Pass a Unix deadline within the actual notebook lease}
diagnostic=()
if [ "${2:-}" = --diagnostic ]; then
    diagnostic=(--diagnostic)
elif [ "$#" -gt 1 ]; then
    exit 2
fi
root=/inspire/ssd/project/exploration-topic/czxs26210936
models=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
runs=$root/runs/dreamlite-official-alignment
python=$root/envs/vlm-r3-ngc2502/bin/python
probe=$root/repos/dreamlite-transition-validation-20260913
collector=$root/repos/dreamlite-result-verification-20260913
package_repo=$root/repos/dreamlite-package-parity-20260913
parent=$runs/9628d71-transition-wording-full2880-20260913
bank=$runs/9628d71-transition-wording-bank/manifest.json
single=$runs/0f40767-transition-confirmation
chains=$runs/0f40767-transition-chains
package=$runs/281b653-transition-writer-package
prepared=$runs/281b653-transition-package-parity
inference=$runs/281b653-transition-package-inference
status=$runs/transition-validation-suite-status.json

# No training restart, output reuse, or checkpoint selection occurs here.
for path in "$single" "$chains" "$package" "$prepared" "$inference" "$status"; do
    if [ -e "$path" ]; then
        printf 'Refuse existing output: %s\n' "$path" >&2
        exit 2
    fi
done
for item in \
    "$probe:0f4076788bf4125bbca8851b6b270d5f8418d534" \
    "$collector:3c335a284f430d924ff27178341dd1de547e12b3" \
    "$package_repo:281b65360bd42e4822f2d5750b541d85661a97dc"; do
    directory=${item%:*}
    expected=${item##*:}
    test "$(git -C "$directory" rev-parse HEAD)" = "$expected"
    test -z "$(git -C "$directory" status --porcelain)"
done
test -f "$parent/terminal.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage=preflight
save_status() {
    "$python" - "$status" "$stage" "$1" "$deadline" <<'PY'
import json
from pathlib import Path
import sys
import time
path, stage, state, deadline = sys.argv[1:]
temporary = Path(path + '.tmp')
temporary.write_text(json.dumps({'stage': stage, 'state': state, 'updated_unix': time.time(),
    'deadline_unix': float(deadline), 'functional_success_requires_raw_review': True}, indent=2) + '\n')
temporary.replace(path)
PY
}
finish() {
    code=$?
    if [ "$code" -ne 0 ]; then
        save_status failed
    fi
    exit "$code"
}
trap finish EXIT
"$python" - "$deadline" "$parent/terminal.json" <<'PY'
import json
from pathlib import Path
import sys
import time
if float(sys.argv[1]) - time.time() < 5400:
    raise RuntimeError('Require at least90min within a fresh idle notebook lease; do not shrink validation')
if json.loads(Path(sys.argv[2]).read_text()).get('state') != 'completed':
    raise RuntimeError('Parent endpoint is not complete')
PY

stage=single_writes
save_status running
"$python" -u "$probe/scripts/probes/official_transition_confirmation.py" \
    --parent-run "$parent" --output "$single" --mode single_writes \
    --deadline-unix "$deadline" "${diagnostic[@]}" > "$single.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" "$collector/scripts/reporting/collect_transition_validation.py" \
    --run "$single" --parent "$parent" --bank "$bank" \
    --plan "$probe/reports/official-transition-validation-plan-20260913.json" \
    --output-prefix "$runs/transition-confirmation" > "$single.collection.log" 2>&1

stage=rgb_chains
save_status running
"$python" -u "$probe/scripts/probes/official_transition_confirmation.py" \
    --parent-run "$parent" --output "$chains" --mode rgb_chains \
    --deadline-unix "$deadline" "${diagnostic[@]}" > "$chains.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" "$collector/scripts/reporting/collect_transition_validation.py" \
    --run "$chains" --parent "$parent" --bank "$bank" \
    --plan "$probe/reports/official-transition-validation-plan-20260913.json" \
    --output-prefix "$runs/transition-chains" > "$chains.collection.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" "$collector/scripts/reporting/verify_rgb_chain_tensors.py" \
    --run "$chains" --output "$runs/transition-chains-tensor-verification.json" \
    > "$chains.tensors.log" 2>&1

stage=export_and_prepare
save_status running
CUDA_VISIBLE_DEVICES='' "$python" "$package_repo/scripts/inference/export_rgb_writer.py" \
    --parent-run "$parent" --output "$package" > "$package.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" "$package_repo/scripts/probes/rgb_package_parity.py" prepare \
    --reference "$chains" --package "$package" --output "$prepared" > "$prepared.log" 2>&1

stage=independent_inference
save_status running
# This separate process receives events/seeds/queries, package and model files;
# it receives no teacher bank, reference outputs, gold answers or optimizer state.
remaining=$("$python" - "$deadline" <<'PY'
import sys
import time
remaining = int(float(sys.argv[1]) - time.time())
if remaining <= 0:
    raise RuntimeError('Validation deadline reached before independent inference')
print(remaining)
PY
)
timeout --signal=TERM --kill-after=30 "$remaining" "$python" -u "$package_repo/scripts/inference/rgb_memory.py" \
    --package "$package" --base-model "$models/DreamLite-base-a9a0f15-20260907" \
    --official-source "$root/Vision-Language-Memory/third_party/DreamLite" \
    --reader-model "$models/Qwen3-VL-4B-Instruct" \
    --commands "$prepared/commands.jsonl" --output "$inference" > "$inference.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$python" "$package_repo/scripts/probes/rgb_package_parity.py" verify \
    --prepared "$prepared" --inference "$inference" > "$prepared.verification.log" 2>&1
stage=all_registered_workloads_finished
save_status completed
