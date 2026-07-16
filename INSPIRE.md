# Inspire R3 runtime

This file records non-secret project context only. Account configuration, proxy
tokens, Hugging Face tokens, and `.inspire/config.toml` must remain outside the
repository.

## Default image and existing notebook

- Notebook: `vlm-r3-h200x2-live-20260717`
- Workspace: `分布式训练空间`
- Project: `前沿课题探索`
- Node reported by the platform: `qb-prod-gpu2007`
- Image: `ngc-pytorch:25.02-cuda12.8.0-py3`
- Allocation: one node, 2 H200 GPUs, 40 CPUs, 400 GiB RAM, 128 GiB shared memory

The R3 runtime uses the NGC image's system PyTorch. It creates a Python 3.12
overlay venv with `--system-site-packages`; pip is never allowed to resolve or
install `torch`, `torchvision`, or `torchaudio`. The observed image contract is
PyTorch `2.7.0a0+ecf3bae40a.nv25.02`, CUDA runtime `12.8`, and exactly two H200
devices with at least 140000 MiB each. A change to any of these values requires
a new prospective runtime lock and new technical-gate evidence.

The old Fudan Slurm scripts are retained only to reproduce historical R1/R2
artifacts. They are not an execution path for R3 and must not be submitted.

## Path conventions

Run `inspire notebook path list` to discover the actual project-personal
directory. Do not infer it from the container's `root` user. Recommended layout:

```text
ssd.me:Vision-Language-Memory/             clean fixed-commit checkout
ssd.me:envs/vlm-r3-ngc2502/                overlay venv
ssd.me:runs/vision-language-memory-r3/      logs, sentinels, manifests, checkpoints
qb-ilm.me:models/vision-language-memory/    immutable model snapshots
qb-ilm.me:cache/huggingface/                Hugging Face cache
qb-ilm.me:cache/torch/                      Torch cache
```

The concrete absolute paths are exported from a private copy of
`configs/inspire.env.example`. Model and run directories must be project-scoped,
absolute, and writable. Formal outputs never go to the container overlay.

## Code and environment migration

Source synchronization uses a clean Git commit, not recursive `notebook scp`.
If the GPU notebook has no public egress, clone/fetch on a reachable CPU
notebook and use the same project filesystem, or start an Inspire egress bridge.
Always checkout the exact prospective R3 commit and verify a clean worktree.

Inside the existing H200 notebook:

```bash
cd /absolute/project/path/Vision-Language-Memory
git checkout --detach <FULL_40_CHARACTER_R3_COMMIT>
test -z "$(git status --porcelain=v1 --untracked-files=all)"

cp configs/inspire.env.example /absolute/private/path/vlm-r3.env
# Replace PROJECT_USER paths in the private copy, then:
source /absolute/private/path/vlm-r3.env
export VLM_EXPECTED_COMMIT=<FULL_40_CHARACTER_R3_COMMIT>
bash scripts/inspire/bootstrap_r3_h200.sh
```

The bootstrap is idempotent for one valid overlay venv. It installs
`requirements/inspire-ngc2502-pinned.txt` with `--no-deps`, verifies imports,
checks that the Torch file and version are byte-for-byte the original NGC
fingerprint, fetches the locked DreamLite source, runs focused unit tests, and
writes:

```text
$VLM_RUN_ROOT/bootstrap/<commit>/environment.json
$VLM_RUN_ROOT/bootstrap/<commit>/infrastructure_preflight.json
$VLM_RUN_ROOT/bootstrap/<commit>/infrastructure_preflight.json.sha256
```

It does not fetch model weights and does not start a GPU experiment. If an
import is missing, update the non-Torch lock prospectively; never fix it by
running an unconstrained pip command in the formal venv.

## Long infrastructure tasks

Any operation that may exceed 20 minutes is launched detached and audited. For
example, model snapshot reconstruction uses the infrastructure-only bypass:

```bash
COMMIT=<FULL_40_CHARACTER_R3_COMMIT>
BOOT=$VLM_RUN_ROOT/bootstrap/$COMMIT/infrastructure_preflight.json
FETCH_RUN=$VLM_RUN_ROOT/infrastructure/model-fetch-$COMMIT

$VLM_VENV_ROOT/bin/python scripts/inspire/launch_background.py \
  --run-root "$VLM_RUN_ROOT" \
  --run-dir "$FETCH_RUN" \
  --stage model-fetch \
  --expected-commit "$COMMIT" \
  --preflight "$BOOT" \
  --infrastructure-stage -- \
  "$VLM_VENV_ROOT/bin/python" scripts/bootstrap/fetch_models.py \
  --model-root "$VLM_MODEL_ROOT"
```

The launcher returns immediately. It records the command and immutable input
hash, forces the R3 deterministic environment, writes `running.json`, streams
to `stdout.log`/`stderr.log`, and atomically creates `terminal.json`. Polling is
read-only:

```bash
$VLM_VENV_ROOT/bin/python scripts/inspire/poll_stage.py "$FETCH_RUN"
```

Exit code `3` means still running, `0` means a successful terminal sentinel,
and `2` means failure, orphaning, or missing evidence. Never pass credentials on
a recorded command line.

## Formal preflight and scientific stages

After both locked model snapshots are complete, create a model-complete formal
preflight:

```bash
FORMAL=$VLM_RUN_ROOT/preflight/$COMMIT/r3_h200_formal.json
mkdir -p "$(dirname "$FORMAL")"
$VLM_VENV_ROOT/bin/python scripts/inspire/preflight_r3_h200.py \
  --repo "$PWD" \
  --model-root "$VLM_MODEL_ROOT" \
  --expected-commit "$COMMIT" \
  --require-models \
  --output "$FORMAL"
```

Scientific background stages omit `--infrastructure-stage`; the launcher then
rejects any preflight without `formal_ready=true`. It also rejects a changed
commit, dirty worktree, preflight SHA mismatch, reused run directory, non-H200
runtime, one-GPU visibility, incomplete model snapshot, or venv-local Torch.

R3-S0, G4-L, G5-L, G6-L, and DL-S must be run serially and fail closed on this
new H200 runtime. Historical A800 results do not satisfy these gates. No
DreamLite pilot is launched until all required technical and micro gates pass.

## Ongoing jobs

The existing notebook is an interactive migration/debug instance. This runtime
layer does not create, stop, delete, or save that notebook, and it does not
submit an Inspire job. Each actual background stage is represented solely by
its run directory and terminal sentinel under `VLM_RUN_ROOT`.
