# Vision Learnable Memory

This repository is the reproducible engineering shell for the DreamLite + Qwen3-VL
stateful-memory experiments. The laptop is for framework development, mock autograd
tests, and API validation. Real model probes and episode training run on a Linux GPU
cluster.

## Canonical baseline

- Runtime pipeline: Diffusers 0.39.0 `DreamLiteMobilePipeline`.
- Read-only reference: ByteVisionLab/DreamLite at the exact commit in `models.lock.json`.
- Updater weights: `carlofkl/DreamLite-mobile` at an immutable HF revision.
- Frozen Reader: `Qwen/Qwen3-VL-4B-Instruct` at an immutable HF revision.
- Transformers: 4.57.3; initial attention backend: PyTorch SDPA.

Qwen source is not cloned because Qwen3-VL is supplied by Transformers. Model weights
are reconstructed from `models.lock.json`, loaded with `local_files_only=True`, and never
committed to Git.

## Exact technical scope

The official mobile pipeline is kept untouched as the inference and numerical reference.
Its public call is inference-oriented: it is under `torch.no_grad()`, creates target noise,
post-processes outputs, and runs model-hook cleanup. The training wrapper instead accepts
explicit source and noise latents, executes exactly four differentiable U-Net/scheduler
steps, and returns the model-space latent.

The first recurrent milestone is deliberately **latent-path BPTT with stop-gradient
conditioning**:

```text
z_previous ----------------> U-Net spatial condition ----------------> z_next
     |
     +-- detach -> VAE decode -> internal Qwen3-VL-2B -> prompt embeddings
```

This is not full gradient flow through DreamLite's internal Qwen condition branch. It also
defines a persistent latent-state algorithm: the next event consumes the previous output
latent directly. That differs from repeated use of the public editor, which would normally
decode, preprocess, and VAE-encode between edits. Both distinctions must be stated in any
experimental claim. Planned ablations are direct-latent versus differentiable
decode/re-encode, and stop-gradient versus differentiable conditioning.

## Local development

Use any supported Python 3.10-3.13 environment for lightweight work; Python 3.10 is the
verified Fudan A800 cluster target. Create and activate the environment, install a suitable CPU/CUDA PyTorch
build **inside it**, then install the pinned application dependencies:

```powershell
cd D:\2026WorkExperience\VisonLearnableMemory
python -m venv .venv-dev
.\.venv-dev\Scripts\Activate.ps1
# Install the selected torch/torchvision build here.
python -m pip install -r requirements\runtime-pinned.txt
python -m pip install -e . --no-deps
python scripts\bootstrap\fetch_sources.py
python scripts\bootstrap\preflight.py --mode local
python -m unittest discover -s tests -v
```

Full model weights are not needed for these local contract tests. A metadata-only HF check
is optional:

```powershell
python scripts\bootstrap\fetch_models.py --metadata-only
```

## Cluster bootstrap

The verified Fudan A800 environment is Python 3.10, CUDA module 11.8,
`torch==2.7.1+cu118`, and `torchvision==0.22.1+cu118`. The earlier Torch 2.4.1
environment is incompatible with Diffusers 0.39.0's custom attention-op annotations and
must not be used. The executable bootstrap is `scripts/cluster/setup_fudan_a800.sh`.

```bash
python -m pip install -r requirements/runtime-pinned.txt
python -m pip install -e . --no-deps
python scripts/bootstrap/fetch_sources.py
python scripts/bootstrap/fetch_models.py --model-root "$VLM_MODEL_ROOT"
python scripts/bootstrap/preflight.py \
  --mode cluster \
  --model-root "$VLM_MODEL_ROOT" \
  --expected-torch "$VLM_EXPECTED_TORCH" \
  --min-gpus 2 \
  --min-gpu-memory-gib 40 \
  --output runs/preflight.json
```

The final two resource flags are the conservative starting gate for the two-device E2E
probe, not a measured minimum. Activations and four-step recurrent graphs dominate memory;
the approximately 14 GB of checkpoint files alone are not a useful VRAM estimate. Prefer
Reader and updater on separate 40/80 GB GPUs and leave non-reentrant U-Net checkpointing
enabled. Record the final cluster-specific Torch/CUDA lock after the first successful run.

## Real-model probe order

Set an RGB source image and model paths first:

```bash
export SOURCE_IMAGE=/absolute/path/source.png
export DREAMLITE="$VLM_MODEL_ROOT/DreamLite-mobile"
export READER="$VLM_MODEL_ROOT/Qwen3-VL-4B-Instruct"
mkdir -p runs/probes
```

Run each gate only after its predecessor passes:

```bash
# 1. Raw float image -> frozen Qwen target CE.
python scripts/probes/reader_pixel_grad.py --model "$READER" | tee runs/probes/01_reader.json

# 2. Frozen TinyVAE decode -> frozen Qwen target CE.
python scripts/probes/vae_reader_grad.py \
  --dreamlite "$DREAMLITE" --reader "$READER" | tee runs/probes/02_vae_reader.json

# 3. Official and differentiable DreamLite trajectories, same inputs/noise/schedule.
python scripts/probes/dreamlite_parity.py \
  --model "$DREAMLITE" --source-image "$SOURCE_IMAGE" | tee runs/probes/03_parity.json

# 4. Surrogate loss -> source latent and DreamLite LoRA.
python scripts/probes/dreamlite_sampler_grad.py \
  --model "$DREAMLITE" --source-image "$SOURCE_IMAGE" --checkpoint-unet \
  | tee runs/probes/04_sampler_grad.json

# 5a. One event: DreamLite -> VAE -> Qwen CE -> LoRA.
python scripts/probes/e2e_episode_grad.py \
  --dreamlite "$DREAMLITE" --reader "$READER" --source-image "$SOURCE_IMAGE" \
  --event "the background is a quiet blue room" \
  --query "What room is remembered?" --target "a quiet blue room" \
  --dreamlite-device cuda:0 --reader-device cuda:1 \
  | tee runs/probes/05_e2e_one_event.json

# 5b. Two-event latent-path BPTT.
python scripts/probes/e2e_episode_grad.py \
  --dreamlite "$DREAMLITE" --reader "$READER" --source-image "$SOURCE_IMAGE" \
  --event "the preferred mug is red" --event "the room has a wooden table" \
  --query "What color mug is preferred?" --target "red" \
  --dreamlite-device cuda:0 --reader-device cuda:1 \
  | tee runs/probes/06_e2e_two_event.json

# 5c. Required negative control: cut the recurrent state path.
python scripts/probes/e2e_episode_grad.py \
  --dreamlite "$DREAMLITE" --reader "$READER" --source-image "$SOURCE_IMAGE" \
  --event "the preferred mug is red" --event "the room has a wooden table" \
  --query "What color mug is preferred?" --target "red" \
  --dreamlite-device cuda:0 --reader-device cuda:1 --detach-between-events \
  | tee runs/probes/07_e2e_detach_control.json
```

Success requires exit code zero, finite loss and gradients, non-zero image/source/LoRA
gradient norms, no gradients on frozen parameters, trajectory parity within the fixed
tolerance, and a non-zero intermediate-state gradient only in the non-detached two-event
run. Parity returns exit code 3 on numerical mismatch; bootstrap/preflight returns 2 when a
required contract fails. The E2E report also records pre-clamp out-of-range pixels and the
fraction of zero gradient through the hard clamp.

Only the final paired two-event runs establish the initial **direct-latent recurrent path**
technical closure. They do not yet establish training usefulness, full-conditioning BPTT,
or equivalence to repeated public DreamLite edits.

## Mechanism-v1 experiment implementation

The preregistered defaults live in `configs/experiments/mechanism_v1.yaml`. The v1 oracle
episode contract has three turn types:

- `event`: call the updater and do not compute a Reader loss;
- `query`: read the current state without calling the updater;
- `mixed`: apply only the annotated event span, then answer the annotated query from the
  resulting state.

Event subtypes are `set`, `overwrite`, `clear`, and `noop`. A distractor is intentionally
an updater call labelled `noop`; a pure query is a strict read-only control-flow branch.
The strict JSON schema recursively rejects hidden-ledger fields, and the Reader-facing
interface receives only the image, query, and candidate choices.

Generate and validate the fixed synthetic corpus:

```bash
python scripts/data/generate_synthetic.py --output-dir data/synthetic_v2 --seed 2026
python scripts/data/validate_synthetic.py data/synthetic_v2
python scripts/data/generate_synthetic.py \
  --output-dir data/synthetic_v2_set_only --seed 2026 --transition-profile set-only
python scripts/data/validate_synthetic.py data/synthetic_v2_set_only
```

Each command creates 5,000 train, 500 dev, 1,000 test-ID, and 1,000 test-OOD episodes plus
a content-addressed schema-v2 manifest. The OOD split is evenly divided among held-out
entities, topics, paraphrase templates, and 9-16-turn length extrapolation. The set-only
corpus is independently generated; it is never made by deleting turns from full episodes.

The lightweight implementation is a hashed event encoder, one-layer BiGRU, 64-channel
64x64 FiLM-ConvGRU state, and differentiable RGB head. `lightweight_overfit.py` uses a
fixed local surrogate only for CPU/API smoke tests. The scientific 64-episode gate uses
the real frozen Qwen Reader and fails closed unless it reaches 90% training MCQ accuracy
within 2,000 optimizer steps:

```bash
python scripts/train/lightweight_episode.py \
  --train data/synthetic_v2/train.jsonl --dev data/synthetic_v2/dev.jsonl \
  --reader "$READER" --output-dir runs/lightweight-qwen \
  --method recurrent --overfit-gate --overfit-episodes 64 \
  --max-optimizer-steps 2000 --overfit-threshold 0.90
```

The formal DreamLite trainer supports whole-episode BPTT, direct-latent and differentiable
decode/re-encode recurrence, deterministic per-event noise, LoRA-only parameter
whitelisting, non-reentrant checkpointing, exact checkpoint/resume, and two-device
DreamLite/Reader placement. It refuses dirty source trees, reused fresh-run directories,
invalid hyperparameters, unexpected trainable base weights, frozen gradients, and zero or
non-finite trainable gradients.

Formal DreamLite runs start from a deterministic uniform neutral-gray blank image encoded
once into the TinyVAE latent space. The colorful deterministic image remains a numerical
probe fixture only. `--learn-initial-state` turns the blank-derived latent into a trainable
parameter for the preregistered initialization ablation; checkpoint evaluation infers and
verifies this protocol from the manifest.

## PrefEval adapter and evaluation

`data.lock.json` pins the independent official PrefEval checkout. Fetch it without touching
the existing `PrefEval-GPT56` worktree:

```bash
python scripts/bootstrap/fetch_datasets.py --data-root data/external
python scripts/eval/prepare_prefeval.py \
  --prefeval-root data/external/PrefEval \
  --output runs/prefeval/all.jsonl --protocol all
```

The adapter uses a fixed 20-topic manifest, binds all three forms by base pair, rejects
privileged implicit-preference fields from model input, implements `oracle-sparse` and
`forced-write` k=0/2/5/10, and creates the deterministic seed-2026 16-topic/4-topic split.
Use `--max-base-pairs-per-topic 10` for the preregistered 200-pair forced-write subset.
The forced-write export uses nested distractor prefixes and a stable base-pair/form noise key,
so oracle-sparse and every k variant share the same initial SET noise and corresponding event
noise. Protocol labels and k therefore cannot change the diffusion draw being compared.

Prediction scoring includes topic-by-form macro accuracy, dynamic-state diagnostics,
10,000-draw paired hierarchical bootstrap confidence intervals, and Holm correction:

```bash
python scripts/eval/score_prefeval.py --predictions predictions.jsonl --output scores.json
python scripts/eval/score_synthetic.py --predictions predictions.jsonl --output scores.json
```

## Strict Fudan probe submission

After committing and synchronizing an identical clean checkout on the cluster, first
generate a dry run:

```bash
python scripts/cluster/submit_probe_gates.py --through G6 --dry-run
```

Run the same command without `--dry-run` to submit G1-G6. Each job is single-node, has an
explicit GPU/time/memory request, runs strict preflight, and is chained with Slurm
`afterok`. Logs and result JSON are written outside the repository. G6 validates that the
two-event positive and detach reports have identical forward metadata/loss within the
fixed tolerance and opposite intermediate-gradient behavior. The submission manifest is
updated atomically after every sbatch, so partial submissions remain auditable.

After the final clean commit re-passes G1--G6, the post-gate orchestrator builds the entire
stop-gated formal matrix. It defaults to a dry run; real submission additionally requires
`--submit` and the exact expected commit:

```bash
python scripts/cluster/submit_experiment_dag.py \
  --expected-commit "$(git rev-parse HEAD)" --through eval \
  --fetch-prefeval --include-ablations --include-prefeval-adaptation
```

Pilot learning-rate selection, its two baseline gains, and reset/shuffle checks use only the
dev split. Test-ID is first evaluated after the selected learning rate is frozen. Failed data,
sanity, lightweight, pilot, or resume gates cancel all dependent Slurm jobs.

## Repository boundaries

- `third_party/DreamLite` remains clean, detached at the locked commit, and is ignored by
  the root repository.
- `PrefEval-GPT56` is an existing dirty nested repository and is intentionally untouched.
- Models, outputs, runs, caches, and local environment/credential files are ignored.
- DreamLite weights retain their non-commercial license; do not redistribute them.
