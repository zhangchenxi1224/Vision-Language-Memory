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

Select and pin the CUDA-matched `torch`/`torchvision` pair only after inspecting the target
node. Copy `configs/cluster.env.example` to an untracked local file, set the shared model
root, HF cache, and exact installed Torch version, then run:

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

## Repository boundaries

- `third_party/DreamLite` remains clean, detached at the locked commit, and is ignored by
  the root repository.
- `PrefEval-GPT56` is an existing dirty nested repository and is intentionally untouched.
- Models, outputs, runs, caches, and local environment/credential files are ignored.
- DreamLite weights retain their non-commercial license; do not redistribute them.
