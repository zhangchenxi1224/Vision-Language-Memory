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

Use any supported Python 3.10-3.13 environment for lightweight work. Create and activate
the environment, install a suitable CPU/CUDA PyTorch build **inside it**, then install the
pinned application dependencies:

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

## R3 Inspire H200 bootstrap

The current R3 target is the existing Inspire notebook
`vlm-r3-h200x2-live-20260717`: one node with 2 H200 GPUs, the official
`ngc-pytorch:25.02-cuda12.8.0-py3` image, Python 3.12, CUDA 12.8, and the
image-supplied `torch==2.7.0a0+ecf3bae40a.nv25.02`. R3 must not install a
replacement PyTorch wheel. It uses an overlay venv with
`--system-site-packages` and installs only the explicitly locked non-Torch packages with
`--no-deps`. Full path, egress, background-sentinel, and formal-preflight instructions are
in [INSPIRE.md](INSPIRE.md).

The verified allocation is one `qb-prod-gpu2007` node with two `143771 MiB`
H200s, 40 CPUs, 400 GiB RAM, and 128 GiB SHM. Public Git/Hugging Face work is
performed by the running CPU preparation notebook
`vlm-r3-egress-cpu-live-20260717`; the GPU notebook reads the clean commit,
environment, data, caches, and model snapshots from project-shared storage.

```bash
source /absolute/private/path/vlm-r3.env
export VLM_EXPECTED_COMMIT=<FULL_40_CHARACTER_R3_COMMIT>
bash scripts/inspire/bootstrap_r3_h200.sh

# After the locked model snapshots have been reconstructed through the
# detached infrastructure launcher, create the scientific preflight.
$VLM_VENV_ROOT/bin/python scripts/inspire/preflight_r3_h200.py \
  --repo "$PWD" \
  --model-root "$VLM_MODEL_ROOT" \
  --expected-commit "$VLM_EXPECTED_COMMIT" \
  --require-models \
  --output "$VLM_RUN_ROOT/preflight/$VLM_EXPECTED_COMMIT/r3_h200_formal.json"
```

The H200 preflight fails on a dirty or different commit, a venv-local Torch copy, any
runtime/package drift, fewer than two H200 devices, incomplete immutable model snapshots,
or a missing deterministic environment variable. Historical A800 technical results are not
accepted as H200 R3 gate evidence; R3-R0, R3-S0, G4-L, G5-L, G6-L, and DL-S are rerun
serially. R3-R0 first requires the repaired Qwen resize to preserve legacy forward tensors
bit-for-bit, match the native CPU mathematical adjoint exactly, remain within preregistered
tolerance of three isolated legacy CUDA backward references, and produce finite, nonzero,
bitwise-repeatable candidate gradients under strict mode. The isolated references contain no
model weights, optimizer, or scientific metric and restore strict mode after every run.
Model identity is closed by `vlm.hf-snapshot-sha256.v1` manifests over every non-cache
snapshot file, not merely revision marker text or aggregate byte counts. Their SHA values
are carried by the formal preflight, immutable plans, stage evidence, and strict training
checkpoint provenance.
The old Fudan launch path is historical only. Any `render_r3_*` code under `scripts/cluster`
is a static command-contract/dry-run fixture; neither its sbatch output nor an A800 result is
authoritative R3 evidence. Formal evidence must be created by the Inspire launcher and bind
its exact formal-preflight and worker-input SHA256 values.

Teacher-assisted micro experiments additionally require the completed immutable
`R3-TC0 -> R3-TF0 -> T0 -> CAL-Set8 -> CAL-Transition16` preparation DAG. Its generated
calibrations live in the run directory rather than the read-only teacher cache. QA-only
experiments fail closed if any teacher parent, cache, sidecar, or calibration is supplied.

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

The lightweight implementation is a hashed event encoder, one-layer BiGRU, an
event-conditioned 16-mode orthogonal DCT-II writer, a 64-channel 64x64 FiLM-ConvGRU
state, and a differentiable RGB head. The spatial writer and pre-GRU FiLM map are bounded;
the update-gate bias starts at -1, retaining most prior state while providing an initial
write ratio of about `sigmoid(-1)=0.269`. The fixed zero
initial state remains non-trainable. Formal training logs per-module gradient norms,
the actual clipping factor, conditioned-input magnitude, gate saturation, and hidden-state
bounds.
`lightweight_overfit.py` uses a fixed local surrogate only for CPU/API smoke tests. The
scientific 64-episode gate uses the real frozen Qwen Reader and fails closed unless it
reaches 90% training MCQ accuracy at exactly the final 2,000th optimizer step. Intermediate
threshold crossings are logged as trajectory diagnostics and never trigger early stopping:

```bash
python scripts/train/lightweight_episode.py \
  --train data/synthetic_v2/train.jsonl --dev data/synthetic_v2/dev.jsonl \
  --reader "$READER" --output-dir runs/lightweight-qwen \
  --method recurrent --overfit-gate --overfit-episodes 64 \
  --max-optimizer-steps 2000 --overfit-threshold 0.90
```

CUDA reproducibility is audited separately from that scientific gate. The paired probe runs
two fresh Python processes serially on one allocated GPU, enables fail-closed deterministic
algorithms and math-only SDPA, and compares exact gradients, optimizer states, RNG states,
per-step traces, and canonical predictions. Its diagnostic renderer uses integer repeat to
256x256 without a crop and disables Qwen processor resizing; therefore its accuracy is not a
D2 result and must not be compared with the production bilinear path. It also replaces CUDA
NLLLoss only in this diagnostic with an equivalent FP32 logsumexp-minus-target formulation;
the default Reader loss remains `F.cross_entropy`.

```bash
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/probes/run_lightweight_determinism_pair.py \
  --train data/synthetic_v2/train.jsonl --reader "$READER" \
  --output-dir runs/repro-pair --steps 100 --device cuda:0
```

After the 1-step and 100-step bitwise audits pass, the same strict path can run the
preregistered `R1/D2R` exact-64 reachability gate. This is a prospective deterministic
lightweight variant, not a retroactive production-bilinear D2 result. Both fresh replicas
must match bitwise, complete exactly 2,000 optimizer steps, and finish with at least
116/128 correct. A threshold failure still writes and compares both complete child reports;
the pair wrapper then exits non-zero with `reproducibility_valid=true` and
`reachability_gate_passed=false`.

```bash
python scripts/probes/run_lightweight_determinism_pair.py \
  --train data/synthetic_v2/train.jsonl --reader "$READER" \
  --output-dir runs/d2r-exact64-pair --steps 2000 --device cuda:0
```

`target-only` above is retained solely for the historical R1/D2R record. The prospective
R2/D2L protocol aligns training with four-choice evaluation: each choice is scored by its
negative mean target-token NLL, then the query loss is a temperature-1 FP32 listwise CE.
The token CE used inside every choice forward is the deterministic FP32
logsumexp-minus-target-token-score formula. All four choice forwards retain autograd to the
same updater image. R2a fails unless every optimizer step has finite, positive gradients at
both that image and the updater parameters.

Advance the listwise protocol strictly through 1, 100, and 2,000 steps:

```bash
# R2a: autograd + bitwise reproducibility smoke.
python scripts/probes/run_lightweight_determinism_pair.py \
  --train data/synthetic_v2/train.jsonl --reader "$READER" \
  --output-dir runs/r2a-listwise-pair --steps 1 --device cuda:0 \
  --reader-loss-mode listwise-choice

# R2b: short paired reproducibility audit; run only after R2a passes.
python scripts/probes/run_lightweight_determinism_pair.py \
  --train data/synthetic_v2/train.jsonl --reader "$READER" \
  --output-dir runs/r2b-listwise-pair --steps 100 --device cuda:0 \
  --reader-loss-mode listwise-choice

# R2c/D2L: exact-64 scientific gate; run only after R2b passes.
python scripts/probes/run_lightweight_determinism_pair.py \
  --train data/synthetic_v2/train.jsonl --reader "$READER" \
  --output-dir runs/r2c-listwise-exact64-pair --steps 2000 --device cuda:0 \
  --reader-loss-mode listwise-choice
```

R2c requires both fresh replicas to match bitwise and, in each replica, at least 116/128
correct in the canonical and left-rotate-one views, at least 28/32 for every target position
in both views, at least 20/24 canonical mixed queries, and at least 60/64 clean/distractor
prediction-text agreements. A clean/distractor pair is valid only when its ordered choices
and target text are identical. The wrapper independently reconstructs all gate semantics;
a self-reported but incomplete or inconsistent child gate cannot pass.

`scripts/probes/qwen_visual_control_upper_bound.py` is a deliberately target-supervised
diagnostic: the answer position selects one of four learned images. It tests whether the
frozen Reader can be controlled through its visual channel, but it is not a memory method,
baseline, or ablation and must never be reported as one.

`scripts/probes/qwen_renderer_control_upper_bound.py` applies the same leaked-label
selection to four trainable hidden-state codes and passes them through the production RGB
head. It isolates renderer-manifold reachability and has the same diagnostic-only status.
`scripts/probes/qwen_event_prefix_semantic_upper_bound.py` is stricter: its code selector
hashes only the ordered visible event-text prefix, while disjoint even/odd candidate
permutations test whether one state image carries answer semantics across answer positions.
That probe is transductive and still is not an updater, baseline, ablation, or generalization
result.

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

## Strict Inspire R3 execution

Long Inspire operations are detached processes with immutable input metadata,
`running.json`, complete stdout/stderr, content hashes, and an atomic `terminal.json`.
`scripts/inspire/launch_background.py` rejects a reused run directory, a changed/dirty
commit, preflight SHA drift, and (for scientific stages) any report without
`formal_ready=true`. Monitor without mutating the stage:

```bash
$VLM_VENV_ROOT/bin/python scripts/inspire/poll_stage.py /absolute/stage/run-directory
```

The scientific order is R3-R0 -> R3-S0 -> G4-L -> G5-L -> G6-L -> DL-S. R3-R0 is the
bitwise-forward-equivalence and strict-backward-repeatability gate; each later stage launches
only after the previous terminal report and scientific validator pass. Model downloads and
teacher-cache construction use the explicitly restricted `--infrastructure-stage` path;
that path cannot authorize a scientific gate. No DreamLite pilot is launched directly.

### Historical Fudan Slurm path (R1/R2 only)

The following command only renders the archived Fudan probe plan for audit/reproduction:

```bash
python scripts/cluster/submit_probe_gates.py --through G6 --dry-run
```

Do not remove `--dry-run` and do not use `setup_fudan_a800.sh` or
`submit_experiment_dag.py` for R3. Their Python 3.10/CUDA 11.8/Torch 2.7.1 A800 lock and
Slurm dependency semantics describe the historical platform, not the current H200 runtime.

## Repository boundaries

- `third_party/DreamLite` remains clean, detached at the locked commit, and is ignored by
  the root repository.
- `PrefEval-GPT56` is an existing dirty nested repository and is intentionally untouched.
- Models, outputs, runs, caches, and local environment/credential files are ignored.
- DreamLite weights retain their non-commercial license; do not redistribute them.
