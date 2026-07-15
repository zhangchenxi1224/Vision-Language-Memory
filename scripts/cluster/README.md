# Fudan A800 execution

The verified runtime is Python 3.10 with PyTorch 2.7.1/cu118. PyTorch 2.4.1 is
incompatible with Diffusers 0.39.0's custom attention-op annotations and must not be
used for the real-model gates. Persistent paths:

```text
/remote-home1/cxzhang/codex_runs/vision-language-memory
/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118_torch271
/remote-home1/cxzhang/codex_models/vision-language-memory
```

`setup_fudan_a800.sh` is intended for a CPU-only Slurm setup job. It creates an isolated
environment, installs pinned dependencies, reconstructs source/model snapshots, runs local
preflight, and executes mock autograd tests. Real probes must use a separate GPU `sbatch`
and strict cluster preflight.

## Post-G6 experiment DAG

`submit_experiment_dag.py` starts after the separately completed G1--G6 technical gates.
It defaults to a no-submit dry-run and writes every `sbatch` plus an atomic
`submission.json`. Actual submission requires an explicit clean commit:

```bash
python scripts/cluster/submit_experiment_dag.py \
  --expected-commit "$(git rev-parse HEAD)" \
  --through pilot

python scripts/cluster/submit_experiment_dag.py \
  --expected-commit "$(git rev-parse HEAD)" \
  --through pilot \
  --submit
```

The chain is data generation/validation, real-Qwen sanity, the real-Qwen 64-episode
overfit gate, three learning-rate pilots, preregistered pilot selection, checkpoint-resume
equivalence, three-seed full training, and the seven-method synthetic/PrefEval matrix.
Every dependency uses `afterok` plus `--kill-on-invalid-dep=yes`; a failed scientific gate
therefore prevents and cancels its descendants. Every task declares `--nodes=1`, validates
the clean commit, activates the torch-2.7.1 environment, and checks the exact torch version.

The selected pilot must be the minimum-dev-loss candidate and must improve by at least ten
percentage points over both blank and frozen DreamLite while reset or shuffle loses at least
ten points. All three checks use **dev predictions only**; test-ID remains untouched until the
learning rate is frozen. `select_pilot.py` fails closed on any other selection split and writes
the selected learning rate atomically. The next job resumes
that candidate from optimizer step 100 and `compare_checkpoints.py` compares the resulting
LoRA, optimizer, cursor, manifest, and trainer state with the uninterrupted run.

Resource and command overrides can be supplied through `--config-json`. Formal ablations are
opt-in with `--include-ablations`. Because the full-transition corpus has no eligible set-only
episodes, this mode generates a separate `synthetic_set_only_v2` corpus. Supplying both
`--set-only-train` and `--set-only-dev` replaces that generated corpus with an explicit
independent dataset; supplying only one path fails closed. The evaluation stage also persists
test-OOD predictions, per-form/per-k PrefEval scores, the adapted state-streaming score, and
merged ablation/noise-robustness summaries.

Default walltimes are conservative ceilings calibrated below the A800 partition's advertised
`MaxTime=3-12:00:00`: 36 hours for one-GPU Qwen, pilot, and evaluation jobs, and 72 hours for
full two-GPU training. This explicitly accounts for four teacher-forced Reader forwards per
MCQ choice set and repeated DreamLite updates; the final ledger reports actual elapsed time,
not these requested ceilings.

At any time after submission, harvest Slurm accounting and content-address the logs/results:

```bash
python scripts/cluster/collect_slurm_results.py \
  --submission /absolute/run/submission.json \
  --output /absolute/run/slurm_ledger.json
```

Add `--require-terminal` for final acceptance. The ledger records every job state and exit
code, actual elapsed GPU-hours by stage, failure count, stdout/stderr hashes, and every file
under the run's `results/` directory. Failed jobs remain in the ledger; they are never dropped
from the resource account.

After the final commit has independently re-passed G1--G6, generate the complete formal DAG
before submitting it:

```bash
python scripts/cluster/submit_experiment_dag.py \
  --expected-commit "$(git rev-parse HEAD)" \
  --through eval --fetch-prefeval \
  --include-ablations --include-prefeval-adaptation

python scripts/cluster/submit_experiment_dag.py \
  --expected-commit "$(git rev-parse HEAD)" \
  --through eval --fetch-prefeval \
  --include-ablations --include-prefeval-adaptation --submit
```
