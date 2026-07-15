# Fudan A800 execution

The verified initial runtime is Python 3.10 with PyTorch 2.4.1/cu118. Persistent paths:

```text
/remote-home1/cxzhang/codex_runs/vision-language-memory
/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118
/remote-home1/cxzhang/codex_models/vision-language-memory
```

`setup_fudan_a800.sh` is intended for a CPU-only Slurm setup job. It creates an isolated
environment, installs pinned dependencies, reconstructs source/model snapshots, runs local
preflight, and executes mock autograd tests. Real probes must use a separate GPU `sbatch`
and strict cluster preflight.
