# R7 deployment audit

## Immutable training snapshot

- Git commit: `c720f6b28e3ce6ef4e8f838a576d3b042d35cd58`
- Worktree:
  `/inspire/ssd/project/exploration-topic/czxs26210936/Vision-Language-Memory-r7-c720f6b-20260830`
- Train SHA-256: `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`
- Dev SHA-256: `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303`
- Selected hard8 SHA-256:
  `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`

The detached worktree was fetched through the public CPU relay, found clean, and passed the 26
targeted R5/R6/R7 tests on the H200 runtime.  A real CUDA aggregation probe on H200 changed the
raw/applied cosine to `0.714142` while preserving the global norm with zero measured relative
error.

## Preserved preflight failures

The first launch attempt used the correct code, data, model-manifest bindings, and CUDA settings,
but omitted two process-level determinism variables.  Both fail-closed controllers exited before
model loading or any optimizer step:

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r7/
  r7-gradient-balance-c720f6b-20260830/
```

| Arm | Exit time | Child exit | Cause | Scientific result? |
| --- | ---: | ---: | --- | --- |
| raw mean | 2.400 s | 1 | missing `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | no |
| unit balanced | 2.423 s | 1 | same fail-closed environment mismatch | no |

The two stderr payloads have identical SHA-256
`4cf1f19da379e999143b9864981825c4bfbe9cc1cb62cb40ecf25ee339e13dc4`.  These directories are
retained as engineering evidence and must never be merged with the valid paired run.

## Corrected paired launch

The corrected run uses a fresh output root:

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r7/
  r7-gradient-balance-c720f6b-detfix1-20260830/
```

| Instance | Arm | Aggregation |
| --- | --- | --- |
| `vlm-dreamlite-full-h200x2-20260720` | `raw-mean-control` | `mean(g_i)` |
| `vlm-r3-h200x2-live-20260717` | `unit-balanced-norm-matched` | equal-unit direction, raw norm matched |

Both launches bind:

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

and the immutable DreamLite/Qwen snapshot-manifest hashes recorded by R6.  Both completed the
identical M0 evaluation and initial eight-segment gradient audit before entering training.  The
first optimizer step had identical loss and pre-clip gradient norm across arms, which confirms a
matched starting state; subsequent divergence is expected from the preregistered aggregation
intervention.

This audit records deployment validity only.  It makes no scientific success claim; endpoint
status and conclusions belong in the completed R7 result package.
