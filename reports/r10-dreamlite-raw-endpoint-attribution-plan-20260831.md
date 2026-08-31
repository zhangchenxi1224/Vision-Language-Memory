# R10 DreamLite raw-endpoint attribution plan

Status: fixed after the preregistered EMA endpoints for DreamLite targets 0--3 were observed to fail, before targets 4--7 completed and before any raw endpoint was evaluated.

## Main-line question

R10 fixes `EMA step128` as its sole DreamLite endpoint. The saved `endpoint_raw.pt` is not allowed to replace that endpoint. It can, however, distinguish two materially different causes before the updater architecture is changed:

1. the raw LoRA weights learned a readable one-SET code but EMA lag erased it at step 128;
2. even the raw LoRA weights did not learn a readable code, so EMA is not a sufficient explanation.

This is a root-cause attribution pass, not a new training experiment and not a success claim.

## Fixed evaluation

- Evaluate all eight immutable R10 F1 targets; no target may be selected or omitted by outcome.
- Revalidate each completed R10 DreamLite terminal, summary, artifact inventory, and SHA binding before loading its raw endpoint.
- Load exactly `run/endpoint_raw.pt` from optimizer step 128.
- Reconstruct the same frozen DreamLite, VAE, Qwen Reader, source-anchored one-SET event, fixed seed, fixed data, and deterministic preprocessing contract.
- Use the same four held-out reverse-cyclic choice views and both `normal` and `reset` controls.
- Persist all row-level evaluations, target summaries, state images, logs, environment, source hashes, and a fail-closed artifact inventory.
- Intermediate checkpoints and best-checkpoint selection are outside this attribution pass.

For descriptive comparability, recompute the unchanged R10 target gate: CE relative change at most `-20%`, all `4/4` views improved, accuracy delta at least `+0.25`, and negative normal/reset difference-in-differences. Passing this descriptive raw gate cannot change the preregistered EMA gate.

## Locked interpretation

| Raw step128 | Existing EMA step128 | Interpretation and next action |
| --- | --- | --- |
| 8/8 | fewer than 8/8 | EMA lag is a sufficient endpoint bottleneck. Repair the endpoint averaging/training horizon before changing representation. |
| 1--7/8 | fewer than 8/8 | EMA may contribute, but the current updater remains target-dependent and insufficient. Continue to the VAE-latent reachability oracle. |
| 0/8 | fewer than 8/8 | EMA is not a sufficient explanation. Continue to the VAE-latent reachability oracle. |
| technical failure | any | Rerun only invalid attribution targets; make no scientific inference from them. |

Even raw 8/8 would remain a one-seed, independently trained, repeated-target diagnostic. It would not establish shared writing, held-out generalization, recurrence, or formal picture-memory success.
