# Full U-Net capacity control

Recorded before dispatch, 2026-09-13. The active goal remains unfulfilled. The Base512 LoRA endpoint and its CFG1 evaluation both failed all five question variants. A separate, unchanged LoRA3500 run remains active. This control tests whether restricting adaptation to rank16 attention LoRA is sufficient to explain the Base512 failure.

## Fixed comparison

- Fresh sealed DreamLite Base initialization; train every original U-Net parameter, add no LoRA. VAE, image/text conditioner and Reader remain frozen. This is an explicit capacity experiment beyond the upstream LoRA example, not a claim that the example uses full U-Net training.
- Keep the official target/noise FM equation, full sigma interval, integer timestep, original raw event conditioning, source only on the condition side, pure-noise inference and upstream native28-step CFG7.5/imageCFG1.
- Exactly512 optimizer updates, accumulation4, AdamW lr5e-5, betas(.9,.999), epsilon1e-8, weight_decay1e-4, clip1; FP32 Writer and bf16 Reader. Same seed20260913 and hash-selected single teacher as Base512. All2048 training draw identities must match Base512. No target selection based on evaluation, no early best-checkpoint selection.
- Bank SHA256 `20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722`; complete Base seal and official source a6e20c8 remain unchanged.
- Same8 evaluation noises, five Reader prompts, blank and different-answer donor controls; retain greedy32-token raw answers and immediate EOS. These are repeated diagnostic benchmark noises, not untouched research holdouts.

## Placement and checkpoint differences

The new two-H200 request could not fit the project's remaining one-GPU quota and was stopped before doing work. The new `dl-align-full-h200x1-20260913` instance has one full-memory H200, NGC25.02/CUDA12.8,20CPU/200GiB and64GiB shared memory, in the same project/workspace/compute group. Writer and Reader are explicitly co-located on cuda:0. This layout is recorded in identity/dispatch rather than silently changing the default two-GPU training path.

Before any optimization, compare all8 baseline latent/image tensors, every native trajectory point, and all50 raw Reader records against the completed Base512 two-GPU baseline. Require bitwise tensor equality and identical raw records, plus runtime/model/condition/schedule equality. Bind that reference with result SHA256 `afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4`. A mismatch stops the run at zero updates for investigation; no tolerance relaxation is implicit.

Full model plus optimizer checkpoints are much larger. Save the first update, every16 updates, the final update and graceful stops. Checkpoints retain exact optimizer and RNG state. After a hard interruption, preserve the pre-recovery physical log and replay updates after the last saved checkpoint. The CPU recovery test verifies bitwise losses, weights and Adam state for this case. Do not compare wall-clock cost as though checkpoint I/O or device placement were identical.

Worker deadline1789251300 (06:15 Beijing time), before the notebook's automatic stop around06:40. Runtime and dispatch are bound to a clean committed checkout. Results require final checkpoint/result hash verification, all512 metrics, matched draw identities, frozen-model audit and complete raw evaluation.

## Interpretation

The primary outcome is paired raw answer plus immediate EOS across all five prompts and fresh generation noises; latent distances and loss are diagnostics. A positive single-teacher result establishes minimum learnability only. It still requires newly fixed noises and multiple event/state controls before an event-conditioned usable-version claim. Failure rejects this specified full-U-Net512 configuration, not all possible capacity or objective explanations.

`P=/inspire/ssd/project/exploration-topic/czxs26210936`; reference run `P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`.
