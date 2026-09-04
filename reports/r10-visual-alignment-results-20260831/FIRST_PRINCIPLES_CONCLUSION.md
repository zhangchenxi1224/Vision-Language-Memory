# R10 first-principles conclusion

## Outcome

- All 16 preregistered R10 runs were technically valid.
- Direct-pixel oracle: **8/8 passed**.
- DreamLite single SET, fixed EMA step128: **0/8 passed**.
- Post-registered DreamLite raw step128 attribution: **0/8 passed**.
- Formal picture-memory success: **false**.
- Locked next branch: test VAE-latent reachability, then redesign only the writer side.

## What is now established

Picture memory requires four nested capabilities: a readable visual direction, a state representation that can reach it, a learned writer that can produce it, and recurrent composition/generalization. R10 isolates only the first two levels.

The direct-pixel arm drove held-out reverse-cyclic CE from 13.16--34.63 to approximately `0.0003--0.0012` on every target. All four held-out option views improved, accuracy rose from 0 to 1, and resetting the image removed the gain. Therefore, for these eight F1 targets, the frozen Qwen Reader, deterministic resize, listwise loss, and image interface admit causal readable codes.

This is a capacity lower bound, not a learned memory result. Each image was independently optimized with answer supervision; R10 does not show a shared encoder, held-out item generalization, recurrence, or closed-model transfer.

## What failed

The same eight targets trained independently through the current source-anchored DreamLite rank-4 LoRA writer. EMA endpoint CE changed by only `-1.43%` to `+0.17%`, no target gained any accuracy, and the fixed gate failed 8/8 times. The 16-step mean training curves remain nearly flat while direct-pixel loss falls by roughly five orders of magnitude.

Raw step128 weights are somewhat stronger than EMA: seven of eight have lower CE than M0, five improve all four held-out views, and the best relative CE change is `-6.90%`. Nevertheless, raw accuracy remains unchanged for all eight and raw also fails 0/8. EMA lag therefore contributes to attenuation but is not a sufficient root cause.

## Gradient clipping is not the sufficient explanation

DreamLite clipping rates span `2.34%--59.38%`, yet every target fails. Targets with very little clipping (for example 2.34%, 3.12%, and 7.81%) still show only negligible endpoint movement, while the most-clipped target shows one of the larger but still insufficient raw improvements. Clipping reflects unstable, target-dependent gradient geometry and may reduce progress, but the evidence rejects “clipping alone caused failure.”

## Visual phenomenon

The successful pixel endpoints are dense, non-semantic, high-frequency textures. They are not human-readable depictions of the stored entity or preference. In contrast, the DreamLite raw endpoint states remain almost uniform gray.

This is aligned with the original research hypothesis: model-readable memory need not be human readable. It also exposes the current architectural mismatch. The unconstrained pixel optimizer can move into Qwen-sensitive directions, whereas the text-conditioned diffusion editor and its source-anchored semantic prior remain in a narrow image region and do not reach those codes.

## Next main-line experiment

Run an eight-target **VAE-latent oracle** with the same F1 targets, fixed reverse-cyclic views, reset controls, and fail-closed gates:

1. optimize one free DreamLite model-space latent per target;
2. decode it through the unchanged frozen VAE and Reader;
3. forbid DreamLite U-Net updates, semantic edit prompts, recurrence, and checkpoint selection;
4. use a fixed endpoint and preserve all latent/image trajectories.

Interpretation is locked:

- latent 8/8: the VAE state representation can carry the code; the bottleneck is DreamLite U-Net conditioning/parameterization/optimization, so replace the semantic editor with a direct event-to-latent writer;
- latent 1--7/8: VAE-space readability is target-dependent; diagnose target/token properties before a shared writer;
- latent 0/8: successful codes lie outside the current VAE-decodable region; bypass or augment the VAE with an image-space residual codec.

Only after a shared writer passes held-out F1 should recurrence, overwrite, ID/OOD, and multi-seed formal evaluation resume.

## Key visual files

- `main/training_diagnostics.png`: direct-pixel loss collapses; DreamLite loss is nearly flat, with target-dependent gradient spikes.
- `main/endpoint_metrics.png`: pixel passes every endpoint dimension; DreamLite fails the CE and accuracy dimensions.
- `main/pixel_endpoint_contact_sheet.png`: model-readable, non-human-readable pixel codes.
- `raw-endpoint-attribution/report-fdb831b/raw_vs_ema_metrics.png`: raw improves more than EMA but remains far below the gate.
- `raw-endpoint-attribution/report-fdb831b/raw_state_contact_sheet.png`: DreamLite states remain visually close to uniform gray.
