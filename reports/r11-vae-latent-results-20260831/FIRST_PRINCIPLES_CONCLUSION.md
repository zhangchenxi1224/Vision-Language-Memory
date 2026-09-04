# R11 first-principles conclusion

## Result

The preregistered R11 VAE-latent reachability oracle passed `8/8` fixed F1 targets. Every target completed exactly 256 Adam steps, used the four forward-cyclic training views 64 times each, preserved frozen VAE/Reader snapshots, and passed all four reverse-cyclic endpoint views plus the reset causal control.

Endpoint normal CE fell by `99.9952%` to `99.9998%` across targets. Normal accuracy changed from `0` to `1` for every target. Resetting the image restored the original high CE, so the gain is carried by the learned visual state rather than the question text or answer position.

## What the causal sequence establishes

The combined R10/R11 evidence localizes the bottleneck:

1. R10 direct RGB pixels passed `8/8`: the frozen Qwen image interface contains answer-relevant visual directions.
2. R10 DreamLite EMA and raw endpoints passed `0/8`: the existing semantic four-step U-Net/LoRA updater did not reach those directions.
3. R11 direct VAE latents passed `8/8`: the frozen DreamLite VAE-decodable image manifold itself contains readable codes.

Therefore the main failure is not that Qwen cannot read image memory, nor that the VAE cannot represent it. The demonstrated bottleneck is the current **writer**: semantic prompt conditioning plus diffusion editing does not learn a reliable event-to-memory code.

## Visual and optimization evidence

- The decoded states rapidly become dense, non-human-readable pseudo-character/texture patterns rather than literal pictures of the entity or preference.
- Most loss reduction occurs in roughly the first 40--60 steps; latent displacement also plateaus around that point.
- Gradients remain finite and nonzero throughout. R11 uses no gradient clipping.
- Endpoint pixel saturation is effectively zero (at most about `1.4e-5` in the trajectory plot), so success is not caused by clipping pixels to black/white boundaries.
- Images look nearly stable after step 64 while Reader CE continues to improve, showing that small model-sensitive visual changes can carry information invisible to ordinary human inspection.

This supports the original Picture Memory hypothesis: useful memory need not be human-readable imagery; it can be a learned protocol in continuous visual/latent directions to which the VLM is sensitive.

## What R11 does not prove

R11 independently optimized one answer-supervised latent tensor per target. It does **not** prove that one shared writer can encode unseen events, that the state can be updated recurrently, or that ID/OOD and multi-seed criteria pass. Consequently `formal_success_claim` remains `false`.

## Locked next main-line experiment

Replace the semantic diffusion editor with a shared event-conditioned residual writer operating directly in DreamLite model-latent space:

`previous latent + event representation -> shared residual writer -> updated latent -> frozen VAE -> frozen Qwen Reader loss`

The next discriminating gate is held-out F1 generalization under a fixed train/dev target split. The writer must receive only the memory event at write time; the query is used only to compute training/evaluation loss. It may not contain per-item trainable latent parameters or select a best checkpoint on the held-out set.

Only after this shared writer passes held-out F1 causal controls should recurrence (`SET`, `OVERWRITE`, `DELETE`, interference), multiple seeds, and final ID/OOD evaluation resume. Further semantic-prompt, EMA, LoRA-rank, or clipping sweeps are off the main line unless the direct latent writer fails with evidence that specifically reopens them.
