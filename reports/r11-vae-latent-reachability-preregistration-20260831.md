# R11 VAE-latent reachability preregistration

Status: fixed after complete R10 direct-pixel `8/8`, DreamLite EMA `0/8`, and DreamLite raw endpoint `0/8` results. No R11 outcome has been observed.

## First-principles question

R10 proves that the frozen Qwen image interface contains readable directions, but the successful images are dense non-semantic textures while DreamLite states remain almost uniform gray. Three explanations remain nested:

1. readable codes exist in unconstrained RGB but not in images decodable by the frozen DreamLite VAE;
2. the VAE latent space can represent them, but the semantic four-step U-Net editor cannot reach them;
3. the U-Net can reach them in principle, but the current rank-4 LoRA conditioning or optimizer cannot learn the map.

R11 tests only the first boundary. It must run before another LoRA-rank, clipping, EMA, prompt, or recurrent-memory experiment.

## Fixed oracle

Use the exact eight immutable R10 F1 targets and fixed data hashes. For each target independently:

- encode the exact uniform `127/255` RGB initial state with the frozen DreamLite VAE posterior mean;
- make that one model-space latent tensor the only trainable object, stored in FP32;
- decode it through the unchanged frozen VAE scale/shift path and unit-RGB clamp;
- pass the decoded image through the unchanged frozen Qwen Reader and deterministic resize;
- do not execute the DreamLite U-Net, text condition encoder, semantic edit prompt, event noise, LoRA, or recurrence;
- optimize with Adam, constant LR `0.05`, zero weight decay, for exactly 256 steps;
- train on the four forward-cyclic choice views exactly 64 times each;
- save latent and decoded-image checkpoints at steps `0/64/128/192/256`;
- evaluate only raw latent step256 on the four disjoint reverse-cyclic views under `normal` and `reset`.

The raw step256 endpoint is fixed. Intermediate checkpoints describe trajectory only and cannot rescue it.

## Gates

A target passes only when all conditions hold:

1. exact technical receipts, one and only one trainable latent, frozen VAE/Reader, immutable model snapshots, 256 finite nonzero-gradient steps, exact view counts, and all checkpoints/images;
2. normal endpoint CE improves by at least 20% relative to its own M0;
3. all four fixed reverse-cyclic views improve;
4. accuracy rises by at least 0.25;
5. normal/reset difference-in-differences is negative.

The oracle arm passes only at `8/8`. Any partial count is diagnostic and never formal success.

## Locked interpretation

| Latent oracle | Conclusion and next action |
| --- | --- |
| 8/8 | The frozen VAE representation can carry the code. The bottleneck lies after representation selection: replace the semantic DreamLite editor with a direct event-to-latent residual writer, then require shared held-out F1 alignment. |
| 1--7/8 | VAE-space readability is target-dependent. Attribute fixed query/choice/token properties before a general writer claim. |
| 0/8 | The current VAE-decodable region has not demonstrated the pixel code. Bypass or augment the VAE with a direct image-space residual codec. |
| technical failure | Rerun only invalid targets in fresh roots; draw no scientific conclusion. |

R11 directly optimizes the answer-supervised state independently per target. Even `8/8` would not establish a shared memory encoder, held-out item generalization, recurrence, ID/OOD success, or transfer to a closed VLM.
