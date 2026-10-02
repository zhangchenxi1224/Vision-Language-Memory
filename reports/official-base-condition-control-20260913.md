# Frozen checkpoint conditioning control

Registered before observing the full-U-Net512 endpoint. The Base512 rank16 checkpoint failed native CFG7.5 and CFG1; its native prompting still differs from the raw-event training example. The full-U-Net512 capacity run is currently training and has passed bitwise baseline parity against Base512.

After the full512 run reaches a verified completed terminal state, evaluate its final checkpoint in two fixed zero-update arms:

1. Native upstream28-step pipeline, native diptych prompt, guidance_scale1 and image_guidance_scale1.
2. The same native28-step pipeline and both scales1, but return the exact cached raw-event training condition and mask when upstream requests prompt encoding. Repeat these tensors across the three native edit branches. Preserve source branches, timestep/sigma schedule, denoising and CFG arithmetic. This is an explicit diagnostic override, not a claim that native default inference uses raw training tensors.

Both arms use the same8 benchmark noises, five Reader question variants, blank and different-answer donor controls as the parent; same final weights, source, model precision, colocated H200 layout and frozen Reader. No optimizer is created by the probe; all model parameters are frozen after loading the parent's bound trained tensors. Verify parameter versions/absence of gradients and model seals after evaluation. Retain the raw32-token greedy answer and immediate EOS, generated tensors and every native trajectory point.

The parent result and final checkpoint hashes must verify, and the checkpoint optimizer cursor must equal the registered512 budget. Cached condition hashes must equal the parent's runtime record. The clean probe source commit and both interventions are recorded in each output identity. Conditioning hooks restore even if generation raises; the probe rejects a native call that fails to consume the override exactly once.

Compare native CFG1 with native CFG7.5 to isolate guidance; compare raw-cache CFG1 with native CFG1 to isolate conditioning at that guidance. Neither arm modifies the parent or becomes a replacement parent result. A positive result requires new noise and event/state checks before any usable-version claim; the benchmark noises have already been observed in prior diagnostics.

Entrypoint: `scripts/probes/official_base_guidance.py --run <full512-run> --output <new-output> --condition-style native|training_raw`. Each arm uses a separate new output directory and runs only after the parent trainer has exited on the newly allocated single H200. Do not interrupt or duplicate the independent LoRA3500 run.
