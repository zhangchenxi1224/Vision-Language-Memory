# Same-question three-state target construction

Registered before dispatch. Full-U-Net512 fitted ambient robustly over16 new noises but returned ambient on all jazz/clear interventions. A shared Writer needs multiple conditional states; adding single-target steps cannot test that learning.

Keep the original entity, slot, five Reader question variants and model snapshots. Construct three conditional groups for **one semantic question**, not three independent questions: ambient, jazz and clear/no active preference. The Writer will receive the event and source image only; no Reader query or separately supplied gold label.

Reuse the hash-selected, already verified ambient endpoint. For jazz and clear, independently optimize exactly one FP32 model-space latent through frozen FP32 VAE and bf16 Reader. Use Adam lr0.05, betas(.9,.999), epsilon1e-8, no weight decay,256 updates, loss mean answer CE plus EOS CE with weight1. Cycle original_open/paraphrase_1/paraphrase_2 on zero-based steps:86/85/85 updates. Hold paraphrase_3/4 out of all gradient calls. This restores the explicit recipe in historical commit46cd36b; it is not three gradients accumulated at each update.

Both fresh states start from the archived initial latent of the previously hash-selected ambient teacher, Gaussian seed7/scale0.5. Verify the original manifest, latent-index file, step0 file and tensor hashes from the sealed bank provenance. This fixes initialization before observing either new state's result. It does not start from an optimized ambient endpoint or choose among new seeds after evaluating.

Use the two idle H200s on dl-align-h200x2-20260913-r2, VAE cuda:0 and Reader cuda:1, unchanged model snapshots, strict deterministic CUDA environment. The runtime loads and freezes the remaining official Base components for model identity checks; the oracle optimizer contains only the latent, and never executes the U-Net. No trained Writer checkpoint is loaded. Deadline1789254021 precedes this notebook's expiry.

Retain all257 latent states, fixed intermediate checkpoints, complete optimizer metrics, reproducibility gradient checks, raw greedy32-token generations, controls and EOS diagnostics. Reread the reused ambient endpoint under this runtime as well. A bank can be sealed only if allthree state targets pass allfive question variants with immediate EOS. On any failure retain every artifact and leave the bank unsealed; do not replace a failed seed or select an intermediate checkpoint as the registered endpoint.

The exported bank records one semantic question and three state groups, shares the original source/control provenance, and binds every raw FP32 target tensor and generation file by SHA256. This remains target construction, not Writer success. After successful sealing, a separately registered full-U-Net official-FM run must learn allstates jointly and pass paired conditional and fresh-noise tests.
