# Conditional diagnostics for the three-state endpoint

Registered before observing the c90896c-three-state-full1536-20260913 endpoint. The original native CFG7.5 evaluation remains primary. Only if its completed, hash-verified development evaluation contains a failed strict-answer/immediate-EOS cell, run both controls below. A passed primary gate continues into the already queued1e4acd2 fresh-noise confirmation; these diagnostics then remain unused.

Use the exact final1536-step checkpoint, three-state bank, eight paired development noises per state, source images, five Reader questions, FP32 Writer/bf16 Reader, and native28-step denoising. Keep all150 raw rows per control, including repeated blank/donor scoring. Do not train, select another checkpoint, skip a state, or change answer scoring.

1. Native condition encoding with textCFG1/imageCFG1. This changes only text guidance relative to the primary endpoint.
2. Cached raw training condition, textCFG1/imageCFG1, inside the same upstream native denoising loop. All three condition rows receive the same exact cached embeddings/masks; at both scales1 the conditional branch is selected. This isolates the additional native prompt wrapping from the train-time raw-event encoding.

Use the already verified probe code1e4acd2bcd9cbf9f0b2878e51694c17bebe06110, scripts/probes/official_base_guidance.py. The separate supervisor only waits, verifies the parent, and runs these two fixed probes sequentially. It neither changes the running parent nor replaces the primary result. Each probe verifies parent result/checkpoint/model/source bindings, freezes every model, and checks gradients/parameter versions at completion. Deadline06:15; the supervisor terminates only its own new process group on expiry, retaining partial logs.

Interpret by state and question, not aggregate loss. If CFG1 helps, report that intervention explicitly. If only the cached-training arm helps, report the conditioning mismatch; neither outcome retroactively makes native CFG7.5 successful. The next training or deployed inference protocol must state any resulting change and undergo fresh-noise, event, source-transition, and chained-update validation.
