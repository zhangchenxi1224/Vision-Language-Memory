# Frozen DreamLite input optimization: fresh Open + EOS campaign

This isolated protocol replaces MCQ computation after the user's explicit migration request. The historical MCQ source and all artifacts remain unchanged. None of their completions count toward the new 158-run campaign.

Only FP32 `x_T` is optimized. Frozen FP32 DreamLite condition encoder, U-Net, scheduler and VAE remain in the differentiable path; the frozen Reader is BF16. Adam uses LR 0.05 and exactly 256 updates. Every initial and final `x_T` and `z` is saved, with 11 RGB/optimizer checkpoints.

The original eight locked F1 examples are converted to fill-in questions by removing the MCQ instruction and never providing answer choices to the Reader. The supervision is mean answer-token CE plus one assistant termination-token CE, weight 1. The termination token is derived from the actual chat template and verified against generation stopping IDs. Only the original question is used for optimization. Five evaluation questions preserve category, entity and the exact original temporal condition, followed by:

```
Use the memory image to answer.
Answer with a short phrase only.
```

The primary success criterion is raw step-256 original-question greedy generation (32-token ceiling) exact match. Prefix correctness, overgeneration and all-five-question consistency remain separate metrics. Blank and SHA-bound historical other-answer donor latents are evaluation controls only; no historical endpoint initializes training. The donor is an MCQ-era control and its Open accuracy is measured, never assumed.

The 158-run preregistration retains six independent full-trajectory reproducibility repeats, Gaussian density, distribution, scale and eight-question panels. Successful final latents form candidate supervision banks; the separately authorized downstream U-Net stage must consume verified EOS artifacts, not the historical MCQ bank. This runner only produces oracle artifacts.

Validation: 20 CPU tests passed; both new Python entrypoints compile and the launcher passes `bash -n`. A real H200 two-repeat full-path gradient probe is mandatory before fresh optimization. CPU checks alone are not training evidence.

Deployment uses the already allocated `vlm-oracle-geometry-h200x4-20260908-r02` Job, with GPU pairs 0/1 and 2/3. Historical campaign PID 54 is paused, and its two MCQ workers were terminated after command identity verification; all four GPUs then reported zero memory and utilization. The new EOS process has an isolated source checkout, output root and commit. Keeping the old main process paused retains the allocation for the separately authorized bank/U-Net continuation. It must never be resumed; release the Job through the official platform stop action after the full chain finishes.
