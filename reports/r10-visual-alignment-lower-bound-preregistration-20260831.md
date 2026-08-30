# Conditional R10 visual-alignment lower-bound preregistration

Status: written after valid R9 targets 0--2 had failed and before endpoint outcomes for targets 3--7 were observed. The partial peek is disclosed explicitly. Target selection, arms, gates, and interpretation below are fixed independently of the remaining outcomes.

## Why this is the next main-line question

Picture memory requires three nested capabilities:

1. the frozen VLM must have an image-space direction that changes the desired answer;
2. DreamLite must be able to write such a direction from one visible event;
3. recurrent updates must preserve and compose multiple writes.

R5--R9 tested increasingly refined versions of capability 3 while assuming capability 2 had already been demonstrated. The evidence audit in `single-step-set-evidence-audit-20260831.md` shows that assumption is unsupported: the archived SET-only fixed endpoint worsened by 31.12%, while its post-hoc best intermediate checkpoint improved only transiently. Nonzero gradients prove differentiability, not learnability.

R10 therefore separates the two lower levels before another recurrent-memory modification. This follows the meeting decision to make image--text alignment work on a small corpus before introducing memory-harness behavior.

## Activation

- If all eight technically valid R9 targets pass, skip R10 and follow R9's locked realized-update-interference branch.
- If zero to seven R9 targets pass, activate R10 unchanged.
- A technical R9 failure must be rerun before counting the pass total.

R10 cannot be used to reinterpret or rescue R9.

## Immutable F1 targets

The formal-v1 train pool contains 7,504 F1 `SET -> QUERY` segments. With seed `20260831`, segments are sorted by

`sha256("R10-VisualAlignment-LowerBound\x1f20260831\x1fF1\x1f" + segment_id)`

and then by `segment_id`. The first eight are fixed:

1. `r5-f1-8015bf53a4067aaa7e882288`
2. `r5-f1-392d41fd097d069c42218e0a`
3. `r5-f1-1aee01c0f3e7684c05c9122c`
4. `r5-f1-807090710dd4c077a97348ba`
5. `r5-f1-02c8aa9dc3523351c4d5f9c7`
6. `r5-f1-11dddf29efebc033a995ab92`
7. `r5-f1-de4899e05b0aa7f3d8373171`
8. `r5-f1-850189e424efde62468b2ef9`

Canonical selected-payload SHA-256: `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845`.

No target may be replaced based on M0 difficulty or outcome.

## Two arms, one causal distinction

### A. Direct-pixel oracle

For each target independently, optimize a 1024x1024 RGB image through the unchanged differentiable Qwen Reader. The image is `sigmoid(logits)`, initialized to the exact uniform `127/255` blank state. Adam uses LR `0.05`, no weight decay, for 128 fixed steps. The four reverse-cyclic choice views each occur exactly 32 times. Raw step128 is the sole endpoint.

This arm answers only: **does a readable image code exist under the current Reader, preprocessing, and listwise loss?** It sees the answer through the training loss and is therefore a capacity oracle, not a memory model or success result.

### B. DreamLite single-SET

For the same target independently, train only DreamLite-mobile U-Net rank-4 LoRA. Keep the R6/R9 source-anchored latent update, four effective sigmas `[0.5, 0.375, 0.25, 0.125]`, frozen Qwen, full gradient, AdamW, clip 10, weight decay `1e-4`, unchanged R5 LR prefix, and EMA `0.995`. Apply the target gradient at coefficient `1.0`; this is a base-learnability upper-bound test, not an R9 coefficient-matched comparison. EMA step128 is the sole endpoint.

The target is trained across the same four choice views, exactly 32 exposures each.

## Fixed gates

Each target in each arm passes only if all conditions hold:

1. complete technical receipts and checkpoints at 0/32/64/96/128;
2. endpoint normal CE improves by at least 20% relative to its own M0;
3. all four fixed choice views have lower CE;
4. accuracy increases by at least 0.25;
5. the normal/reset difference-in-differences is negative.

An arm passes only at `8/8`. Any smaller count is diagnostic, never partial success. Intermediate checkpoints are descriptive and cannot rescue step128.

## Locked interpretation

| Pixel oracle | DreamLite | Conclusion and next experiment |
| --- | --- | --- |
| 8/8 | 8/8 | Frozen visual readout and one-step DreamLite write are supported. Next run shared multi-item F1 training with held-out dev; do not jump directly to recurrence. |
| 8/8 | 0--7/8 | The image channel is usable; DreamLite's update parameterization, conditioning, or optimizer is the bottleneck. Change only the updater side. |
| 1--7/8 | any | Image-code learnability is target-dependent. Attribute failure to fixed query/option/token properties before a general claim. |
| 0/8 | any | The current Reader/loss/preprocessing interface has not demonstrated a pixel code. Test a post-resize pixel or visual-token oracle before changing recurrence. |
| technical failure | — | Rerun only invalid targets in fresh roots. |

## Success boundary

R10 is a one-seed repeated-target bottleneck diagnostic. Even 8/8 in both arms does not establish picture-memory success, generalization, or closed-model transfer. Formal success still requires a shared learned updater, held-out ID/OOD data, multiple seeds, fixed endpoints, and causal state intervention.

The machine-readable contract is `configs/experiments/r10_visual_alignment_lower_bound.json`.
