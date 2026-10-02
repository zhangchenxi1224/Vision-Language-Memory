# Base official FM budget control

Recorded before dispatch, 2026-09-13. This is a scientific diagnostic within the active usability goal.

## Evidence and hypothesis

Mobile and native Base with matched rank16/512-update training both yielded 0/8 on every question variant. Base CFG=1 at the same checkpoint also yielded 0/8 on every variant. The selected oracle survived all three RMS0.03 isotropic perturbations on original and paraphrase_4, and teacher-to-generated interpolation through fraction0.3. Therefore the evidence does not support either extreme fragility or default CFG as sufficient explanations. Test whether the 512-update pilot underfits relative to the upstream example's 3500-update budget.

## Fixed run

- Exact training checkout remains `ac34ab20b50fae82d56d1aef13fc995ee35c302a`; no new training implementation.
- New output `P/runs/dreamlite-official-alignment/ac34ab2-base3500-single-20260913`.
- Fresh initialization from the same verified Base checkpoint, single hash-selected teacher, seed20260913, rank16/alpha16, lr5e-5, accum4, weight_decay1e-4, clip1; **3500** optimizer updates,14,000 training draws. No continuation masquerading as the old512 run, no post-hoc best checkpoint.
- Original raw event/source conditioning, official target-noise FM on the full sigma range, native28-step inference with defaultCFG7.5 and imageCFG1, FP32 Writer/bf16 frozen Reader.
- Identical bank SHA `20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722`, complete Base seal, official source a6e20c8, same separate-H200 placement. Worker deadline1789254021, before notebook expiry.
- Same8 benchmark noise seeds and5 prompts plus blank/donor; greedy32 tokens, raw answer and immediately following EOS. These benchmark seeds were observed in prior diagnostics; they are unseen by optimization, but no longer untouched research holdouts. Any positive result must be followed with newly fixed seeds and event/counterfactual tests before usability claims.
- Checkpoints include optimizer/RNG each update. Technical completion requires all3500 metrics and verified result/checkpoint hashes. Nominal runtime about100minutes including load/evaluation at observed1.53seconds per update; not a timeout guarantee.

## Interpretation

Compare endpoint raw answer+EOS, per-seed behavior and latent error, not unpaired loss alone. First512 draw identities must match the512 pilot; first512 losses will also be compared as a determinism sanity check. Zero functional gain does not settle all capacity questions but rejects this specific budget increase as sufficient. A positive single-question endpoint is a mechanism result, not event-conditioned generalization.

`P=/inspire/ssd/project/exploration-topic/czxs26210936`.
