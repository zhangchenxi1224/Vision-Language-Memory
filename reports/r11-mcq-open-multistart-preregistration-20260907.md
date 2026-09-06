# Old R11 paired MCQ / open-answer multistart pilot

Preregistered 2026-09-07 (Asia/Shanghai); configuration only, training not started. Config: `configs/experiments/r11_mcq_open_multistart.json`, SHA-256 `7f280485b2037d0e712c8487dbdc05e85aaba9f71507952198859da559457f55`.

## Fixed scope and activation

Only old R11 direct VAE-latent optimization is included. The pilot target is zero-based **target_index 1**, `r5-f1-392d41fd097d069c42218e0a`: the current music preference of `indigo desk train 001123`, gold `ambient`. This is the Target 1 chosen in the prior research plan; do not choose a different question after replay results.

Start only after stage 1 is technically complete (48 open generations and 32 matched MCQ anchor rows), the old matched **reverse-cyclic4** anchor has been reproduced, and the original model snapshots and device are available. If the technical anchor fails, diagnose and repair before optimization. Activation does not depend on whether stage 1 gives scientifically favorable open-answer accuracy, blank advantage, or donor sensitivity.

The fixed budget is **18 runs, 4,608 Adam steps**: two objective arms × (one blank start + eight random starts) × 256 steps. This is a first pilot, not a claim about every R11 question or an estimate of latent-distribution equality. Existing artifacts remain unchanged.

## Paired starts and objectives

For each seed 0–7, create a separate CPU `torch.Generator` and draw a float32 standard-normal tensor of shape `[1,4,128,128]`. Normalize the draw in float64 to whole-tensor RMS 1. Let `z_ref` be the original common blank-source latent promoted to float32, canonical SHA-256 `719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`. Construct:

`z0(seed) = float32(float64(z_ref) + 0.1 * RMS(float64(z_ref)) * epsilon(seed))`.

RMS uses all 65,536 coordinates. Compute reference RMS once from the common source. Create each initial tensor once, save raw noise, normalized direction and latent plus hashes, and clone exactly the same latent into both arms. Require paired initial tensor hashes to match. Also run each arm from the exact unperturbed blank start, which is a reference case rather than a ninth random sample. Do not orthogonalize directions, select seeds, vary radius, or regenerate a seed after seeing outcomes.

- **MCQ arm:** original choice CE from the four candidates' negative mean token NLL; original forward-cyclic4 views with phase `0` from the old segment-ID hash rule. View index is `(step_zero + phase) % 4`, so every view has exactly 64 training exposures.
- **Open arm:** mean teacher-forced CE for the gold answer under the frozen `original_open` prompt from stage 1. Correct answer tokens are supplied only as supervised continuation; they are absent from the generation prompt. No candidate list, paraphrase training, or explanation target is added. The existing target-only CE convention does not append EOS to the target; generation still uses the model's original EOS policy, and any failure to stop is recorded.

Both arms use Adam with constant LR 0.05, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay 0, no gradient clipping and exactly 256 updates. The only trainable object is an unconstrained float32 model-space latent. VAE and Qwen Reader remain frozen in eval mode, with the original VAE scale/shift, compute dtype, unit-RGB clamp and stage-1-validated reader preprocessing. No DreamLite UNet is used. Adam starts empty for every run; seed 0 and identical Python/NumPy/Torch CPU/CUDA RNG state are restored before each arm. The fixed run order alternates which arm runs first in successive pairs.

## Technical controls and trajectory records

Before optimization, verify paired initial hashes and repeat the same-objective forward/gradient computation from the same RNG state. Record equality or maximum absolute difference, model/processor snapshots, environment, determinism settings and device. These checks establish input and computation fidelity; **they do not estimate accumulated trajectory repeatability**. There are no duplicate full blank trajectories, so no numerical trajectory-noise floor or empirical same-point threshold may be claimed.

Save each raw float32 latent at steps 0–256 (256 KiB tensor payload per step), plus full Adam and RNG checkpoints at 0/64/128/192/256. Log training loss, gradient norm/RMS/nonzero fraction, actual parameter-update norm/RMS, latent RMS and distances to own start and shared blank reference at every step. A resume must restore latent, Adam, step/view position and RNG together.

At fixed steps **0/64/128/192/256**, both arms receive no-gradient, RNG-preserving probes of all four **forward-cyclic** training views and the fixed original-open token CE. Probe frequency is fixed prospectively for this pilot and will not be increased or reduced based on observed trajectories. Generation is evaluated only at step 256. Raw latent traces permit one-step and four-step movement analysis without extra model calls. There is no early stopping, best-checkpoint selection or seed removal for poor scientific performance. Nonfinite or otherwise technically invalid runs remain logged as failures; any repair/rerun is separately identified.

## Fixed endpoint evaluation

The primary endpoint is raw **step 256** for every run. Each arm/start receives both stage-1 prompts (`original_open`, `paraphrase_open`) under matched endpoint, common blank and fixed donor image conditions. The donor is always the old **target_index 2** endpoint (`orange`, `r5-f1-1aee01c0f3e7684c05c9122c`), already fixed by the first-stage ring rule, decoded through the same VAE. It is an image-corruption intervention with different entity/attribute content, not a known counterfactual state for this question.

Greedy generation uses `do_sample=false`, `max_new_tokens=32` and the original model EOS policy. Save exact serialized prompts and input/generated token IDs, raw output with and without special-token removal, output length and stop/truncation reason. Gold, aliases and answer prefixes are absent from generation input. Score with the same frozen normalized exact match as stage 1: casefold, collapse/trim whitespace, and repeatedly remove trailing whitespace and Unicode punctuation only. Aliases remain empty; no keyword extraction or semantic relabeling replaces the primary score.

All 18 endpoints also receive the original four **reverse-cyclic** MCQ views: `[[3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1]]`. Preserve all candidate scores, CE, margins and correctness. Report matched original-open exact match as primary open success; correctness on both open prompt variants as secondary robustness; and correctness on all four reverse-cyclic views as MCQ success. Report these outcomes separately, with fixed blank/donor contrasts. Do not compare raw numerical training losses across the two different objectives.

This yields 108 endpoint open generations, 72 endpoint MCQ-view evaluations, 360 fixed forward-view probe evaluations and 90 fixed open-CE probes. Repeated blank/donor evaluations are technical repeats of fixed controls and do not increase the number of independent starts.

## Behavior and geometry analysis

The independent initialization unit is one of **eight random directions**, paired across arms. Show each pair's open and MCQ success, and success counts out of eight per arm. Report blank-start results separately. Primary geometry includes all eight random runs; any successful-only geometry is clearly secondary to avoid outcome-dependent sample selection. The 28 within-arm pairs share eight trajectories and are not 28 independent samples; latent coordinates, steps, prompts and views are not extra replicates.

For every saved step and fixed endpoint, report latent RMS/L2; RMSE and cosine relative to shared `z_ref`, relative to own `z0`, and between independent-start trajectories; update residuals `z(t)-z0`; shared-reference residuals `z(t)-z_ref`; and paired cross-arm endpoint distances and cosines. The primary within-arm dispersion is mean pairwise squared RMSE over the eight random starts, normalized by its step-0 value. Also show each pair's distance ratio to its own nonzero initial distance, distance from the same-arm blank trajectory, initial-versus-final distance-matrix association, and one-step/four-step update distances. These are descriptive measurements, not distribution-equality tests. Do not use a rising latent norm to define artificial contraction, and do not equate high raw-latent cosine with matching update directions.

The pilot asks whether changing the objective improves direct answer generation, whether image interventions change those answers, and whether different starts contract, preserve differences or form multiple effective endpoints. Similar loss, matching outputs, small geometric distance and a stationary optimizer are different statements. With eight directions and no trajectory-noise-floor control, report measured distances and patterns rather than declaring one unique solution, exact basin identity or equal latent distributions. **No MMD test** is planned. Further questions, radius sweeps, interpolation and multi-question training require a subsequent prospectively defined extension.
