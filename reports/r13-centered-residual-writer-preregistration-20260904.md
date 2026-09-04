# R13 mean-centered conditional residual writer preregistration

Status: locked after the complete R12 paired result and source-only decomposition audit, before any R13 model execution or fresh-final Reader output.

## Why R13 is the next minimal experiment

R11 proved that independently optimized VAE latents can carry machine-readable memory (`8/8`). R12 then trained one shared event-to-latent writer. Its engineering chain was valid, and normal images sharply improved over blank reset images, but normal and wrong-event donor images were almost interchangeable. The matched constant-event arm reached similar loss. Post-hoc localization showed that the frozen event representation retained target-value information (83.3%/95.8% linear-probe accuracy on the two held-out R12 subsets), while the learned coefficients fell to 20.8%/20.8%. The event coefficients were almost parallel to one common vector: their mean event-specific residual was only about 4.7% of total coefficient norm.

The smallest causal intervention is therefore not another learning-rate, clipping, LoRA-rank, or diffusion-step sweep. R13 removes only the identified shortcut:

> Freeze the R12 common visual code and permit the trainable branch to write only a residual whose exact mean over all fixed training events is zero.

If ordinary Reader CE can then learn normal images that beat reset, wrong-event donor, and the frozen base on every view, event-specific shared writing is established. If it cannot fit even train-audit, the next justified test is an explicit symmetric donor-ranking objective; recurrence remains premature.

## Exact R12 decomposition

The R12 conditioned step-1152 writer is decomposed algebraically. Let `c(e)` be its bounded event coefficient, `B` its unit-normalized latent basis, and `z0` the blank VAE latent. Over the fixed 144 training events:

`c_bar = mean_train c(e)`

`z_base = z0 + 80 * c_bar * B`

`z_R13(e) = z_base + 80 * (c(e) - mean_train c(e)) * B`

At R13 step 0, `z_R13(e)` reconstructs the R12 conditioned endpoint for every source event up to a preregistered maximum absolute tolerance of `1e-5`. The 144-event reduction order is fixed to canonical segment-ID order so floating-point hashes are platform-stable. The common base is then frozen. During training, the coefficient network and basis may change, but `mean_train(c(e)-mean_train c(e))` is recomputed differentiably on every forward and must remain within `1e-6`; its projected mean latent residual must remain within `1e-5`. Thus neither parameter block can move the average training image.

Source-only preregistration hashes were derived inside the locked formal H200 runtime (`torch 2.7.0a0+ecf3bae40a.nv25.02`):

- R12 conditioned step-1152 checkpoint: `d34091ed...067f9`
- R12 event cache: `5cfc2380...6f49c`
- R12 train features: `2532c503...fe865`
- source mean coefficients: `0a4106c7...e2ae`
- common latent delta: `24f6c455...672a`
- frozen base latent: `11281993...25e8`

No Reader inference or R13 optimization was used to derive these hashes.

## Data and information boundary

R13 retains the exact R12 F1 train set (144 events), train-audit subset (36), and exposed dev-select set (24). The former R12 sealed final has now been observed, so it is honestly reclassified as `dev_replay` (24) rather than reused as a sealed claim.

A new `dev_final` contains 24 previously unexposed entities, one for each dev target value and exactly `6/6/6/6` target positions. It excludes every train, dev-select, and dev-replay entity. Selection used structural IDs only with seed `20260904`; payload SHA-256 is `87c72875...e07c6`. It remains unread by the model until the fixed endpoint.

The writer receives only visible `event_text`, encoded by the frozen R12 endpoint LayerNorm and attention pooling. Query text, choices, target index, extracted answer text, segment ID, entity lookup parameters, and per-item latents are forbidden. Only frozen Reader CE may use the associated query and target during training. No auxiliary target classifier, development gradient, or best-checkpoint selection is allowed.

## Fixed training

- Exact R12 schedule: 32 epochs × 144 events = 4,608 micro-forwards; accumulation 4; 1,152 AdamW steps.
- Each event receives every forward-cyclic choice view exactly eight times.
- LR `1e-3`, network weight decay `1e-4`, basis weight decay `0`, constant schedule.
- Objective: Reader CE + the unchanged stable `0.10·relu(residual_rms−0.50)^2` penalty + `1e-4·mean(centered_coefficient²)`.
- No gradient clipping and no backward loss divisor.
- Raw checkpoints at `0/288/576/864/1152`; raw step 1152 is the sole endpoint.

## Causal evaluation and success gate

All four reverse-cyclic views are evaluated for every target under:

- `normal`: frozen base plus its own event residual;
- `reset`: unchanged blank image;
- `donor`: frozen base plus a fixed different-value event residual;
- `base`: frozen common image with zero residual.

The R12 normal/reset/donor gate is unchanged: at least 20% relative CE improvement, all four views improve, accuracy increases by at least 0.25, and difference-in-differences is negative. R13 adds the same-strength normal-versus-base gate, preventing a false pass where the frozen generic image supplies all improvement or donor residuals merely make predictions worse.

The diagnostic arm passes only with technical validity and exact target counts `36/36 train-audit`, `24/24 dev-select`, `24/24 dev-replay`, and `24/24 fresh dev-final`. Partial counts are diagnostic only. Even a complete R13 pass establishes shared one-SET writing, not full Picture Memory success; recurrent SET/OVERWRITE/CLEAR, interference, fixed full ID/OOD evaluation, and multiple seeds remain mandatory.

## Locked interpretation

| Outcome | Interpretation and next action |
| --- | --- |
| All four splits pass | The R12 generic-code shortcut is removed and causal shared one-SET writing generalizes to fresh entities. Advance to recurrent state algebra. |
| Train passes, held-out fails | Conditional mapping does not generalize; repair event-to-code mapping before recurrence. |
| Train fails | Ordinary CE is insufficient under the no-common-code constraint; test symmetric donor-ranking credit assignment next, keeping the evaluation unchanged. |
| Reset improves but donor/base gates fail | Generic prior or harmful-donor artifact; false positive, no advance. |
| Technical gate fails | Invalid execution; repair and rerun without scientific inference. |
