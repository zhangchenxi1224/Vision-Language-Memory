# R14 symmetric wrong-donor ranking preregistration

Status: locked after the complete R13 result and before any R14 Reader inference, optimization, or model-dependent outcome.

## Why this is the next minimal experiment

R11 established latent reachability: eight independently optimized VAE latents each produced a human-unreadable image that passed the frozen Reader gate. R12 and R13 then tested a shared event-to-image writer. R13 technically completed and reached 100% normal train-audit accuracy, but only `14/36` train targets passed the complete normal-versus-reset/donor/base causal gate. Reused held-out pass counts were `1/24`, `5/24`, and `6/24`. Wrong-event donor and zero-residual base images remained correct too often.

This rules out “the VAE image space cannot carry a readable code” as the main bottleneck, but it does not establish shared memory. The immediate failure is event attribution: ordinary own-image CE rewards the correct answer without directly penalizing a paired wrong-event image for supporting the same answer.

R14 therefore changes exactly one scientific factor:

> For the same anchor query, target, and choice permutation, train each own image to beat one fixed different-value donor image by a four-choice CE margin.

The R13 writer architecture, exact R12 algebraic initialization, data, schedule, optimizer, endpoint, and causal evaluation remain fixed.

## Fixed symmetric training pairs

The 144 F1 training events contain 36 target values with four events per value. With seed `20260904`, the 36 value groups are deterministically ordered and paired into 18 cross-value group pairs. Members of each paired group are deterministically matched one-to-one. This produces a fixed-point-free involution of all 144 events, or 72 undirected event pairs.

- Every source has exactly one donor.
- Every donor maps back to its source.
- Source and donor always have different target values.
- Reordering the input records cannot change the mapping.
- Canonical pair-list SHA-256: `6d6ae8a27374a505164db1db0be4caeec059f29cf1b95720a20867d30424683c`.

The training pairing is not used for evaluation. Evaluation retains the unchanged R12/R13 split-local `donor_derangement` mapping.

## Fixed writer and information boundary

R14 starts from the same R13 mean-centered conditional residual writer:

`z(e) = fixed_base + 80 * centered_coefficient(e) * unit_basis`

The fixed base is the exact R12 common component (`SHA-256 51fcc191...95fb5`). The trainable residual has exactly zero mean across the fixed 144 training events. The coefficient network and 48 latent bases remain trainable; the event feature extractor, DreamLite VAE, DreamLite UNet/text encoder, and Qwen3-VL Reader remain frozen. The DreamLite denoising UNet is not executed.

The writer receives only frozen features of visible `event_text`. Query text, choices, target index, extracted answer text, segment ID, entity lookup parameters, and per-item latents are forbidden writer inputs. Query and target are used only by the frozen Reader loss, as in R13. Development gradients, auxiliary target classifiers, and checkpoint selection are forbidden.

## Fixed objective and execution

For source event `e`, paired donor `d(e)`, and the identical anchor query/target/permutation:

`L_rank = relu(ln(4) + CE_own - CE_donor)`

`L = CE_own + L_rank + mean(own, donor)[L_residual + L_coefficient]`

The margin `ln(4)` requires `CE_donor >= CE_own + ln(4)` once the hinge is satisfied. Equivalently, the donor probability assigned to the anchor target must be no more than one quarter of the own-image probability and therefore no more than `0.25`. The hinge stops pushing beyond that bounded condition.

- Ranking weight: `1.0`.
- Residual penalty: unchanged `0.10 * relu(residual_rms - 0.50)^2`, averaged over own and donor.
- Coefficient penalty: unchanged `1e-4 * mean(centered_coefficient^2)`, averaged over own and donor.
- Two differentiable frozen-Reader calls per micro-step.
- Exact R12 schedule: 32 epochs, 4,608 micro-steps, gradient accumulation 4, 1,152 AdamW steps.
- Writer and basis LR `1e-3`; network weight decay `1e-4`; basis weight decay `0`.
- No gradient clipping and no adaptive checkpoint selection.
- Raw checkpoints at `0/288/576/864/1152`; raw step 1152 is the sole endpoint.

## Unchanged causal evaluation

Every target in `train_audit`, `dev_select`, `dev_replay`, and `dev_final` is evaluated under all four reverse-cyclic choice views and both `m0` and fixed step-1152 checkpoints. The four endpoint conditions remain:

- `normal`: fixed base plus the target event residual;
- `reset`: unchanged blank latent image;
- `donor`: fixed base plus a fixed different-value event residual;
- `base`: fixed common image with zero conditional residual.

The per-target normal-versus-reset, normal-versus-donor, and normal-versus-base gates are byte-for-byte inherited from R13. The diagnostic arm passes only if the technical gate passes and all target gates pass: `36/36`, `24/24`, `24/24`, and `24/24`.

All four subsets have now been exposed by prior experiments. They are reused only for a controlled objective-only comparison, not as a sealed confirmation. Even an R14 arm pass is only a candidate shared one-SET mechanism and cannot be reported as full Picture Memory success.

## Locked interpretation

| Outcome | Interpretation and next action |
| --- | --- |
| All four splits pass | Ranking resolves the fixed-suite attribution failure. Run a newly sealed fixed-full-data, multi-seed ID/OOD confirmation before any success claim or recurrence work. |
| Train passes, held-out fails | The writer can separate memorized events but does not generalize across entities; improve the conditional event-to-code map. |
| Train fails | Explicit donor credit assignment is insufficient under the centered 48-basis writer; isolate representation, basis capacity, and optimization geometry with the next minimal diagnostic. |
| Donor/base gate fails | Any normal/reset gain is a generic-code or harmful-donor false positive; do not advance. |
| Technical gate fails | Invalid execution; repair and rerun without scientific inference. |

Formal success still requires newly sealed fixed-full-data ID/OOD evaluation, multiple seeds, shared recurrence, SET/OVERWRITE/CLEAR, interference controls, and all causal state controls.
