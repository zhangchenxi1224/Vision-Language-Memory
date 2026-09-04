# R15 synchronous complete-round-robin donor preregistration

Status: locked after the complete R14 result and after a model-free schedule audit, but before any R15 Reader inference, optimization, checkpoint, image, or model-dependent outcome.

## Scientific question

R11 proved latent reachability: separately optimizing one VAE latent for each of eight fixed targets produced machine-readable, human-unreadable visual codes. R12--R14 did not establish a shared writer. R14 was technically valid but passed the full causal target gate for only `17/36`, `1/24`, `4/24`, and `7/24` targets on train-audit, dev-select, dev-replay, and dev-final.

R14 nevertheless learned its fixed training contrast: over the last 64 directional micros, own CE was `0.6153`, donor CE was `2.9833`, and `60.94%` satisfied the `ln(4)` margin. The delivered audit then measured two defects in that contrast estimator:

1. only `44/2304 = 1.91%` of bidirectional pair/epoch instances shared an optimizer update; the median opposite-direction lag was 11 updates;
2. every event saw exactly one fixed wrong value in all 32 epochs, while the train-audit evaluation donor identity and donor target-value overlap were both `0/36`.

R15 asks one bounded question:

> Holding the writer, exact initialization, formal data, optimizer, fixed endpoint, Reader/VAE, and all normal/reset/donor/base gates fixed, does a complete update-synchronous contrast estimator fit and generalize the one-SET visual-memory boundary?

The corrective package has two inseparable components: both directions are evaluated under one parameter snapshot, and every event covers every wrong target value. R15 is not designed to attribute a gain to either component separately.

## Fixed source, data, and initialization

- R14 delivered-results commit: `066c44c2c530fba323f0edd76954245cbe91505e`.
- R14 training-source commit: `b0e7abc9ce737ed0d80832c56ff773b187bb47ad`.
- R14 complete archive SHA-256: `41669e4d3c5395b7227d9e9b599d4e679a629a1bf6cd9fbe152d3fba7c4ac138`.
- Formal train SHA-256: `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`.
- Formal dev SHA-256: `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303`.
- Fixed train selection: 144 F1 events, 36 target values, four members/value; payload SHA-256 `07ce170b749b1394550cce6a8e0e9586d3110180d94104a8646de2c222936d62`.
- Train-audit/dev-select/dev-replay/dev-final payload SHA-256 values remain `a6de7819...abc4f`, `e744484c...901c`, `0bd95da0...f0b8`, and `87c72875...e07c6`.
- The writer starts from the exact R12-conditioned step-1152 algebraic decomposition used at R13/R14 step zero. It must not start from the R14 endpoint.
- Initial latent SHA-256 remains `719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`.
- Fixed base latent SHA-256 remains `51fcc191ac3914ebdbfe07a914d95a51c20458de9a4352de999d7cb8c3595fb5`.

The R13/R14 centered writer is unchanged: frozen 2048-dimensional event features feed the trainable `2048-512-48` coefficient MLP; coefficients are bounded by `2*tanh(raw/2)`, centered over the fixed 144 training events, and projected through 48 unit-normalized latent bases with output norm 80. The DreamLite VAE and Qwen3-VL Reader remain frozen. DreamLite UNet/text denoising is not executed.

The writer receives only frozen features of visible event text. Query, choices, target index, extracted answer, segment/entity ID, round/pair/member index, and per-item trainable lookup state are forbidden writer inputs. Training labels are used only to construct supervised cross-value negatives and frozen-Reader losses.

## Locked complete round-robin schedule

The schedule seed is `20260905`; F1 pool construction retains the R14 seed `20260904`. Target groups and the four members inside each group are independently SHA-256 ordered, so input-record order cannot affect the schedule.

1. The first 35 rounds are a deterministic one-factorization of `K_36`: all `C(36,2)=630` unordered target-value pairs occur exactly once.
2. Each round contains 18 disjoint target-value pairs. Each value pair is expanded into four event pairs, yielding 72 atomic pair micros per round.
3. Round 36 repeats the target-value matching of round 1 but changes member matching by `+1`; all 72 event-pair identities therefore differ from round 1.
4. The 36 member shifts are exactly:

   `0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,2,3,1`.

5. Odd shifts reverse sign at the two ends of a pair. Their 18 perfect matchings are therefore oriented by a deterministic Euler decomposition so every target receives effective member offsets `+1` and `-1` exactly nine times. Consequently every event faces donor member ranks 0/1/2/3 exactly nine times.
6. Every event occurs once per round, sees all 35 wrong target values, and sees 36 distinct donor event identities.
7. The unchanged forward-cyclic training view is `(round + SHA256(segment_id)[:2] mod 4) mod 4`; every event receives each of four views exactly nine times.
8. Pair micros are member-major within a round. Every optimizer update contains exactly two complete pairs from two different value-pairs, hence four distinct events and four distinct target values. Accumulation never crosses a round boundary.

Canonical full-receipt schedule SHA-256:

`2495ce15ed5242b88f1d88b6caed29694a6340a9982c3870b1720004cf75ffb8`

This was calculated twice byte-identically from the formal train file by source-only commit `97bfc8338912ba58eccc29e46c1fc02ba00bb918`. The receipt contains 2,592 pair rows, is 1,640,429 bytes, has artifact SHA-256 `5ac6c3fea168ecc8f0b265d5200004a6233b8ef45387921b7d26e4bd1d8f02b2`, and explicitly records `model_calls=0` and `reader_calls_executed=0`.

## Atomic objective and optimizer boundary

For one pair `(A,B)`, generate images `I_A` and `I_B` once from the same writer parameter state. Before any optimizer update, execute exactly four differentiable Reader calls:

- `CE_AA = CE(q_A, I_A, y_A)`;
- `CE_AB = CE(q_A, I_B, y_A)` using exactly the same A query, target, and permutation;
- `CE_BB = CE(q_B, I_B, y_B)`;
- `CE_BA = CE(q_B, I_A, y_B)` using exactly the same B query, target, and permutation.

Define:

`H_A = relu(ln(4) + CE_AA - CE_AB)`

`H_B = relu(ln(4) + CE_BB - CE_BA)`

For image `i`, `R_i` is the unchanged residual-RMS penalty plus coefficient-L2 penalty:

`R_i = 0.10 * relu(rms(delta_i) - 0.50)^2 + 1e-4 * mean(centered_coefficient_i^2)`.

The sole pair sum is:

`S_AB = CE_AA + CE_BB + H_A + H_B + R_A + R_B`.

Execution is locked to:

```text
(S_AB / 4).backward()
accumulate exactly two complete pairs
optimizer.step()
```

There is no additional division by two. This exactly preserves R14's average scale over four directional objectives per update. `I_A` and `I_B` must be reused across their two Reader calls without detach, stale-image cache, or parameter update between directions.

## Fixed optimization and compute disclosure

- AdamW; writer/basis LR `1e-3`; network weight decay `1e-4`; basis weight decay `0`; constant schedule.
- No gradient clipping and no early stopping.
- 36 rounds, 2,592 pair micros, 5,184 directional examples, 10,368 Reader calls, and 1,296 optimizer steps.
- Exactly four Reader calls/pair and eight Reader calls/update.
- Checkpoints only at steps `0/324/648/972/1296`; raw step 1296 is the sole scientific endpoint.
- Seed 0 and strict deterministic execution.

Reader calls, directional examples, and optimizer steps are 12.5% greater than R14. R15 is therefore not compute-matched, and any gain cannot be uniquely attributed to synchronization or negative coverage. Pair-level image reuse also means total writer/VAE forwards are not described by that 12.5% figure.

## Unchanged causal evaluation

The four R14 evaluation subsets, all four reverse-cyclic views, `m0`, fixed final endpoint, and four conditions remain unchanged:

- `normal`: fixed base plus the target event residual;
- `reset`: unchanged blank-latent image;
- `donor`: fixed base plus the same deterministic different-value evaluation donor residual used by R14;
- `base`: fixed common visual base with zero conditional residual.

Reverse-view permutation SHA-256 is `4cd725a443d8661dccccbff2d714876aee317c654ef2aa1bcb79aee307d64bbd`. The exact R14 evaluation donor-map SHA-256 values are:

| split | items | donor-map SHA-256 |
| --- | ---: | --- |
| train-audit | 36 | `6516ac458dc85e26413eff6792dacc556c584bf5395c1bad6b980e7ea0dd4ef3` |
| dev-select | 24 | `39696d988be0ea79bf48483757fa2b99438de2f27f0bc17fd14c8e0d17665c1f` |
| dev-replay | 24 | `fe3bf8163fd75e075af4cb4d455ae7d2408230af8f59734b165f8833eb9a0aa5` |
| dev-final | 24 | `4dc7306af7d4df9fa4c4edb9536357270e10b65206f49dcf6f11d9936be370ee` |

All 12 per-target thresholds remain byte-for-byte identical to R14: four normal-vs-m0 gates, four normal-vs-donor gates, and four normal-vs-base gates. The diagnostic arm passes only with a technical pass and target pass counts `36/36`, `24/24`, `24/24`, and `24/24`.

All current dev subsets have been exposed by R12--R14. No R15 development gradient, checkpoint selection, schedule tuning, or member-orientation selection may use their outcomes. Even an all-pass R15 result is only a fixed-suite candidate mechanism; `formal_success_claim` must remain false.

## Fail-closed technical gate

Before scientific interpretation, the run must prove all of the following from its own artifacts:

- exact source commit, clean tree, config/preregistration/data/model-parent hashes, R12 initialization reconstruction, and fixed-base hash;
- exact schedule SHA above; 630 unique value pairs in rounds 1--35; all 35 wrong values and 36 donor identities/event; member ranks and training views each `9/9/9/9`;
- exactly 2,592 pair micros, 5,184 directions, 10,368 Reader calls, 1,296 optimizer steps, and checkpoint set `0/324/648/972/1296`;
- every pair has four calls from one parameter snapshot and every update has two whole, different value-pairs/four events/four values, with no cross-round accumulation;
- own/donor calls share each anchor's query, target, and permutation; donor CE uses the anchor target rather than the donor target;
- recomputed `H_A`, `H_B`, `R_A`, `R_B`, `S_AB`, and `S_AB/4`, with no second accumulation divisor;
- all four image-to-Reader gradient paths are connected; no image detach/cache or between-direction update;
- finite nonzero unclipped gradients; fixed feature extractor, VAE, Reader, UNet, and text encoder snapshots unchanged;
- complete endpoint evaluation rows for every split/checkpoint/condition/view; exact evaluation donor-map and reverse-view hashes above.

Any failed or missing check makes the execution technically invalid and permits only repair/rerun, not scientific inference.

## Locked interpretation

| Outcome | Interpretation and next action |
| --- | --- |
| All four splits pass | The complete synchronous contrast estimator resolves the fixed-suite one-SET attribution boundary. Treat only as a candidate; immediately run newly sealed fixed-full-data ID/OOD, multiple seeds, and then recurrence/overwrite/clear/interference confirmation. |
| Train passes, held-out fails | The writer can separate fixed events but the frozen event representation or shared event-to-code map does not generalize. Diagnose representation before adding recurrence. |
| Train fails | Complete pair credit assignment and wrong-value coverage are insufficient under the centered 48-basis writer. Isolate representation, basis capacity, and optimizer geometry with the next preregistered minimal test. |
| Donor/base gate fails | Apparent normal/reset gain is a generic-code or donor false positive; do not advance. |
| Technical gate fails | Invalid execution; repair and rerun with no scientific conclusion. |

Formal Picture Memory success still requires a newly sealed fixed-full-data ID/OOD protocol, multiple seeds, shared recurrent state transitions, SET/OVERWRITE/CLEAR, interference controls, and causal state controls.
