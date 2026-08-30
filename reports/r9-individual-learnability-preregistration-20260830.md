# Conditional R9 individual-learnability preregistration

This branch is locked before R8 outcomes and activates only if both technically valid R8 arms fail the unchanged hard8 learnability gate.

## First-principles question

R8 removes every first-order ascent relation between the applied pre-AdamW gradient and the eight current segment gradients. If hard8 still fails, another aggregation rule is not the next informative test. The remaining ambiguity is whether each recurrent transition is learnable by the shared DreamLite updater in isolation, or whether some transitions are not representable/credit-assignable even without simultaneous competitors.

R9 trains eight independent models, one for each fixed hard8 segment. A run sees its target segment once per optimizer step for 128 steps, matching the 128 exposures that the same segment receives in R8. The applied gradient is `g_target / 8`, preserving that segment's coefficient in the R8 raw mean instead of granting an eight-fold gradient advantage. All other architecture, source anchoring, optimizer, schedule, seed, checkpoints, and frozen-model contracts remain unchanged.

## Fixed execution and evaluation

- Fixed hard8 SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`.
- Eight target runs, in the exact R8 segment order; no target selection after outcomes.
- Latent recurrent state, source sigma schedule `[0.5, 0.375, 0.25, 0.125]`, horizon 4, full gradient, rank 4, frozen Qwen3-VL, listwise CE, AdamW, clip 10, EMA, seed 0.
- Exactly 128 optimizer receipts and 128 target-micro receipts per run; checkpoints at 0/32/64/96/128; EMA step128 is primary.
- M0 and EMA step128 evaluate all eight hard8 segments, not only the target, under four reverse-cyclic choice views and `normal`, `reset`, `cross_episode_swap`, and `temporal_swap`. This retains valid swap donors and measures leakage to untrained segments.
- Intermediate checkpoints are descriptive only and cannot replace step128.
- The eight runs are a bottleneck diagnostic and cannot establish fixed-data, multi-seed, ID/OOD success.

## Per-target gate

The technical gate requires finite gradients/weights, exact `1/8` scaling before the unchanged global clip, exact receipt/checkpoint inventories, a frozen Reader/base updater, and complete causal evaluation rows.

The target transition is individually learnable only if all of the following hold at EMA step128:

1. Target normal mean CE falls by at least 20% from its own M0.
2. CE improves in all four fixed choice views. A one-unit bootstrap CI is deliberately not used because it would create false statistical certainty.
3. Target normal accuracy rises by at least 0.25 (one of four fixed views).
4. Target normal/reset difference-in-differences is negative, showing that improvement depends on the recurrent picture state rather than only a changed textual/readout bias.

Non-target changes, all reset/swap cells, clipping, gradient/image statistics, raw rows, checkpoints, logs, hashes, and inventories are reported but cannot rescue a failed target gate.

## Locked interpretation

- `8/8` targets pass: the transitions are individually representable at their original R8 weight; remaining failure is shared-parameter/optimizer interference across updates, not basic per-transition readability. The next paired test must address realized update interference, not rank/data/loss tuning.
- `1–7/8` pass: transition heterogeneity is causal. Compare fixed segment structure/family properties and design the next single-variable repair around the failing property; do not average passing and failing segments into a success.
- `0/8` pass: simultaneous batch aggregation is not the bottleneck. Reopen the recurrent source-anchor/temporal-credit architecture using the existing single-step SET positive control as the lower bound.
- Any technical failure: rerun only the invalid target in a fresh root; make no scientific inference for that target.

If either valid R8 arm passes hard8, R9 is not run; the winning R8 law advances to the already-fixed full-data pilot.

