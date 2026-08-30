# R8 CommonDescent Bottleneck preregistration

## Why R8 is necessary

R7 reproduced the corrected source-anchored raw control and then changed only the relative directional vote of the same eight micro-segments. Both arms failed the unchanged hard8 and fixed-dev gates. Unit balancing produced a minimum raw/applied cosine of `-0.5081`, raised clipping from `6.25%` to `32.03%`, and still changed hard8 accuracy by zero. Therefore gradient magnitude dominance is not a sufficient explanation.

The unresolved distinction is between **unequal weighting** and **destructive direction conflict**. Equal-unit averaging fixes the former but can still yield an aggregate with negative dot product against one or more segment gradients. R8 asks whether removing those first-order ascent relationships is enough for the identical eight recurrent state transitions to learn.

## One changed variable

Both arms recompute the same eight gradients in the same order:

```text
raw control:       g_raw = mean(g_i)
projected arm:     d = argmin_v 0.5 ||v-g_raw||^2
                   subject to dot(g_i, v) >= 0 for all i
                   g_applied = d ||g_raw|| / ||d||
```

The projected arm normalizes constraint rows only for numerical conditioning; positive row scaling does not change the feasible half-spaces. Because there are eight micros, the implementation deterministically enumerates all 256 active sets, solves each KKT system in float64, and uses ascending bitmask order as a fixed tie break. It then restores the raw-mean norm before the unchanged clip and AdamW step. This is stronger and less order-dependent than randomized sequential PCGrad.

The claim is deliberately limited: constraints hold in pre-AdamW gradient geometry. Adam moments, preconditioning, weight decay, and finite step size mean this is not a proof that every realized micro loss decreases after the optimizer step.

## Frozen experiment and gates

R8 repeats the exact R6/R7 hard8 (`eeade3e...7828ea6`) for 128 steps with source sigma schedule `[0.5, 0.375, 0.25, 0.125]`, latent state, horizon 4, rank 4, frozen Qwen3-VL, unchanged CE, AdamW, LR schedule, clip 10, EMA endpoint, seed 0, and the same fixed dev suites. The raw arm is rerun rather than borrowed so the shared-engine refactor is itself controlled.

The projected arm fails closed if no feasible KKT solution exists, the projection collapses to zero, any projected micro cosine is below `-1e-5`, norm error exceeds `1e-5`, the intervention never activates, or the 128-step receipt inventory is incomplete. Scientific hard8 and fixed-dev gates are exactly R7.

## Decision boundary

- Projected passes and raw fails: advance the projected law to the fixed full-data pilot.
- Both fail: stop tuning batch aggregation and run an individual-learnability decomposition over the same eight multi-step segments.
- Raw passes and projected fails: reject the projection.
- Both pass: prefer the simpler raw law unless fixed-dev evidence favors projection.
- Any projection technical failure: make no scientific inference; repair and rerun in a new root.

R8 is still a one-seed repeated-subset diagnostic and can never be called formal picture-memory success. The full success contract remains multi-seed, fixed full data, ID/OOD, and reset/swap causal dependence.

The machine-readable contract is [`configs/experiments/r8_common_descent_bottleneck.json`](../configs/experiments/r8_common_descent_bottleneck.json).
