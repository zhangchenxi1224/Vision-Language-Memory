# R7-GradientBalance-Bottleneck preregistration

## First-principles diagnosis

R6 separated two different optimization problems.  Source anchoring reduced the median
per-segment gradient norm from `4718.65` to `6.59`, the max/min ratio from `594.7` to `25.4`, and
the training clip rate from `100%` to `6.25%`.  Pure-noise redraw was therefore a real source of
gradient-scale pathology.  However, source anchoring still failed the identical hard8 overfit
gate, changed hard8 accuracy by zero, and produced no significant fixed-dev or causal-state
improvement.

The remaining measured bottleneck is aggregation.  One optimizer step currently uses

```text
g_raw = (g_1 + ... + g_8) / 8
```

for eight different state-transition segments.  In the R6 source-anchored audit, `42.9%` of
pairwise gradient cosines were negative.  Raw averaging can therefore cancel desired updates,
while the residual `25.4x` norm spread lets a few segments determine the batch direction.

R7 asks only whether equalizing each segment's directional vote fixes this local bottleneck.  It
does not change the memory task, visual-state architecture, Reader, loss, data, LoRA rank,
learning-rate schedule, clip threshold, checkpoint endpoint, or success criteria.

## Paired aggregation law

Both H200 arms use the corrected R6 source-anchored updater with effective sigma schedule
`[0.5, 0.375, 0.25, 0.125]`.  Both compute all eight micro-gradients independently in the same
deterministic order.

| Arm | Applied pre-clip gradient |
| --- | --- |
| `raw-mean-control` | `g_raw = mean(g_i)` |
| `unit-balanced-norm-matched` | `u = mean(g_i / ||g_i||)`, then `g = u * ||g_raw|| / ||u||` |

The norm match is essential: it keeps the unchanged global clip and optimizer input magnitude
comparable, so the intervention tests **which segments determine update direction**, not whether a
larger or smaller learning signal helps.  Each step records raw/applied cosine, norm-match error,
micro-gradient norm spread, and pairwise conflict.  Non-finite or zero micro-gradients, an empty
aggregate, or relative norm error above `1e-5` fails closed.

The legacy sigma-1 arm is not the R7 base because its source-state coefficient is exactly zero;
it cannot be the preferred recurrent-memory update law even if gradient reweighting lowers CE.
R7 starts from the R6 repair that actually retains the prior latent state and isolates the next
measured failure mode.

## Fixed bottleneck and decision boundary

R7 repeats the exact R6 hard8 selection (SHA-256
`eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`) for 128 optimizer steps
and evaluates each arm against its own M0 on hard8, formal-select-32, mechanism-select-32, reset,
cross-episode swap, and temporal swap.  The hard8 and fixed-dev gates are unchanged from R6.

- If only unit balancing passes hard8, advance source anchoring plus unit balancing to the fixed
  full-data pilot.
- If both fail, unit balancing is insufficient; test deterministic conflict projection on this
  same hard8 before changing rank, data volume, Reader, or loss.
- If both pass, use fixed-dev evidence and the simpler update law to select.
- If only raw mean passes, reject unit balancing.

This is still a one-seed repeated-subset diagnostic and can never be called formal picture-memory
success.  Formal success remains gated on fixed full data, at least two consistent seeds, ID/OOD,
paired endpoint-versus-M0 uncertainty, and reset/swap causal dependence.

The machine-readable frozen contract is
[`configs/experiments/r7_gradient_balance_bottleneck.json`](../configs/experiments/r7_gradient_balance_bottleneck.json).
