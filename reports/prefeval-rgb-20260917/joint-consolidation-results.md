# Plan08 joint consolidation: completed, adoption failed

Execution commit `89ea585f36f53808dc0f3e77f5d8b59259739468`; registration
`f41aa88a39fd498813e672c62a036eaf9683e3c25ced1623c1fe3198fd28875e`.
All GPU work ran on `dl-clear-retain-h200x4-20260914`. All 80 endpoints completed
64 additional updates each and froze before evaluation. No Writer updates.

The joint objective restored training-form recovery while retaining application
ranking. It improved joint capability relative to recovery-only repair, but
query transfer, complete multi-slot recovery, selective clear and original MCQs
remain below adoption requirements. This is an independent latent-teacher trial,
not an available shared DreamLite Writer.

## Paired endpoint evidence

B is the archived Plan07 initialization; C repairs all three recovery forms at
every update; J adds .25 times the existing ranking objective on the same image
before the same optimizer step. Fresh Adam, LR .01, unchanged populations and
scoring. Every read for a state uses its one saved/reopened PNG.

| Metric | Plan07 B | C | J |
|---|---:|---:|---:|
| Training recovery | 186/264 | 262/264 | 260/264 |
| Gradient-excluded recovery | 83/176 | 121/176 | 116/176 |
| Original MCQ | 45/84 | 38/84 | 48/84 |
| Training ranking | 323/336 | 293/336 | 334/336 |
| Observed transfer ranking | 165/168 | 142/168 | 167/168 |
| Strict transfer generation | 54/168 | 32/168 | 64/168 |
| Same-PNG complete recovery + transfer ranking | 15/40 | 14/40 | 19/40 |
| Same-PNG conjunction + original MCQ | 5/40 | 4/40 | 7/40 |
| All-four-rotation consistency | 85/96 | 36/96 | 93/96 |
| Both-correct overwrite pairs | 8/8 | 5/8 | 8/8 |

| Complete recovery | B | C | J | Target |
|---|---:|---:|---:|---:|
| K1 | 15/20 | 19/20 | 17/20 | 18/20 |
| K2 | 0/4 | 1/4 | 1/4 | 3/4 |
| K3 | 0/4 | 1/4 | 1/4 | Reported |
| K4 | 0/12 | 0/12 | 0/12 | 9/12 |

Both new arms complete 0/4 selective-clear states and 0/4 offline teacher chains.
J original MCQ 48/84 fails the unchanged 68/84 target. Transfer ranking passes
135/168; recovery fails. C also fails the overwrite-pair adoption requirements.
Historical failed gates stay unchanged. Observed transfer queries remain
excluded from gradients; they are no longer fresh holdouts or final tests.

J versus C rescues 25 transfer-ranking decisions with no regressions, and
12 original MCQs with 2 regressions. J versus B preserves all 165 previously
correct transfer rankings and rescues 2; MCQs gain 5 and lose 2. J gains 4
complete joint states over B with no state regressions, but its slot conjunction
has 18 gains and 6 regressions. State averages do not erase slot losses.

## Semantic-group aggregation and optimization evidence

| Group macro | B | C | J |
|---|---:|---:|---:|
| Original MCQ | 0.461905 | 0.367262 | 0.519643 |
| Observed transfer ranking | 0.988393 | 0.843750 | 0.996429 |
| Active-slot joint conjunction | 0.427976 | 0.405357 | 0.562500 |

Reopened-PNG mean training answer CE is .008335 for C and .012901 for J;
mean EOS CE is .00002389 and .00004750. These small training losses coexist
with failed query transfer. In J, recovery/application gradient dot products
are negative on 1,630/2,560 updates; actual displacement has negative recovery
and application directional derivatives on 2,390 and 2,515 updates respectively.
Local gradient disagreement therefore does not by itself establish an
unavoidable capability trade-off. No gradients were projected or reweighted.

## Compute and independent verification

5,120 updates; 55,296 gradient Reader forwards (C 16,896; J 38,400).
Training processed tokens: C 2,715,392; J 7,807,040. Summed per-target training
seconds: C 7,485.85; J 18,682.15. This is fixed-update, unequal compute.
Evaluation: 2,568 generations, 1,584 ranking decisions, 6,336 actual candidate
forwards, 528 endpoint recovery-CE forwards; 2,063,198 processed evaluation
tokens and 4,865.39 summed evaluation seconds. Summed times are not job wall time.

54 focused tests passed before dispatch. The independent CPU verifier rebuilds
all metrics from raw reads and candidate/token records, checks complete input
proofs, exact budgets, source/endpoint hashes, optimizer and shared-image traces.
Its local `final-verified.json` matches the remote receipt byte-for-byte. Historical
Plan05–07 sources remain immutable; no historical inference is repeated.

Raw evidence is in `joint-consolidation-v1-run/` (locally unpacked for connector
review); tracked `joint-consolidation-final-verified.json` contains detailed
state/slot, group, rotation and paired outcomes. Archive parts and their hashes
are recorded in `joint-consolidation-archive.json`.
