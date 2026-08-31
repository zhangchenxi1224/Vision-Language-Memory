# R9 first-principles conclusion

R9 is a one-seed bottleneck diagnostic, not a formal Picture Memory result. Each of the eight fixed hard targets was trained independently for 128 optimizer steps at its original `1/8` target-gradient coefficient. Every target passed the technical gate; zero passed the preregistered scientific gate.

| Target | Family | Endpoint CE change | Improved held-out views | Accuracy change | Clip rate | Gate |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | F2 | -0.10% | 3/4 | 0.00 | 0.78% | FAIL |
| 1 | F2 | -0.02% | 3/4 | 0.00 | 0.00% | FAIL |
| 2 | F3 | +2.92% | 0/4 | 0.00 | 1.56% | FAIL |
| 3 | F3 | -0.95% | 3/4 | 0.00 | 0.00% | FAIL |
| 4 | F5 | -0.60% | 3/4 | 0.00 | 1.56% | FAIL |
| 5 | F5 | +0.45% | 2/4 | 0.00 | 0.78% | FAIL |
| 6 | F6 | -0.76% | 2/4 | 0.00 | 0.00% | FAIL |
| 7 | F6 | -2.23% | 4/4 | 0.00 | 0.00% | FAIL |

The scientific gate required all of: endpoint CE change at most -20%, improvement in 4/4 fixed held-out choice views, accuracy gain at least +0.25, and negative normal/reset difference-in-differences.

## What this rules out

- The optimizer executed real nonzero updates; the realized update/weight ratio was nonzero throughout.
- Gradient clipping was rare (0--1.56%), so persistent clipping is not a sufficient explanation.
- Each target was optimized alone, so simultaneous hard8 gradient conflict is not a sufficient explanation.
- Failure spans F2, F3, F5, and F6, so it is not confined to one transition family.

## What remains unresolved

R9 shows weak score movement without functional answer recovery. Target 7 is the clearest example: all four held-out views improved, but mean CE fell only 2.23% and accuracy did not change. Therefore, differentiability and local CE movement do not demonstrate a readable visual memory code.

The missing lower bound is now the main-line question: does the frozen Reader/preprocessing/loss interface contain a learnable image-space code at all, and if so can the current DreamLite one-step updater write it? Conditional R10 was preregistered before the remaining R9 outcomes and is activated by the final 0/8 result. It compares a direct-pixel capacity oracle against single-SET DreamLite with a full target-gradient coefficient, while preserving the fixed data, Reader, four disjoint endpoint views, causal reset control, and no-best-checkpoint rule.

Raw run root: `/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r9/r9-individual-learnability-0eddfa2-20260831-rerun1`

Local delivery: `C:\Users\Expedition\DreamLite_R9_IndividualLearnability_Final_20260831`
