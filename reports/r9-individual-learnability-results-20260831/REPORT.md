# R9 individual-learnability decomposition

This is a one-seed repeated-hard8 bottleneck diagnostic. It cannot establish formal picture-memory success.

- Decision: `reject_batch_aggregation_reopen_recurrent_alignment_and_temporal_credit`
- Result: 0/8 target gates passed.
- Reason: No transition is learnable alone at its original coefficient; simultaneous aggregation is not the sufficient bottleneck. The archived single-step SET endpoint is not a valid positive control, so direct-pixel and one-step DreamLite visual-alignment lower bounds must be established before recurrence is changed again.
- Git commit: `0eddfa273ae159deac8304db37f3c2a7baf04cee`
- Hard8 SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`

| Target | Family | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD | Clip rate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | F2 | FAIL | 17.4609 | 17.4427 | -0.10% | 3/4 | +0.000 | -0.0182 | 0.78% |
| 1 | F2 | FAIL | 9.1652 | 9.1629 | -0.02% | 3/4 | +0.000 | -0.0022 | 0.00% |
| 2 | F3 | FAIL | 14.1901 | 14.6042 | +2.92% | 0/4 | +0.000 | +0.4141 | 1.56% |
| 3 | F3 | FAIL | 12.6276 | 12.5078 | -0.95% | 3/4 | +0.000 | -0.1198 | 0.00% |
| 4 | F5 | FAIL | 33.1775 | 32.9796 | -0.60% | 3/4 | +0.000 | -0.1979 | 1.56% |
| 5 | F5 | FAIL | 36.2293 | 36.3907 | +0.45% | 2/4 | +0.000 | +0.1614 | 0.78% |
| 6 | F6 | FAIL | 14.3333 | 14.2240 | -0.76% | 2/4 | +0.000 | -0.1094 | 0.00% |
| 7 | F6 | FAIL | 32.0521 | 31.3385 | -2.23% | 4/4 | +0.000 | -0.7135 | 0.78% |

A target passes only when all preregistered endpoint conditions hold. Training loss, non-target improvement, or an intermediate checkpoint cannot rescue a failed target.
