# R7 gradient-balance paired diagnostic

**Decision:** `reject_unit_balance_as_sufficient_test_conflict_projection`

Neither arm learned the identical hard8 state algebra; equal weighting is insufficient and deterministic conflict projection is the next fixed-bottleneck test.

| Arm | aggregation | hard8 delta CE | hard8 delta acc | hard8 gate | formal delta CE | mechanism delta CE | fixed-dev gate | min raw/applied cosine | max norm error | clip rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw-mean-control | raw-mean | -0.0805 | +0.000 | FAIL | 0.0189 | -0.0009 | FAIL | 1.0000 | 0.00e+00 | 0.0625 |
| unit-balanced-norm-matched | unit-balanced-norm-matched | -0.0774 | +0.000 | FAIL | 0.0239 | -0.0435 | FAIL | -0.5081 | 1.29e-07 | 0.3203 |

This result cannot be called formal picture-memory success. Any advancing arm must still pass the unchanged full-data endpoint, ID/OOD, reset/swap, and multi-seed gates.
