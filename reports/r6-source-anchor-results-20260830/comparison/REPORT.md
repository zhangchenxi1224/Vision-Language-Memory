# R6 source-anchor paired diagnostic

**Decision:** `reject_sigma_as_sufficient_test_gradient_balancing`

Neither arm can overfit the identical hard8 state algebra; edit start sigma is not sufficient and the recorded gradient conflict becomes the next main hypothesis.

| Arm | sigma | hard8 delta CE | hard8 delta acc | hard8 gate | formal delta CE | mechanism delta CE | fixed-dev gate | negative grad cosine | grad max/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy-pure-noise | 1.00 | -1.4418 | +0.000 | FAIL | 0.5653 | -0.1023 | FAIL | 0.464 | 594.7 |
| source-anchored | 0.50 | -0.0805 | +0.000 | FAIL | 0.0189 | -0.0009 | FAIL | 0.429 | 25.4 |

This result cannot be called formal picture-memory success.  Any advancing arm must next pass the unchanged full-data endpoint, ID/OOD, reset/swap, and multi-seed gates.
