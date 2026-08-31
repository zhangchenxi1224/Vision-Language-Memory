# R11 VAE-latent reachability result

R11 is a per-target representation oracle. It is diagnostic and is not a shared memory-writer success claim.

- Decision: `replace_semantic_editor_with_shared_event_to_latent_writer`
- Target passes: 8/8.
- Reason: All eight independently optimized VAE latents pass the fixed causal Reader gate. The frozen VAE representation can carry the code; the next bottleneck test is a shared event-to-latent writer with held-out F1 targets, not further semantic-prompt or EMA tuning.
- Source training commit: `f4a018f3c4eef453fff3367b049ea732332e8c37`
- Aggregation commit: `ef29c7704f76a5a419ac95b68d9329f37deb9512`
- Fixed F1 payload SHA-256: `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845`

| Target | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | PASS | 15.098970 | 0.000272 | -99.9982% | 4/4 | +1.000 | -15.098699 |
| 1 | PASS | 27.697927 | 0.000177 | -99.9994% | 4/4 | +1.000 | -27.697751 |
| 2 | PASS | 31.182292 | 0.000239 | -99.9992% | 4/4 | +1.000 | -31.182053 |
| 3 | PASS | 34.895833 | 0.000213 | -99.9994% | 4/4 | +1.000 | -34.895620 |
| 4 | PASS | 33.182292 | 0.000284 | -99.9991% | 4/4 | +1.000 | -33.182008 |
| 5 | PASS | 17.502603 | 0.000849 | -99.9952% | 4/4 | +1.000 | -17.501754 |
| 6 | PASS | 13.291671 | 0.000193 | -99.9986% | 4/4 | +1.000 | -13.291478 |
| 7 | PASS | 35.317709 | 0.000080 | -99.9998% | 4/4 | +1.000 | -35.317629 |

Even 8/8 establishes only that independently optimized VAE latents can carry target-specific codes. Formal Picture Memory success still requires one shared event-conditioned writer, held-out ID/OOD targets, recurrence, multiple seeds, and causal state controls.
