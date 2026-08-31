# R10 DreamLite raw-endpoint attribution

This post-registered attribution cannot replace or rescue the preregistered EMA endpoint.

- Raw descriptive gates: 0/8.
- Existing EMA gates: 0/8.
- Decision: `ema_is_not_sufficient_explanation_run_vae_latent_oracle`
- Reason: No raw step128 target passes. EMA lag is not a sufficient explanation; test whether the VAE latent space itself contains readable codes before redesigning the writer.

| Target | Raw gate | Raw relative CE | EMA relative CE | Raw views | Raw accuracy delta | Raw/EMA CE difference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | FAIL | -1.04% | +0.17% | 4/4 | +0.00 | -0.1849 |
| 1 | FAIL | -6.90% | -1.43% | 4/4 | +0.00 | -1.5156 |
| 2 | FAIL | -2.29% | -1.43% | 4/4 | +0.00 | -0.2656 |
| 3 | FAIL | -5.99% | +0.03% | 4/4 | +0.00 | -2.1250 |
| 4 | FAIL | -3.20% | -0.81% | 4/4 | +0.00 | -0.8021 |
| 5 | FAIL | +0.16% | -0.27% | 2/4 | +0.00 | +0.0755 |
| 6 | FAIL | -0.41% | -0.29% | 3/4 | +0.00 | -0.0156 |
| 7 | FAIL | -0.16% | +0.09% | 3/4 | +0.00 | -0.0885 |

Raw results are root-cause evidence only; formal success remains false.
