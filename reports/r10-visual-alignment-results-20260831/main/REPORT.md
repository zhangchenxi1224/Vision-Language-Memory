# R10 visual-alignment lower bound

R10 is a one-seed, repeated-target bottleneck diagnostic. It cannot establish formal picture-memory success.

- Decision: `redesign_dreamlite_updater_only`
- Direct-pixel oracle: 8/8.
- DreamLite single SET: 0/8.
- Reason: The frozen Reader/image channel is usable on all eight targets, but the current DreamLite updater does not pass all eight. Restrict the next repair to updater parameterization, conditioning, and optimization.
- Git commit: `ba86e2c5b6a4d55f97ba386ad135fea546a22dbf`
- Fixed F1 payload SHA-256: `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845`

| Arm | Target | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD | Clip rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct-pixel-oracle | 0 | PASS | 15.0912 | 0.0004 | -100.00% | 4/4 | +1.000 | -15.0908 | n/a |
| direct-pixel-oracle | 1 | PASS | 27.6771 | 0.0003 | -100.00% | 4/4 | +1.000 | -27.6768 | n/a |
| direct-pixel-oracle | 2 | PASS | 30.2031 | 0.0012 | -100.00% | 4/4 | +1.000 | -30.2019 | n/a |
| direct-pixel-oracle | 3 | PASS | 33.9948 | 0.0006 | -100.00% | 4/4 | +1.000 | -33.9942 | n/a |
| direct-pixel-oracle | 4 | PASS | 32.1875 | 0.0011 | -100.00% | 4/4 | +1.000 | -32.1864 | n/a |
| direct-pixel-oracle | 5 | PASS | 17.6016 | 0.0004 | -100.00% | 4/4 | +1.000 | -17.6012 | n/a |
| direct-pixel-oracle | 6 | PASS | 13.1641 | 0.0004 | -100.00% | 4/4 | +1.000 | -13.1637 | n/a |
| direct-pixel-oracle | 7 | PASS | 34.6250 | 0.0004 | -100.00% | 4/4 | +1.000 | -34.6246 | n/a |
| dreamlite-single-set | 0 | FAIL | 15.2839 | 15.3099 | +0.17% | 1/4 | +0.000 | +0.0260 | 8.59% |
| dreamlite-single-set | 1 | FAIL | 27.6875 | 27.2917 | -1.43% | 4/4 | +0.000 | -0.3958 | 59.38% |
| dreamlite-single-set | 2 | FAIL | 30.9375 | 30.4948 | -1.43% | 4/4 | +0.000 | -0.4427 | 27.34% |
| dreamlite-single-set | 3 | FAIL | 35.2917 | 35.3021 | +0.03% | 2/4 | +0.000 | +0.0104 | 14.84% |
| dreamlite-single-set | 4 | FAIL | 33.5000 | 33.2292 | -0.81% | 4/4 | +0.000 | -0.2708 | 7.81% |
| dreamlite-single-set | 5 | FAIL | 17.4531 | 17.4062 | -0.27% | 2/4 | +0.000 | -0.0469 | 2.34% |
| dreamlite-single-set | 6 | FAIL | 13.3438 | 13.3047 | -0.29% | 2/4 | +0.000 | -0.0391 | 3.12% |
| dreamlite-single-set | 7 | FAIL | 35.7656 | 35.7969 | +0.09% | 1/4 | +0.000 | +0.0313 | 19.53% |

Every target must pass every preregistered endpoint condition. Training loss, an intermediate checkpoint, or partial target success cannot override this decision.
