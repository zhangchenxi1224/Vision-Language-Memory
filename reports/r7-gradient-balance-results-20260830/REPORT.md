# R7 gradient-balance paired diagnostic delivery

This is a one-seed repeated-hard8 bottleneck diagnostic and cannot establish formal picture-memory success.

- Git commit: `c720f6b28e3ce6ef4e8f838a576d3b042d35cd58`
- Implementation: `explicit-micro-gradient-unit-balance-v1`
- Selected-segment SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`

| Arm | hard8 M0 CE | endpoint CE | delta CE | relative | improved units | accuracy delta | hard8 gate | formal delta | mechanism delta | fixed-dev gate | clip rate | min raw/applied cosine | max norm error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| raw-mean-control | 21.1545 | 21.0740 | -0.0805 | -0.0038 | 4 | 0.0000 | FAIL | 0.0189 | -0.0009 | FAIL | 0.0625 | 1.0000 | 0.00e+00 |
| unit-balanced-norm-matched | 21.1545 | 21.0771 | -0.0774 | -0.0037 | 5 | 0.0000 | FAIL | 0.0239 | -0.0435 | FAIL | 0.3203 | -0.5081 | 1.29e-07 |

Interpretation must follow the preregistered gates. Both arms share the same M0 architecture and differ only in the applied gradient direction; the unit-balanced norm is matched before the unchanged clip.
