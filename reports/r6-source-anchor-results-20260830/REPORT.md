# R6 source-anchor paired diagnostic delivery

This is a one-seed repeated-hard8 bottleneck diagnostic and cannot establish formal picture-memory success.

- Git commit: `e1ab129ae86a39814a9ce0ce17ac06965f2e835c`
- Implementation: `scheduler-effective-sigma-v2`
- Selected-segment SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`

| Arm | hard8 M0 CE | endpoint CE | delta CE | relative | improved units | accuracy delta | hard8 gate | formal delta | mechanism delta | fixed-dev gate | clip rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| legacy-pure-noise | 16.5563 | 15.1145 | -1.4418 | -0.0871 | 5 | 0.0000 | FAIL | 0.5653 | -0.1023 | FAIL | 1.0000 |
| source-anchored | 21.1545 | 21.0740 | -0.0805 | -0.0038 | 4 | 0.0000 | FAIL | 0.0189 | -0.0009 | FAIL | 0.0625 |

Interpretation must follow the preregistered gates. Absolute CE across arms is secondary because each update law has a different M0; the primary comparison is endpoint versus that arm's own M0 with paired uncertainty and causal state controls.
