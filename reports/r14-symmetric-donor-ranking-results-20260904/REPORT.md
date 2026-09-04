# R14 symmetric fixed-donor ranking: result

**Decision: technical pass; scientific diagnostic fail. No formal Picture Memory success is claimed.**

| split | passed all gates | required | normal accuracy | donor accuracy | base accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_audit | 17 | 36 | 86.8% | 34.0% | 50.0% |
| dev_select | 1 | 24 | 25.0% | 31.2% | 29.2% |
| dev_replay | 4 | 24 | 32.3% | 29.2% | 29.2% |
| dev_final | 7 | 24 | 50.0% | 32.3% | 21.9% |

## First-principles diagnosis

The objective learned the fixed training contrast: last-64 own CE fell while donor CE rose and 60.9% of late micro-steps satisfied the ln(4) margin.

Only 17/36 train-audit and 1/24, 4/24, 7/24 held-out targets passed every causal gate. This is a scientific failure, not a full Picture Memory result.

The objective was symmetric only over an epoch: just 44/2304 pair-epoch directions shared an optimizer update, with median lag 11. Each event also saw one fixed negative, while 0/36 train-audit evaluation donor identities and 0/36 donor target values overlapped that negative.

## Locked next test

Use update-synchronous bidirectional pair loss and deterministic rotating negatives while preserving the R14 writer, Reader-call budget, optimizer, endpoint, data, and normal/reset/donor/base gates.
