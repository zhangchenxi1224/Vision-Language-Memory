# R13 mean-centered conditional residual writer: result

**Decision:** technical pass; scientific diagnostic fail. No formal Picture Memory success is claimed.

## Fixed causal result

| split | passed all gates | required | normal accuracy | donor accuracy | base accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_audit | 14 | 36 | 100.0% | 49.3% | 50.0% |
| dev_select | 1 | 24 | 19.8% | 39.6% | 29.2% |
| dev_replay | 5 | 24 | 42.7% | 29.2% | 29.2% |
| dev_final | 6 | 24 | 44.8% | 28.1% | 21.9% |

## Interpretation

The fixed base plus centered conditional residual reached 100% train-audit normal accuracy, 0% reset accuracy, and 14/36 full causal target gates, improving on R12's 0/36.

Wrong-event donor and zero-residual base remained correct too often in train-audit, while held-out normal accuracy fell to 19.8%-44.8% and the conditional residual often underperformed the fixed base. The shared event-to-code map is therefore overfit and insufficiently discriminative.

The decoded images are measurable high-frequency visual codes without a human-readable semantic payload. This is compatible with Picture Memory, but it neither proves uniform all-pixel storage nor establishes causal/generalizable memory by itself.

## Locked next experiment

Explicit symmetric own-versus-wrong-donor ranking during training, with the R13 normal/reset/donor/base evaluation held unchanged.
