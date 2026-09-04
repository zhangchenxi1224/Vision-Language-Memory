# R12 shared event-to-latent writer: paired result

**Decision:** `diagnose_shared_writer_fit_boundary`

The conditioned writer did not fit every held-in F1 target; localize event representation, coefficient mapping, and shared latent-basis limitations before expanding scope.

## Fixed causal gates

| arm | train audit | dev select | sealed dev final | arm gate |
| --- | ---: | ---: | ---: | --- |
| conditioned | 0/36 | 0/24 | 0/24 | FAIL |
| constant-control | 0/36 | 0/24 | 0/24 | FAIL |

## First-principles localization

The frozen event representation is linearly predictive of the target value (ridge audit: train-audit 100.0%, dev-select 83.3%, dev-final 95.8%).
After the learned coefficient head, the same audit falls to 72.2%, 20.8%, and 20.8%. The dominant failure is therefore conditional-code collapse after the event encoder, not absence of event information in the frozen encoder.

The R12 scientific outcome remains unchanged by this post-hoc audit. R12 is diagnostic-only and cannot establish full Picture Memory success.
