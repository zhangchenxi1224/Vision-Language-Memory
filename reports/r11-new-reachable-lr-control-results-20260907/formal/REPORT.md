# R11_new reachable low-LR control

- Mode: `formal`
- Engineering gate: `True`
- lr-005 M0 MSE: `0.0001353120606`
- lr-005 raw256 MSE/M0: `0.1232824383`
- lr-005 gate: `False`
- lr-001 M0 MSE: `0.0001353120606`
- lr-001 raw256 MSE/M0: `0.1209362048`
- lr-001 gate: `False`
- Classification: `learning_rate_reduction_insufficient`

The target is fixed from the parent reachable control and is non-semantic.
This isolates optimizer scale at the fixed alpha=0.99 initialization.
Formal Picture Memory success and Phase 2 remain false.
