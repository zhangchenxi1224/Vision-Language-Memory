# R11_new reachable low-LR control

- Mode: `technical-preflight`
- Engineering gate: `True`
- lr-005 M0 MSE: `0.0001353120606`
- lr-005 gradient norm: `0.0006489048621`
- lr-001 M0 MSE: `0.0001353120606`
- lr-001 gradient norm: `0.0006489048621`

The target is fixed from the parent reachable control and is non-semantic.
This isolates optimizer scale at the fixed alpha=0.99 initialization.
Formal Picture Memory success and Phase 2 remain false.
