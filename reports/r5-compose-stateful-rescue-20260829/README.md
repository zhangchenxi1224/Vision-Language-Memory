# R5-Compose evidence package

This directory is the repository copy of the completed R5-Compose delivery.  R5 ended as a
**negative mainline result with a diagnostic residual rescue**; it is not evidence of successful
multi-step picture memory.

## What is included

- `FINAL_REPORT.md`: concise human-readable method, results, and conclusion.
- `FINAL_REPORT.json`: complete machine-readable report payload.
- `EXECUTION_SUMMARY.json`: optimizer, gradient, topology, provenance, and checkpoint hashes.
- `POSTHOC_PAIRED_ENDPOINT_AUDIT.json`: matched endpoint-versus-M0 uncertainty audit.
- `main_seed_summary.csv`: intentionally header-only because no 640-step seed was launched after
  the preregistered 128-step pilot failed its mechanism gate.
- `figures/`: gradient fidelity, pilot/rescue endpoint CE, and rescue training-loss plots.
- `DELIVERY_MANIFEST.json`: immutable manifest from the original delivery package.  It covers the
  original seven report/CSV/figure artifacts; the later execution summary and post-hoc audit are
  supplementary files and therefore are not retroactively inserted into that manifest.

## Source of truth for large/raw artifacts

The complete run tree remains on Inspire at:

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r5/r5-compose-stateful-rescue-b02a411-20260829
```

Important child runs:

```text
runs/pilot-latent-h2-full
runs/rescue-latent-h4-k0-tau05
runs/gradient-audit-latent-h4
```

Those directories contain manifests, stdout/stderr, per-step metrics, evaluation rows, state
examples, and checkpoints.  Checkpoint content hashes are recorded here so a later transfer can
be verified without treating repository availability as scientific evidence:

| Run | Endpoint | SHA256 |
| --- | --- | --- |
| pilot latent+h2 full | raw | `9a6907d3edb508593aeb96a022232691c3017cb9364b5e3294076ef4a9e0cdbe` |
| pilot latent+h2 full | EMA | `5cbef4d2c8ef2243a428b1ffc4a6ca2cba0d6fb73c0b292b84a5ea8c8350f3c3` |
| rescue latent+h4 tau=0.5 | raw | `9eb077a41693780ebd2f223369877b08d5d9e0a3f325c77d5cd1f7523c371266` |
| rescue latent+h4 tau=0.5 | EMA | `19472c226a46e1f0dbc21238ea1a16f952a7e93058750ea21e83046edafd649e` |

Model/data bindings and all code revisions are preserved in `EXECUTION_SUMMARY.json`.  The failed
recursive checkpoint transfer under the local delivery root is deliberately excluded because it
contains an incomplete zero-byte payload and is not valid evidence.

