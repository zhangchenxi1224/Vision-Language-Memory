# R6 source-anchor paired diagnostic evidence package

R6 completed successfully as an experiment, but it is a **negative learnability result** and is
not evidence of successful picture memory.  Both arms passed the technical integrity gate; neither
arm passed the preregistered repeated-hard8 overfit gate or the fixed-dev generalization gate.

## Question and matched design

R6 tested whether repeatedly redrawing the updater state from pure diffusion noise was the main
cause of the R5 gradient pathology.  The two arms used the same code revision, selected examples,
Reader, loss, LoRA configuration, optimizer settings, seed, 128 optimizer steps, and endpoint
evaluation.  The controlled change was the effective edit-start flow sigma:

- `legacy-pure-noise`: effective sigma `1.0`.
- `source-anchored`: source-conditioned state with effective sigma `0.5` and schedule
  `[0.5, 0.375, 0.25, 0.125]`.

The scheduler implementation was corrected before the final paired run so these values are the
effective post-shift flow sigmas.  The final pair is bound to commit
`e1ab129ae86a39814a9ce0ce17ac06965f2e835c` and implementation
`scheduler-effective-sigma-v2`.

## Main results

| Arm | hard8 CE delta vs own M0 | 95% CI | accuracy delta | dev formal CE delta | clip rate | negative pairwise gradient cosine | hard8 gate | dev gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| legacy pure noise | -1.4418 | [-3.4306, +0.1401] | 0.0000 | +0.5653 | 1.0000 | 46.4% | FAIL | FAIL |
| source anchored | -0.0805 | [-0.2134, +0.0581] | 0.0000 | +0.0189 | 0.0625 | 42.9% | FAIL | FAIL |

Source anchoring reduced the median per-segment gradient norm from `4718.65` to `6.59`, reduced
the max/min norm ratio from `594.7` to `25.4`, and reduced clipping from `100%` to `6.25%`.
Therefore, pure-noise initialization was a real cause of gradient-scale pathology.  However, this
repair did not produce hard8 memorization, accuracy improvement, fixed-dev improvement, or causal
state evidence.  Gradient direction conflict remained high, so edit-start sigma is not a
sufficient explanation or solution.

The preregistered decision is
`reject_sigma_as_sufficient_test_gradient_balancing`: the next mainline experiment must change the
gradient aggregation law while holding the task, Reader, loss, data, updater, and success gates
fixed.  A full-data or hyperparameter-scale-up run is not justified before the unchanged hard8
bottleneck can be learned.

## Included repository artifacts

- [`REPORT.md`](REPORT.md): compact rendered endpoint table.
- [`ANALYSIS.json`](ANALYSIS.json): complete paired uncertainty and gate analysis.
- [`comparison/REPORT.md`](comparison/REPORT.md) and
  [`comparison/comparison.json`](comparison/comparison.json): original arm-to-arm decision.
- [`training_metrics.csv`](training_metrics.csv): all 128 optimizer-step metrics for both arms.
- [`gradient_conflict.csv`](gradient_conflict.csv): per-segment gradient norms and directions.
- [`endpoint_summary.csv`](endpoint_summary.csv): M0/endpoint paired endpoint metrics.
- [`training_diagnostics.png`](training_diagnostics.png),
  [`gradient_conflict.png`](gradient_conflict.png), and
  [`endpoint_metrics.png`](endpoint_metrics.png): rendered figures.
- [`DELIVERY_MANIFEST.json`](DELIVERY_MANIFEST.json): SHA-256 manifest for rendered artifacts and
  source inventories.
- [`RAW_ARTIFACTS.json`](RAW_ARTIFACTS.json): raw archive, checkpoint, provenance, and storage
  bindings.

## Complete raw artifacts

The validated local delivery is:

```text
C:\Users\Expedition\DreamLite_R6_SourceAnchor_Final_20260829
```

The authoritative compressed archive is
`raw/r6-complete-e1ab129.validated-resume.tar.gz`: `314,369,636` bytes, SHA-256
`79e65a55d70173ab7cfdaf715678912ff9125f7058262c0851c8303c549fc826`.  It expands to 62 files
and `380,851,786` bytes under `raw/complete-validated`, including all five checkpoints per arm,
raw/EMA endpoints, manifests, environment/runtime records, stdout/stderr, optimizer and
micro-step metrics, gradient audits, evaluation rows, summaries, and comparison outputs.

The original Inspire run remains at:

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r6/r6-source-anchor-e1ab129-20260829
```

Integrity was checked at three levels: every transfer chunk against the remote SHA list, the full
archive against its remote size and SHA-256, and every extracted run artifact against each arm's
`artifact_inventory.json`.  The earlier `a665240` partial runs used incorrect scheduler-sigma
semantics, were intentionally terminated before endpoints, and are excluded from this evidence
package; their audit trail is preserved in
[`../r6-source-anchor-sigma-audit-20260829.md`](../r6-source-anchor-sigma-audit-20260829.md).
