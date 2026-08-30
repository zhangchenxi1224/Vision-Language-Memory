# Single-step SET evidence audit

Status: scientific-integrity note written while R9 was running, after targets 0 and 1 had completed and before targets 2--7 had endpoint outcomes. This note does not amend the locked R9 protocol or gates.

## Question

Does the archived project evidence support the premise that a single `SET -> QUERY` transition has already been learned by DreamLite at a fixed endpoint?

## Evidence

| Experiment | Fixed comparison | Result | What it establishes |
| --- | --- | ---: | --- |
| R4 64-step full-transition transfer | M0 CE `4.830679` -> step64 CE `4.498315` | `-6.88%` | A mixed SET/OVERWRITE/CLEAR/NOOP aggregate improved. It does not isolate single-step SET. |
| R4 64-step SET-only recovery | M0 CE `1.044407` -> fixed step64 CE `1.369451` | `+31.12%` (worse) | The only archived SET-only fixed endpoint failed. |
| R4 SET-only intermediate checkpoint | M0 CE `1.044407` -> step40 CE `0.988628` | `-5.34%` | A descriptive intermediate fluctuation; it cannot replace the predeclared endpoint or serve as a positive control. |
| R4 256-step FreePixel formal run | M0 CE `9.664663` -> step256 CE `8.986636` | `-7.02%` | Aggregate seed-0 mechanism signal on eight dev episodes; event-specific SET learnability and visual-state causality were not established. |
| Unit/gradient probes | nonzero finite gradients reach source state and LoRA | technical pass | Differentiability/gradient connectivity, not optimizer learnability or endpoint retention. |
| R5 conditional rescue | delayed CE `13.7966` -> `13.5234` | `-1.98%` | A one-seed recurrent mixed-family diagnostic; not a single-step F1 positive control and not formal success. |

Primary archived source for the R4 transfer numbers: `DreamLite_R4_handoff_20260823/HANDOFF.md`, SHA-256 `88c0085f0fc74248cc0188cfc061f50157b24241d4c4300d78db3122a9f4b660`.

## Judgment

The statement "single-step SET has been proven learnable" is not supported by the archived fixed-endpoint evidence. The strongest matching experiment is SET-only, and its fixed endpoint worsened substantially. The earlier premise appears to have conflated one or more of:

1. an improving mixed-event aggregate;
2. a post-hoc best intermediate checkpoint;
3. a nonzero-gradient connectivity test;
4. actual fixed-endpoint, causal single-step learnability.

These are different claims. Only the fourth is a valid lower bound for recurrent picture memory.

## Consequence for R9 and the next branch

- R9 remains unchanged. Its eight hard8 targets, endpoint, coefficient, controls, and locked pass criteria must finish exactly as launched.
- The phrase "using the existing single-step SET positive control" in the R9 locked `0/8` interpretation must be treated as an unsupported premise, not as evidence.
- If R9 is `0/8`, the next main experiment must first establish the missing one-step visual-alignment lower bound before further recurrent-memory or gradient-aggregation changes.
- If R9 is `1--7/8`, transition heterogeneity remains causal, but no broad architectural-success claim is allowed until the same one-step lower bound is established.
- If R9 is `8/8`, R9 supports individual hard8 representability at the tested coefficient; it still does not retroactively validate the historical SET-only claim.

## Minimum valid lower-bound evidence

A future one-step F1 diagnostic must preserve the current scientific evaluation contract: frozen Reader, teacher-free image state, listwise CE, four fixed reverse-cyclic choice views, fixed endpoint, normal/reset causal comparison, immutable target selection, artifact hashes, and no best-checkpoint rescue. At minimum it must require:

1. technical validity and complete receipts;
2. endpoint CE improvement on every fixed choice view;
3. a substantial preregistered relative CE reduction and nonzero accuracy gain;
4. normal state outperforming reset at the endpoint;
5. replication across multiple independently selected F1 units before it can be called a lower bound.

If DreamLite fails this lower bound, the next diagnostic should compare it with a directly optimized visual-state oracle under the identical Reader/loss/evaluation contract. That comparison distinguishes an unreadable Reader/image interface from an insufficient DreamLite update parameterization without changing the final research objective.

## Provenance

- Historical plan that asserted the premise: SHA-256 `94a6e839cebe3dfd35d4ecaebc2da5193841e893f291acfeb96f7bc3b9f4c690`.
- Locked R9 preregistration before this note: SHA-256 `e7078f622508ce05d0592a0427a13f974bc6d6003cd769e21b36c198ddf976d9`.
- R4 formal report: SHA-256 `7871e95030a7507094aeedb63481a97c637b7a502c967de1cde127ed56644841`.

This audit is a correction of evidentiary interpretation, not a scientific result and not a success claim.
