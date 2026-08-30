# R8 common-descent deployment audit — 2026-08-30

## Scientific contract

- Question: does projecting the raw eight-segment batch gradient into the common first-order descent cone restore hard8 learnability?
- Arms: `raw-mean-control` and `common-descent-projected-norm-matched`.
- Only changed variable: pre-clip micro-gradient aggregation.
- Fixed: exact R6/R7 hard8, seed 0, source-anchor schedule, rank 4, frozen Qwen3-VL Reader, listwise CE, AdamW schedule, clip 10, EMA endpoint, 128 optimizer steps, fixed dev and reset/swap evaluation.
- This is a one-seed repeated-hard8 bottleneck diagnostic. It cannot claim formal picture-memory success.

## Locked inputs

- Training commit: `82e983743be73919f257d441f1cacb2b7f601288`
- Train SHA-256: `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`
- Dev SHA-256: `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303`
- Selected hard8 SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`
- DreamLite manifest SHA-256: `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159`
- Qwen3-VL manifest SHA-256: `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c`

## Pre-launch verification

- Local R5–R8 implementation/reporting suite: 46 passed.
- H200 R7/R8 core and direct-entrypoint suite: 16 passed.
- Both wrapper files were executed from `/tmp`, outside the repository, before launch.
- CUDA full-size synthetic projection smoke: 1,644,544 dimensions, one raw violating micro-gradient, zero projected violations, minimum projected cosine `1.362391799375473e-07`, norm-match relative error `0`, active set mask `128`.
- The same smoke passed with TF32 explicitly enabled and with strict deterministic CUDA (TF32 disabled).
- Both target instances were idle before launch: two NVIDIA H200 GPUs each, 1 MiB used per GPU, 0% utilization, no compute process.

## Fail-closed engineering attempts

These roots are retained for audit and must not be interpreted as training experiments.

1. `r8-common-descent-371d9a9-20260830`
   - The R8 controller wrapper could not import `scripts` when invoked as a file.
   - Failure occurred before `launch.json`, model loading, or any optimizer step.
   - Fix: commit `0c41a441316214a226cf6bcd44c332268a516bb4` plus a direct-file regression test.
2. `r8-common-descent-0c41a44-20260830`
   - Both controllers passed input/commit/data validation, but the R8 trainer wrapper had the same direct-file import defect.
   - Both terminal records are `failed`, child exit code 1, elapsed about 0.04 seconds, no summary and zero optimizer steps.
   - Fix: commit `82e983743be73919f257d441f1cacb2b7f601288` plus a second direct-file regression test.

No threshold, metric, dataset, hyperparameter, or scientific gate was changed while fixing these entrypoints.

## Valid scientific run

- Root: `/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r8/r8-common-descent-82e9837-20260830`
- Raw instance: `vlm-dreamlite-full-h200x2-20260720`
- Projected instance: `vlm-r3-h200x2-live-20260717`
- Raw launch UTC: `2026-08-30T09:29:54.517287+00:00`
- Projected launch UTC: `2026-08-30T09:30:33.837251+00:00`
- Both arms wrote data/schedule audits, deterministic runtime manifests, loaded DreamLite and the frozen Reader on separate H200 GPUs, completed M0/gradient-conflict setup, and emitted real optimizer-step receipts.
- First-step parity check passed: both arms reported loss `21.024068474769592` and pre-clip aggregate norm approximately `3.175676` at optimizer step 1.
- Divergence began only after the changed aggregation law could affect parameters, as intended.

## Completion handling

On terminal completion, require all of the following before interpretation:

1. each arm has a passed `terminal.json`, a completed R8 summary, and exactly 128 optimizer-step records;
2. every source artifact matches its controller inventory size and SHA-256;
3. projected steps have no projected micro-cosine below `-1e-5`, zero projected violation count, nonzero raw-conflict incidence, active intervention, and norm error at most `1e-5`;
4. paired comparison uses each arm's own M0 and the preregistered hard8/fixed-dev gates;
5. raw data, logs, checkpoints, evaluation rows, reports, CSVs, plots, manifests, and checksums are downloaded and delivered before the next scientific iteration.
