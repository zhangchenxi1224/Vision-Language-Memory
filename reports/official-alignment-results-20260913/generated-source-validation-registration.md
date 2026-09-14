# Generated-source continuation: complete validation registration

## Execution update: 2026-09-14, after new allocation

At Unix time 1789383905.6168113 (19:05:05 CST), the actual driver, four
training workers and validation waiter were all live. Optimization reached
3,487 of the fixed 4,832 updates. The complete measured baseline gate passed:
302 images/latents and their full trajectories match the trained4f parent
bitwise, and all 3,020 raw generation records match except the administrative
phase label. This establishes parameter restart parity, not a new trained result.

The entire baseline raw-record archive was independently downloaded and recounted
locally with `scripts/reporting/verify_generated_source_baseline_local.py`:
`b62ec02-complete-baseline-raw-evidence.tgz`, 599,397 bytes, SHA256
`a8764af237bb41a8382dd063d021d64b8b5bd27ba6a24c966700c8779c9efa2e`.
Both phases contain all 3,020 registered cells and score 1,510/1,510 matched
token-plus-immediate-EOS responses. The adjacent local-verification JSON records
the complete recount, exact parent lineage and canonical runtime identity.
Tensor payloads and checkpoints remain remote; their equality is recorded by
the actual GPU baseline gate, not independently recomputed from this raw archive.
The fixed trained endpoint, both functional suites, CLI and PNG acceptance remain
pending. Earlier observations below are retained as execution history.

`dl-source-aug-h200x4-20260914` obtained four H200s on `qb-prod-gpu911`.
Actual CUDA12.8 / Torch2.7.0a0+ecf3bae40a.nv25.02 / NCCL2.25.1 / AVX512 and
the three clean source checkouts were verified in the new container. See
`generated-source-instance-runtime-20260914.json` for the observed runtime.

The complete `c83aca0-source-runtime-preflight-r2` passed at Unix time
1789379425.4063559 (2026-09-14 17:50:25 CST): all972 source/condition pairs,
all27 actual finite-gradient draws, unchanged parameters and the final source-file
integrity recheck. The archive is 78,264 bytes, SHA256
`9811cd1fba14018db54379e8822f3e3a466b19bf92441d170d38398a249372e8`.
The archive was downloaded and its complete metadata recounted locally.
`source-path-fix-preflight-comparison.json` additionally shows identical bytes
before and after the type fix for canonical runtime, all source-condition bindings,
selected draws and recorded losses. Earlier failed attempts remain failed.

The full `b62ec02-generated-source-full4832` job has now been dispatched on this
new notebook. Live processes were observed: driver58922 and four GPU workers
60713–60716. At that observation the workers were preparing the full runtime;
the four-rank gradient gate and newly measured baseline still precede optimization.
The full validation chain was also started (supervisor83262). Its original suite
is explicitly `waiting_for_fixed_endpoint`; the observed-expression suite and
complete PNG readback follow sequentially. All use deadline1789406700, or
2026-09-15 01:25 CST, before the observed allocation's automatic stop.
No new trained endpoint or functional score is claimed yet.

The subsequent four-rank startup gate also passed. All four initial parameter
digests equal the fixed4f `16bf3923...` digest. Actual serial and four-rank draws
and losses were identical; gradient relative L2 error was
`3.846881746819174e-08` and relative maximum error was
`5.316059966006986e-08`, both below the unchanged `2e-6` tolerance.
`b62ec02-four-rank-startup-proof.json` binds both complete gate reports to the
actual training registration, identity, runtime and both augmentation seals.
Its independently observed remote SHA256 is
`e877db9ac63a285a9180229fa7545f702cb67abf1b591f467ce9d38a7a10fcdb`.
Full baseline generation is now running; this startup gate does not assert
baseline equality, completed optimization, or functional acceptance.

## Registration and earlier deployment observations

The next training source is `b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36`.
It fixes the concrete source-path type error found by the zero-update GPU probe.
The earlier `ef163b26e33f62c496ed0da8744ebb7bf1163873` training version and its
`f29cbfb7fa7cc749642c1a180914778b29482b41` validation source are superseded before
any optimizer run. The old launch helper is explicitly disabled. Training has a
separate clean checkout; current validation changes do not alter its source.

The fixed training plan is `generated-source-training-preregistered.json`, SHA256
`86a6bba8fa438da673af73223cfe2d3ab57759564159c05385b151d828e5c922`.
The complete observed-expression regression plan is
`generated-source-observed-wording-preregistered.json`, SHA256
`848276bfa79a90e7b9d9f85519763708ba7ff956cfed1938d8391063ec909c1d`.
This preserves all previously observed cases and noise. It is not a new holdout.
The previous plan bytes remain under `ef163b2-superseded-*`. Only the execution
source commit changes in these plans; target/noise/sigma draws, source selections,
learning rate, initial parameters, budget, cases and scoring remain unchanged.

The endpoint collector checks all 19,328 actual training draws, all 108 source
conditions and their nine choices (972 source/condition pairs), and all original
historical expression draws. Source PNG and latent identities are checked against
the independently sealed 24-image source pool. The CPU collector binds recorded
native embeddings; it does not claim to re-encode them.

Acceptance remains 1,510 development answers, both complete 1,800-answer functional
matrices, actual CLI replay, and all 796 PNGs / 3,980 raw readback rows (3,600 matched
answers). No failure is dropped and no best checkpoint is selected.

Local verification on 2026-09-14: `test_generated_source_draw_evidence.py`,
`test_complete_png_readback.py`, and `test_fresh_wording_validation.py` completed
with **34 passed in 621.84 seconds**. Changed Python entrypoints compiled and
`git diff --check` passed. These tests validate the evidence pipeline, not an
unexecuted model endpoint.
After rebinding to the actual `b62ec02` fix, the same complete test selection was
rerun: **34 passed in 638.96 seconds**. Together with the five source-augmentation
tests this is 39 passing tests for the corrected implementation and validation.

The first zero-update GPU preflight (`ae81916`) stopped before model loading because
its launcher omitted the snapshot environment supplied by the formal training
pilot. Its log and failed status remain under `ae81916-source-runtime-preflight`.
Probe source `b82228e4689f08be925452700d12b1da947f4e9a` reused the exact training
launcher's `snapshot_environment(bank)`. It still imports all training code from
the immutable `ef163b2` checkout. All 972 conditions were encoded and all 27
registered gradient draws executed. The final file recheck failed because source
paths had been registered as strings but the consumer requires `Path.open()`.
This is a failed preflight, not a successful endpoint. Its entire archive is
`b82228e-source-runtime-preflight-evidence.tgz`, 78,615 bytes, SHA256
`4728906f1e12b889817052ca3116cdc9b1b6abdadd427cd58195e9ccc30459f7`.
The complete local recount is in the adjacent local-verification JSON.

The path fix keeps `Path` keys and adds a test that installs all 24 PNGs into 108
conditions, executes the consumer's actual file hash check, and detects a changed
PNG. All five source-augmentation tests passed in 26.30 seconds.

Probe `c83aca055df8bb71b11e28ac242f0471f516d8da` selects the exact `b62ec02`
training checkout. Its first execution stopped because the sparse deployment
omitted `03-observed-wording-baseline-preregistered.json`. This deployment error
and its partial evidence are preserved separately; no runtime or gradient success
is claimed. The missing committed file has now been added to the clean sparse
worktree and its hash verified as
`5bc4d57752f770f6201cfc78b96da5884850b4af4c5b41de73715ba883d4335b`.
The next attempt uses a separate `c83aca0-source-runtime-preflight-r2` output.
The new training launcher requires this retry's full success before dispatch.
The partial `c83aca0` execution archive is 1,788 bytes, SHA256
`cca8c8c58e035e8608f70c22660da232d2defc0a66df8932b5bf36b4a350c2a7`.
Both archives were downloaded and independently recounted locally using
`scripts/reporting/verify_source_preflight_local.py`; neither is labeled passed.
After adding the missing sparse file, the exact remote training checkout passed
CPU execution of `source_pool_binding` and reconstruction of the entire training
plan, reproducing the registered `86a6bba8...` digest. GPU verification remains due.

As of these observations, the new four-H200 notebook remains pending and no new
optimizer updates have run. A completed zero-update probe still cannot replace
the full measured baseline, four-rank gradient parity, optimization, or functional
acceptance. The goal remains incomplete.

## Verified deployment for the next allocation

Project root is `/inspire/ssd/project/exploration-topic/czxs26210936`; the run root
is `runs/dreamlite-official-alignment` below it.

- Training: `repos/dreamlite-generated-source-training-r3-20260914`, clean
  `b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36`. This is an independent Git worktree;
  the earlier `training-r2` shared-clone attempt failed before deployment.
- Probe: `repos/dreamlite-generated-source-preflight-r3-20260914`, clean
  `c83aca055df8bb71b11e28ac242f0471f516d8da`.
- Validation: `repos/dreamlite-generated-source-validation-r2-20260914`, clean
  `2c5189a0847acd6653b687031ab13e6cd4cfc53f`.
- Queued notebook: `dl-source-aug-h200x4-20260914`, four H200s, still `PENDING`
  when checked at 2026-09-14 17:34:52 CST. Recheck actual allocation, environment,
  GPU processes and remaining lease before launching anything.

Three uploaded shell helpers accept an explicitly verified future Unix deadline:
`launch-generated-source-preflight-next-20260914.sh`,
`launch-generated-source-training-r2-20260914.sh`, and
`run-generated-source-full-validation-r2-20260914.sh`.
They are in the run root; shell syntax and successful uploads were checked.
The first produces `c83aca0-source-runtime-preflight-r2`; the second refuses to
start unless that exact retry passed, and produces `b62ec02-generated-source-full4832`.
The third uses the exact `2c5189a` source to execute both functional suites and PNG
readback. None of these next-run helpers has been launched. The original
`launch-generated-source-training-20260914.sh` now exits 75 immediately so the
known-bad `ef163b2` run cannot be launched accidentally.
