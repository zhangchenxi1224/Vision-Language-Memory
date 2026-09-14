# Generated-source continuation: complete validation registration

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
