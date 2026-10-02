# Live execution: official PrefEval A/B RGB memory, 2026-09-24

## Current objective and authority

**2026-09-24 13:01 CST: Qwen judge STOPPED by API Arrearage.** Both local
processes6276/27440 have exited; stderr records HTTP400 with codeArrearage.
No automatic paid retry, model change or training restart was attempted.
5,168 successful checks were saved, covering1,288 complete answers of12,060
and10 partial answers. Zero malformed/truncated successful responses so far;
this incomplete, nonrepresentative subset is not a final A/B score.
Successful-response usage:3,083,739 input +163,182 output tokens; this is usage,
not a billing statement. Latest saved judgment12:49:09 CST.
Read `judge-qwen38-interruption-summary.json` and `QWEN38_JUDGE_STATUS.json`.
Raw judgments, both logs and failure receipts are archived in
`judge-qwen38-interrupted-evidence.jsonl.gz` (272 records,1,166,237 bytes,
SHA256 efcf61f9c6cb02561c67c7b9cb8cb096f11d16ad0ea08ce0f9b0922d67c5b07b).
Completion automation is now PAUSED, pending account recovery and user-directed
resume. Resume must reuse saved checks. Goal/new training remain paused.

**Latest judge instruction, 2026-09-24: use `qwen3.8-max` to score saved free
answers.** No new training is authorized. Read `QWEN38_JUDGE.md` and its launch
receipt `QWEN38_JUDGE_STATUS.json`. Existing local DashScope credentials now
work; the first real answer's four official checks completed successfully.
The remote write-ackmix A/B completion markers are both present; each has462
condition files plus two shard markers. Do not restart their finished Readers.
Their complete raw archive has942 records,4,746,006 bytes, SHA256
4b54b4d23b868ec45968745ee3a46977d7582031276762d231bae2c43a18c785.
Generation scores will be labeled as official PrefEval prompts/parsers/aggregation
with a substituted Qwen judge, not the original Sonnet judge configuration.
Older claims below that no judge key is configured are superseded.

**Latest user instruction, 2026-09-24: new iterations are PAUSED. Let the
currently running remote experiment finish.** Goal is paused. The existing
write-ackmix drivers/readers were resumed after a brief SIGSTOP; no new run was
started. Receipt `EXISTING_RUN_CONTINUES_20260924.json` supersedes
`PAUSED_BY_USER_20260924.json`. At11:31 CST the fixed matrix had A354/462 and
B357/462 completed conditions, with the four original Readers active.
Do not start new training, diagnostic, retain, K2/K4 or recovery jobs. Monitor
only the existing dispatch, archive and report its complete results, then pause
the completion-monitor automation. If interrupted, preserve/report the state
and wait for user direction; do not restart the instance or training.
Read `COMPLETED_EXPERIMENTS_REVIEW_20260924.md` for the full completed-experiment
review. All older next-iteration instructions below are historical and do not
override this pause. Incomplete readout subsets are not final performance.

User authorized implementation and evidence-driven iteration on 2026-09-24 using
the inspire skill. Only GPU notebook `dl-clear-retain-h200x4-20260914` is allowed.
No codex-with-chatgpt. Read `NEXT_EXPERIMENT_PLAN.md`; old Plan13 / direct28-gradient
plans are superseded. Target: demonstrate shared-Writer visual memory on unseen
preferences, question paraphrases and real RGB recurrence, then K2/K4 updates.
Teacher fit alone does not complete this objective. Preserve failures and original
PrefEval scoring; retain DreamLite native source conditioning / official FM.

Latest execution check 2026-09-24 10:30 CST: BOTH write-ackmix FM endpoints and
all616 benchmark PNGs are COMPLETE (308 per arm). Four Reader workers are now
running the fixed matrix: A191/462 and B194/462 condition files at that snapshot,
no completed Reader shard yet. A0 PID1184536 GPU0, A1 PID1184538 GPU1,
B0 PID1184456 GPU2, B1 PID1184457 GPU3; all four hold about9.4 GiB GPU memory.
Parent drivers722505/722506 remain alive. Former rollout children have exited.
No new failure receipt. Do not interpret incomplete readout subsets as final
paired scores. Wait for the original drivers; do not redispatch any live reader.

Both arms have8192 draws over64 training states,128 conditions, exactly64 draws
per acknowledgment version per state. No dev state or retain position entered
training. `ackmix-training-summary.json` and `ackmix-training-evidence.jsonl.gz`
are locally verified and uploaded to the release. The gzip has15 records including
all4096 optimization rows;919,045 bytes; SHA256
32dce16b3244a0dd609d54b41d71de17e8b572a285b6152a10996f288dc3f806.
Parameter files are1,560,539,490 bytes each. A SHA256
bfe2393ac7ad48bd492c32bccf0709777fd3469a70b6357ff577ffbb05cbe03e;
B229f714cf94278be495fab6bd5648e32973bb202c57818dfd4288572fd64eb2e.
Direct shared-disk Writer upload session16674 COMPLETED. A asset584909618 and
B584909646 each have1,560,539,490 bytes and GitHub SHA256 matching the source
checkpoint hashes above. Remote `writer-release-upload-write-ackmix-{A,B}.json`
and combined receipt exist. Do not repeat completed uploads.
All616 new PNGs are now fully uploaded; session4097 COMPLETED successfully.
Archive A `student-rgb-A-write-ackmix.tar`:384,788,480 bytes, asset584959962,
SHA2568bb7c2ecaf1b58dad07024fe8c71f1e9fc21ac3923f20949e58788fd00b599df.
Archive B `student-rgb-B-write-ackmix.tar`:278,763,520 bytes, asset584959960,
SHA2561bd77fa79ff9a1d38b29b928b54383dd3a4b47409470dd19217e0d0e4b9dec38.
All source PNG hashes were checked before packing; both GitHub tar digests match.
Remote receipts `student-png-release-upload-write-ackmix-{A,B}.json` and combined
receipt exist. `ackmix-assets-manifest.json` records completed Writer/PNG assets
and training evidence. Do not repeat any of these transfers. Full optimizer
states remain on shared disk. No corrected performance claim is available until
the full paired readout completes. No new training/evaluation process was launched
by this check; all four original Readers and both drivers remain alive.

Latest 2026-09-24 08:30 CST: exact-training-input diagnostic is COMPLETE,
all128 PNGs/readouts and four pipeline markers. All former workers exited.
Read `TRAIN_INPUT_RESULTS_AND_ACKMIX_PLAN.md` and the archived diagnostic summary.
T1 exact SFT input: A38/64, B62/64; same-seed benchmark input A29/64, B30/64;
teachers A44/64, B64/64; blank30/64. B gains32 correct /0 lost merely by restoring
the SFT acknowledgment, with unchanged weights/disclosures/noise. This locates a
major input-expression generalization failure, rather than universal FM failure.
It proves neither unseen-preference memory nor recurrence/K4. There are now7,440
natural answers pending the official judge; no A/B method winner is declared.

The complete first write pilot remains A69/180 / B71/180 on dev T1 over two
correlated seeds, vs blank36/90 and text79/90. See WRITE_PILOT_RESULTS.md.
Correction `write-ackmix` used each arm's own fixed
write endpoint, unchanged64 targets, fresh AdamW and2048 additional official FM
updates. Each training disclosure alternates its released SFT acknowledgment
and cached Reader acknowledgment,64 draws per version. Dev data is excluded.
The64 Reader acknowledgments become training conditions; their later scores are
training fit. The90 dev preferences still test transfer. No OOD tuning.
Five targeted checks and syntax compilation pass. Correction launched once at
08:39 CST, session `de87783932384e7b813c5c7cd76d945e`, commit e6f379f0fce1ae8dc6ebf9c298ac66295ddec3f0.
`dispatch-ackmix.json` records current host suffix3eiz2r5ftf and GPU binding.
A driver PID722505, session6e0d5023532c49f39dfc36e3b02c3e42, owns GPUs0/1;
FM child PID722509 runs on GPU0. B driver PID722506,
session59cf1f27561142ecbc0574ec41560e9d, owns GPUs2/3; FM child PID722510 GPU2.
Both logged27/2048 steps in the first progress check and have since completed
2048/2048. Former FM children exited; current rollout PIDs are at the top.
Do not duplicate dispatch.
Drivers under `pipeline-ackmix/{A,B}` schedule FM2048, then the fixed benchmark
RGB/readout shards, and stop for analysis. Check their `job-*.json`, logs and
completion markers. New artifacts use stage `write-ackmix`; original `write`
endpoints and references stay frozen. No retain/K2/K4 is queued. The correction
adds both input diversity and optimizer steps; do not attribute any improvement
solely to augmentation without a matched budget control.

Historical exact-input dispatch b2187482df5649ae812ffde225bec0a2 (commit1ea44f8)
finished successfully on the current host suffix3eiz2r5ftf. Completed outputs:
`rollouts/{A,B}/training-initial`, `evaluations/training-input-students/{A,B}/write`
and `pipeline-train-input/{A,B}/complete-{0,1}.json`. Do not relaunch it.
`training-input-evidence.jsonl.gz` has273 records,800,238 bytes and SHA256
2a7f6031b4b1e184d2aaea0982c998f0418d51b3d4cbb597004ace17c485c8eb;
local hash/records verified. All64 training states per arm remain in the evidence.
Full diagnostic raw answers/summary and all128 PNGs are now uploaded to the same
GitHub release. A training-initial tar85,514,240 bytes SHA256
c9f3f9346e310fd9bfaf7ac823f8121acc1447d98b73677661dede0418a2f663;
B63,621,120 bytes SHA2563af015fbf7e947163e6e78df962be310630039ee75de9e799c1c4aca49c4843a.
GitHub digests match source archives; upload session65658 completed.
Remote receipts `training-input-png-release-upload-{A,B}.json`.

The complete write evidence archive has1,398 raw records and7,004,601 bytes;
SHA2565956ab0ac0c7b8800fe4d83ef6180b41f5958096b1eaedd1f8ff3360042747d9.
Local gzip/hash/record count verified; full summary, raw evidence and manifest
are uploaded to the existing GitHub release. Both full Writer checkpoints,
all616 benchmark student PNGs and128 teacher endpoints are already uploaded.

The app already has an unfinished historical Goal; `create_goal` rejected replacing
it. Its old instance/skill wording is superseded by the current user instruction and
this execution record. Do not falsely complete the old goal just to replace its text.

## Runtime and paths

- Live platform query: RUNNING, 4 H200 / 80 CPU / 900 GiB, node qb-prod-gpu2468.
- Current host: `dl-clear-retain-h200x4-20260914--a823c55e800a-3eiz2r5ftf`.
- Old host `...-facprpli2x` automatically stopped after8h at about07:12 CST;
  STOPPED was confirmed07:13:55, then `notebook start --no-wait --post-start none`
  restarted the same notebook. It became RUNNING on the same node with four H200s
  and a fresh8h window at about07:16. Old processes were absent before resume.
  No instance was created, quota changed, unrelated task stopped or live worker restarted.
- The restart retained A434/462 and B428/462 conditions and both shard0 completion
  markers. New readers have already saved further conditions. Reuse endpoint files
  on any future interruption; never shorten the fixed matrix to fit a session.
- CLI: WSL Ubuntu `/home/zhangchenxi/.local/bin/inspire` 7.1.6. GPU notebook exec
  needs a local PTY; CPU transfer uses `dl-align-cpu-20260914-r3`.
- Root: `/inspire/ssd/project/exploration-topic/czxs26210936`.
- New clean Git checkout: `ROOT/repos/dreamlite-prefeval-official-20260924`.
- Output: `ROOT/runs/dreamlite-prefeval-official-ab-20260924`.
- Python: `ROOT/envs/vlm-r3-ngc2502/bin/python`.
- Models: `/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`.
- Code is sparse Git clone/pull of `codex/dreamlite-prefeval-rgb-20260917`, not the
  older remote checkout with a separate local Git history.

## Started stages

Question authoring completed for all 154 = 64 pilot train + 90 dev records.
Frozen Qwen3-VL-4B saw only original questions, no preference/answer/options.
Two original workers stopped on duplicate forms; raw drafts/failures were retained.
Resumed draft collection preserved already authored records. Codex reviewed and
corrected pronoun role reversal, task drift, dropped constraints and duplicates
before any latent optimization. All T1 strings remain original. T2/T3 are training
paraphrases; O1/O2 are evaluation-only. Final question bank:
`pilot-questions-3plus2.json`, SHA256
`59fa2719c952547de88563d80ecd8ec03900b59fde9c87900682faeab5f7e3ec`.
Authoring sessions: `9237470ed5ef4ba49478098ab454cd8d` then
`290db340dd8b440390c1497e1606820f`. Original draft archive is in the run folder as
`questions-draft.tgz` and locally `.cache/official-ab-questions-draft.tgz`.

**Paired target-latent training is COMPLETE: 64/64 states per arm.**

- Launch commit: `dc0b6b53c9a8811a97b445e5d5996fba44c96242`.
- Session ID: `57459d0119c04fa0a750593b209da6dc`.
- Remote receipt: `dispatch-train.json`; use it before any resume/launch.
- Worker PIDs: A0=667197 (GPU0), A1=667198 (GPU1), B0=667199 (GPU2), B1=667200 (GPU3).
- Each shard has 32 train states; 288 updates/state, 96 per T1/T2/T3. Fresh Adam .05.
- A official full-answer mean CE; B official XML short-answer mean CE; end token
  included once in the token average. VAE/Reader frozen; real uint8 forward + STE.
- By 2026-09-23 18:17 UTC all128 endpoints were complete, with
  finite nonzero image gradients. B loss near zero is a training fit observation,
  not a memory-capacity or generalization result.
- Files: `teachers/{A,B}/{topic-index}/optimization.jsonl`, `resume.pt`, `latent.pt`,
  `memory.png`, `original-generation.json`, `complete.json`. Checkpoint every24 steps.
- Resume only after previous workers are dead and inputs match; attempts distinguish
  replayed updates after the last durable checkpoint. Do not count retries as new budget.

## Implementation and next actions

- `scripts/experiments/prefeval_official_ab.py`: author / target training.
- `scripts/inspire/launch_prefeval_official_ab.py`: four workers and launch receipts.
- `scripts/train/train_prefeval_official_fm.py`: two shared Writer stages using the
  existing `official_flow_bridge`, `predict_velocity`, native condition encoder and
  4f parameter export. New bank retains all finite endpoints, avoiding the older
  loader's fixed256 / successful-short-answer-only contract.
- `scripts/eval/prefeval_official_rgb.py`: native RGB rollouts, official-format
  Reader tasks and controls. The complete fixed teacher/reference/student write
  matrix is archived. Exact-training-input diagnostic readings are now pending.
- Four targeted CPU checks pass: exact upstream MCQ format/parser, balanced labels/forms
  plus dev exclusion, Writer current-exchange-only inputs. No broad engineering suite.
- Fixed mismatched-image donors in `pilot-mismatch-controls.json`: all154 are from
  the same split/topic but a different registered semantic group and different
  correct-option text. The old next-row shortcut was corrected before any student
  evaluation. Different groups are not necessarily contradictory; this is a
  memory-dependence control, not a guarantee every donor implies a wrong answer.

**Completed resumed pilot: both driver and all evaluation shard markers exist.**
Resume dispatch `961a79b635a746faba7b369be48d0122`, commit
`b72776c391e5547982feab21bb5b24a3fb9dd2ed`, new host `...-3eiz2r5ftf`.
`dispatch-pilot.json` contains the new host/GPU binding; the original dispatch was
preserved as `dispatch-pilot-8ee9246d01204653a3d2d0b19b9f4d61.json`.
- A driver PID8080, session `36dd762f273a4e028eb59bca72ab7149`;
  remaining Reader PID12555 GPU1.
- B driver PID8086, session `37e3358541634d49b0624af673805724`;
  remaining Reader PID12553 GPU3.
All resumed Reader and driver processes have exited successfully. All teacher,
FM2048, references, rollout and student shard0 endpoints were reused. Do not
launch another pilot. These PIDs and receipts are historical; monitor the
train-input diagnostic described at the top instead.

**Historical pilot on the previous host (ended by automatic stop):**
Launch commit `c13d154b5654dda25caadcbb5857ec214122b090`, session
`8ee9246d01204653a3d2d0b19b9f4d61`, remote `dispatch-pilot.json`.
A driver PID810688 owns GPUs0/1 after its teacher workers finish; B PID810689 owns
GPUs2/3. `pipeline/{A,B}/running.json` and `job-*.json` track children. Do not launch
duplicates. Each driver schedules FM-write and teacher evaluation, then common
references and fresh write-only benchmark RGB/student evaluation. It stops at the
fixed write-pilot endpoint for analysis before scheduling retain training.

Latest child sessions/receipts at 2026-09-23 21:03 UTC:

- A driver session `05035f9043f54f9cb09896925f5ad8b4`; FM PID1059861 GPU0
  and teacher evaluation PID1059862 GPU1 are finished. References PID1601241 GPU1
  finished shard0/2 of common blank/full-text inputs and exited.
- B driver session `a86c5b7b60a74f70a1e352cc38477e48`; FM PID1071409 GPU2
  and teacher evaluation PID1071410 GPU3 are finished. References PID1604949 GPU3
  finished shard1/2 of common inputs and exited.
- No new teacher/FM/evaluation failure observed. The two authoring failures remain
  in the raw evidence; no failed teacher was removed.
- Write checkpoints are fixed at2048 updates. A SHA
  `683366926f3351a82acb926e0fadd487ceacb8436f31478d723c005839bbe3d7`, B SHA
  `eea5387129d42ed156c2439cc2161a80515bcf1920011b203c1891d02b2da849`.
- Early planned shard0/2 RGB generation launched on freed GPU0/2 with
  `dispatch-write-rollout.json`, session `5da1a1f9763843fca009ad6fab33b710`,
  commit69c35c5; A PID1608267, B PID1608268. Both workers finished154 PNG endpoints
  per arm (77 records x2 seeds) and their processes exited.
  This does not change examples, inference seeds, budgets or scoring. A per-shard
  file lock prevents a later original driver from generating the same shard
  concurrently; it reuses complete PNG markers. Do not dispatch the same job again.
- Early planned shard1/2 launched at19:47 UTC on the same freed GPU0/2,
  `dispatch-write-rollout-1.json`, session `c8cfa81ba0bf496e97d51b1a6df499d0`,
  commit `5f9fc13e119603e186e973ae8eff8196ef2b9718`;
  A PID2147450 GPU0, B PID2147451 GPU2. Both rollout workers finished; their
  processes were gone by20:24 UTC. Both arms have all308 PNG completion markers.
- Original drivers successfully reused both completed rollout shards and launched
  their scheduled student workers. All308 common reference conditions are complete.
- Early student shard0/2 launched at20:26 UTC, receipt `dispatch-write-students.json`,
  session `9d09cb75edfe4eb387243e4342952ff5`, commit
  `5fc44d061ebb43850798fec1e1d1b7750584e461`; A PID2605840 GPU0, B PID2605843 GPU2.
  Both Reader models allocated GPU memory and produced their first matched and
  mismatched condition files. Evaluation path is `evaluations/students/{A,B}/write`.
  Expected per arm154 records x(2 matched seeds +1 mismatched)=462 condition files,
  each containing five question forms x(generation +MCQ). Do not redispatch.
- Commit5fc44d0 adds a student per-shard lock and returns before model loading when
  an evaluation shard is complete. Completed rollout shards now verify their
  parent checkpoint hash and return before allocating a model. This lets the
  original drivers later join the early shard0 readers safely and launch shard1
  on GPUs1/3 after their reference workers finish. Samples, budgets and scoring
  are unchanged. Local syntax compilation passed; first GPU readout files exist.
- Actual shard1 Readers: A PID2894200 GPU1 (driver session05035f9043f54f9cb09896925f5ad8b4),
  B PID2936584 GPU3 (a86c5b7b60a74f70a1e352cc38477e48). Driver-spawned shard0
  PIDs2894199 /2936583 wait on the already running early shard0 workers' locks;
  they allocate no model and will reuse the completed evaluation. Four GPU model
  processes are2605840 /2894200 /2605843 /2936584. Do not start additional readers.

Next: monitor the running acknowledgment correction documented at the top, then
finish its full154-state benchmark matrix and paired T1 analysis. The original
exact-input diagnostic is complete and must not be relaunched. Preserve the
original write results; all new weights/PNGs/readouts use `write-ackmix` paths.
Both original acknowledgments and Reader acknowledgments constrain the same
training target. Confirm held-out90-dev gains beyond blank/mismatch before
moving to actual training-source retain or K2/K4. Neither lower FM loss nor
better64-train fit is evidence of new-preference generalization.

The official generation judge uses Claude3 Sonnet through Bedrock. No local API
environment variables or AWS credentials profile were found; user has been asked
for an existing teacher API configuration location (never request pasted secrets).
Continue training, MCQ and raw generation while this is unresolved. Generation
outputs remain `pending_official_judge`, not silently scored with string matching
or the same small Reader. Do not claim official generation accuracy without it.
CPU notebook also has no relevant API environment flags or AWS config/credentials.
`scripts/eval/judge_prefeval_official_rgb.py` is ready to import the pinned upstream
judge parsers and aggregation and read the original four prompt files. It uses
the released Sonnet/Bedrock parameters; no API calls or formal generation scores
have occurred. Save the user's supplied configuration location in private runtime
config, never keys in Git. First T1 dev selection can be specified with
`--glob 'students/*/write/*.json' --split dev --families T1` before the full matrix.

## Continuation metadata

Complete reference summary and `reference-and-rollout-evidence.jsonl.gz` are now
downloaded into this report directory and uploaded to the existing GitHub release.
The gzip has1,101 parsed raw records,1,827,394 bytes, SHA256
`4077f20bf9de1901acaa0f5d587d81d41a2a9442439bff4a49396d6599c80ab4`.
It contains full reference answers and PNG generation metadata; actual student
PNG binaries are now also in separate complete Release archives. Remote
`write-evaluation-progress-20260923T2103.json` is only an interim student snapshot;
it is not the final shared-Writer result. Do not tune from its partial subsets.

Teacher raw text records and manifest are downloaded locally and published at
https://github.com/zhangchenxi1224/Vision-Language-Memory/releases/tag/prefeval-official-ab-teachers-20260924
and will also be tracked in this report directory. Full banks A148,888,318 bytes /
B125,634,394 bytes are archived on the shared disk under `OUTPUT/archives`.
The first scp session78878 timed out at600s, leaving an incomplete B archive.
Retry19009 completed with timeout3600. Both complete bank hashes were verified at
`.cache/official-ab-20260924/archives/archives/teacher-bank-{A,B}.tgz`.
GitHub upload session4088 completed. Both bank assets show uploaded with the exact
manifest byte sizes; the complete teacher evaluation summary and write/teacher raw
evidence are uploaded too. All128 teacher PNGs/latents and their optimizer endpoints
are now in the release. The old incomplete local B file is not a valid archive.
Windows may show zero file size until WSL closes a download; WSL stat shows progress.

Both original Writer downloads (1,560,357,922 bytes expected each) TIMED OUT after
7200s. Local sessions8324 /1745 have exited. Their incomplete local files are
`.cache/official-ab-20260924/writer-A-write-2048.pt` (238,924,800 bytes) and
`writer-B-write-2048.pt` (264,253,440 bytes); do not load or upload these partial
files. The complete remote checkpoints and their hashes are unchanged. No
optimizer checkpoints were included; those remain on the shared disk.
CPU notebook has no `gh` executable. The complete teacher banks are already uploaded.
A direct shared-disk-to-GitHub upload attempt (local session24026, SSH stdin)
failed with remote-command exit255. Follow-up found no upload process, no upload
receipt and no Writer assets on GitHub. It did not affect any GPU worker or change
the weights. A short non-secret stdin probe later succeeded. Committed direct
uploader `scripts/reporting/upload_prefeval_writer_assets.py` avoids the long
inline command. Its first run (local46525) exited before upload because the CPU
environment has no `requests`; commit400ebf4 uses only standard-library HTTPS.
Direct-upload session73524 COMPLETED successfully. Both Writer files are exactly
1,560,357,922 bytes, GitHub asset IDs584642828 /584642831; GitHub digests equal the
training hashes above. Remote `writer-release-upload-{A,B}.json` and combined
`writer-release-upload.json` preserve receipts. Authentication was transient stdin-only.
Student-PNG upload session92986 also COMPLETED successfully, using
`upload_prefeval_writer_assets.py --kind student-pngs`. It verified each original
PNG against its rollout marker and archived all308 endpoints per arm:
- `student-rgb-A-write.tar`:354,949,120 bytes, asset584690289, SHA256
  `ac64e1a8ed9be6aec4e55ec2605cc32cad3a88fdc2ace180f5d303038a300f00`.
- `student-rgb-B-write.tar`:255,866,880 bytes, asset584690279, SHA256
  `f3e3a65b6faada53be8fabfdb48c3ec6cb9cb6269cb04b5a593731ac4a1b77d7`.
All four assets are `uploaded` with matching GitHub SHA256; local
`write-assets-manifest.json` records the verified API metadata. Do not duplicate
these transfers. Full Writer optimizer states remain on shared disk.
A one-image local inspection transfer timed out at90s; that partial file is not a
valid local student PNG and yielded no visual diagnosis. This did not affect
the complete archives uploaded directly from the CPU notebook.

Read-only training analysis is archived in `fm-training-geometry.json` and
`writer-input-condition-comparison.json`, with scripts and interpretation in
`FM_DIAGNOSTIC_NOTES.md`. No development/OOD score was used in these diagnostics,
and no new model training was launched. Target-noised training residuals decreased,
but do not establish free-running inference or task-level memory preservation.

`teacher-training-summary.json`: each arm64 states x288 =18,432 optimizer updates.
A exposed4,019,040 target tokens; B129,024 (including end tokens). Recorded training
step time sums are3041.68s /3058.43s; these exclude model loading, checkpointing and
free generation and are not complete billed GPU hours. Last-cycle mean CE is
0.311666 /0.00000758, comparing different targets and not a performance ranking.
Unscored diagnostic original generations reached300-token truncation in23/64 A
and63/64 B; this observation motivates checking free-answer transfer, not changing
one arm's generation limit. Formal fixed-prompt paired evaluation is still running.

Old `dreamlite` automation could not be found in local TOML or the app automation
database (zero automations). A new thread heartbeat `dreamlite-a-b` was successfully
created ACTIVE, every30 minutes, with this experiment's current constraints and
quiet-unless-meaningful-change instructions. No codex-with-chatgpt, no old Plan13.
Remote jobs continue independently of foreground execution. Save new status here
and at the top of START_HERE / PLAN10_LIVE at important milestones.
