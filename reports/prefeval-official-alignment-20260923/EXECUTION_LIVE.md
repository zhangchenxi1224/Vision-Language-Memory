# Live execution: official PrefEval A/B RGB memory, 2026-09-24

## Current objective and authority

User authorized implementation and evidence-driven iteration on 2026-09-24 using
the inspire skill. Only GPU notebook `dl-clear-retain-h200x4-20260914` is allowed.
No codex-with-chatgpt. Read `NEXT_EXPERIMENT_PLAN.md`; old Plan13 / direct28-gradient
plans are superseded. Target: demonstrate shared-Writer visual memory on unseen
preferences, question paraphrases and real RGB recurrence, then K2/K4 updates.
Teacher fit alone does not complete this objective. Preserve failures and original
PrefEval scoring; retain DreamLite native source conditioning / official FM.

Latest 2026-09-24 06:33 CST: student evaluation reached A330/462 and B328/462
conditions; all four existing Reader workers and both drivers remain alive.
No new GPU experiment failure. Common references completed308/308 conditions.
Both full Writer checkpoints and all616 student PNGs are now uploaded to GitHub;
all four large assets have matching source/GitHub SHA256. See `write-assets-manifest.json`.
T1 MCQ dev: blank36/90, full text79/90; train: blank30/64, full text59/64.
Read `WRITE_REFERENCE_RESULTS.md` and the complete archived reference matrix.
All student RGB endpoints are complete,308/308 per arm (154 records x2 seeds).
Four GPUs continue reading the fixed student PNG matrix, without new training.
Do not interpret these incomplete subsets as final paired scores.
Both shared FM-write runs completed2048/2048; teacher
PNG evaluation completed64/64 per arm. Read `FIRST_TEACHER_AND_FM_RESULTS.md`.
A T1 MCQ44/64; B64/64. B is64/64 on each of allfive forms; A O1/O2 joint40/64.
These are training-content teacher results.640 natural answers await the official
judge; no method winner or shared visual-memory success has been declared.

The app already has an unfinished historical Goal; `create_goal` rejected replacing
it. Its old instance/skill wording is superseded by the current user instruction and
this execution record. Do not falsely complete the old goal just to replace its text.

## Runtime and paths

- Live platform query: RUNNING, 4 H200 / 80 CPU / 900 GiB, node qb-prod-gpu2468.
- Current host: `dl-clear-retain-h200x4-20260914--a823c55e800a-facprpli2x`.
- At05:40 CST platform auto-stop was1h31m away (approximately07:12 CST).
  Recheck before assuming continued allocation; do not change the fixed evaluation
  budget to fit a session. Existing endpoint files support resumption.
  At the observed throughput, the remaining132/134 conditions per arm may outlast
  this allocation. Keep the running workers. If automatic stop interrupts them,
  confirm STOPPED before using `notebook start` on this same designated instance;
  refresh the host/GPU binding, confirm old processes are gone, and resume the
  original pilot with `--resume`. It reuses completed PNGs and condition files.
  Do not create another instance, alter quota, or restart a still-running worker.
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
  Reader tasks and controls. FM and teacher reading are complete. Fresh student
  RGB rollout has executed successfully; its full matrix/readout is still pending.
- Four targeted CPU checks pass: exact upstream MCQ format/parser, balanced labels/forms
  plus dev exclusion, Writer current-exchange-only inputs. No broad engineering suite.
- Fixed mismatched-image donors in `pilot-mismatch-controls.json`: all154 are from
  the same split/topic but a different registered semantic group and different
  correct-option text. The old next-row shortcut was corrected before any student
  evaluation. Different groups are not necessarily contradictory; this is a
  memory-dependence control, not a guarantee every donor implies a wrong answer.

**Pilot continuation drivers are RUNNING: references complete; student readings occupy all four GPUs.**
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

Next: finish the fresh single-write student RGB/readout matrix, then analyze
shared-Writer performance against the complete teacher matrix. First read
`FM_DIAGNOSTIC_NOTES.md`: all64 train-content benchmark exchanges use a different
acknowledgment from FM training (disclosure itself is identical for64/64).
Thus train-content benchmark scores are not exact training-condition fit.
After the fixed matrix, generate a paired diagnostic for all64 original SFT
initial exchanges with the same seed0/native inference and unchanged Reader
tasks. Keep it separate from the official benchmark. The diagnostic is implemented
in commit3227d58 and staged remotely; it has NOT been dispatched. Four focused
official-format/split checks and syntax compilation pass. Launch only after both
original pilot completion markers exist and the fixed matrix has been analyzed:
`launch_prefeval_official_ab.py train-input --output OUTPUT`. It uses four shards
(A0/A1/B0/B1,32 training records each), one original SFT initial exchange per state,
the frozen write checkpoint, native28 steps and the existing seed0/noise namespace.
No target-latent or FM update occurs. PNGs go to `rollouts/{A,B}/training-initial`;
readouts to `evaluations/training-input-students/{A,B}/write`, with the same five
question forms, answer limits, option order and official scoring. The full actual
initial exchange is saved in new PNG completion metadata. Distinguish input-generalization from
FM/free-running generation error before choosing the correction. Do not replace
benchmark acknowledgments with SFT replies. Then real train-side source rollouts and fixed2048
retain updates, followed by fresh benchmark RGB chains and same-PNG evaluation.
Use two fixed inference seeds and 0/5/10 prefixes. Keep O1/O2 out of selection.
Compare teacher / single-write student / recurrent student to locate failures.
Only extend data/capacity after these results, not from training loss alone.

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
