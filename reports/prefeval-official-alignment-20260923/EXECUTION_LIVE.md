# Live execution: official PrefEval A/B RGB memory, 2026-09-24

## Current objective and authority

User authorized implementation and evidence-driven iteration on 2026-09-24 using
the inspire skill. Only GPU notebook `dl-clear-retain-h200x4-20260914` is allowed.
No codex-with-chatgpt. Read `NEXT_EXPERIMENT_PLAN.md`; old Plan13 / direct28-gradient
plans are superseded. Target: demonstrate shared-Writer visual memory on unseen
preferences, question paraphrases and real RGB recurrence, then K2/K4 updates.
Teacher fit alone does not complete this objective. Preserve failures and original
PrefEval scoring; retain DreamLite native source conditioning / official FM.

Latest 2026-09-24 03:03 CST: both shared FM-write runs completed2048/2048; teacher
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
- At initial query auto-stop was 5h45m away; recheck before assuming continued allocation.
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

**Pilot continuation drivers are RUNNING: common references, with early student RGB generation on freed GPUs.**
Launch commit `c13d154b5654dda25caadcbb5857ec214122b090`, session
`8ee9246d01204653a3d2d0b19b9f4d61`, remote `dispatch-pilot.json`.
A driver PID810688 owns GPUs0/1 after its teacher workers finish; B PID810689 owns
GPUs2/3. `pipeline/{A,B}/running.json` and `job-*.json` track children. Do not launch
duplicates. Each driver schedules FM-write and teacher evaluation, then common
references and fresh write-only benchmark RGB/student evaluation. It stops at the
fixed write-pilot endpoint for analysis before scheduling retain training.

Latest child sessions/receipts at 2026-09-23 19:08 UTC:

- A driver session `05035f9043f54f9cb09896925f5ad8b4`; FM PID1059861 GPU0
  and teacher evaluation PID1059862 GPU1 are finished. References PID1601241 GPU1
  runs shard0/2 of common blank/full-text inputs.
- B driver session `a86c5b7b60a74f70a1e352cc38477e48`; FM PID1071409 GPU2
  and teacher evaluation PID1071410 GPU3 are finished. References PID1604949 GPU3
  runs shard1/2 of common inputs.
- No new teacher/FM/evaluation failure observed. The two authoring failures remain
  in the raw evidence; no failed teacher was removed.
- Write checkpoints are fixed at2048 updates. A SHA
  `683366926f3351a82acb926e0fadd487ceacb8436f31478d723c005839bbe3d7`, B SHA
  `eea5387129d42ed156c2439cc2161a80515bcf1920011b203c1891d02b2da849`.
- Early planned shard0/2 RGB generation launched on freed GPU0/2 with
  `dispatch-write-rollout.json`, session `5da1a1f9763843fca009ad6fab33b710`,
  commit69c35c5; A PID1608267, B PID1608268.34 PNG endpoints per arm observed.
  This does not change examples, inference seeds, budgets or scoring. A per-shard
  file lock prevents a later original driver from generating the same shard
  concurrently; it reuses complete PNG markers. Do not dispatch the same job again.
- The remaining shard1 and student readings stay scheduled in the original drivers.
  Inspect progress before opportunistically advancing more work on freed GPUs.

Next: finish common references and the fresh single-write student RGB/readout matrix,
then analyze shared-Writer performance against the complete teacher matrix.
Then real train-side source rollouts and fixed2048
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

Both final Writer checkpoints (1,560,357,922 bytes each) are being downloaded via
CPU scp, timeout7200: A local session8324 -> `.cache/official-ab-20260924/writer-A-write-2048.pt`;
B session1745 -> `writer-B-write-2048.pt`. Verify the above checkpoint hashes after
completion, then upload them to the same experimental GitHub release. Do not
claim full weight synchronization until actual upload succeeds. No optimizer
checkpoints were included in this transfer; those remain on the shared disk.

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
