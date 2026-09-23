# Live execution: official PrefEval A/B RGB memory, 2026-09-24

## Current objective and authority

User authorized implementation and evidence-driven iteration on 2026-09-24 using
the inspire skill. Only GPU notebook `dl-clear-retain-h200x4-20260914` is allowed.
No codex-with-chatgpt. Read `NEXT_EXPERIMENT_PLAN.md`; old Plan13 / direct28-gradient
plans are superseded. Target: demonstrate shared-Writer visual memory on unseen
preferences, question paraphrases and real RGB recurrence, then K2/K4 updates.
Teacher fit alone does not complete this objective. Preserve failures and original
PrefEval scoring; retain DreamLite native source conditioning / official FM.

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

**Paired target-latent training is RUNNING.**

- Launch commit: `dc0b6b53c9a8811a97b445e5d5996fba44c96242`.
- Session ID: `57459d0119c04fa0a750593b209da6dc`.
- Remote receipt: `dispatch-train.json`; use it before any resume/launch.
- Worker PIDs: A0=667197 (GPU0), A1=667198 (GPU1), B0=667199 (GPU2), B1=667200 (GPU3).
- Each shard has 32 train states; 288 updates/state, 96 per T1/T2/T3. Fresh Adam .05.
- A official full-answer mean CE; B official XML short-answer mean CE; end token
  included once in the token average. VAE/Reader frozen; real uint8 forward + STE.
- Progress observed through step264 for first states in all four workers, with
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
  Reader tasks and controls. New Writer/rollout integration still needs its first
  actual GPU execution; do not describe it as validated or completed yet.
- Three targeted CPU checks pass: exact upstream MCQ format, balanced labels/forms
  plus dev exclusion, Writer current-exchange-only inputs. No broad engineering suite.

Next: finish all 64 targets per arm, evaluate teacher PNGs and common text/blank
references; start the fixed 2048-update shared-FM write pilot for each arm without
filtering failed teachers. Then real train-side source rollouts and fixed 2048
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

## Continuation metadata

An attempt to update the existing `dreamlite` heartbeat prompt with partial fields
was rejected because name/rrule/status are required. The existing automation's
local TOML has not been found; do not claim its prompt was updated. Current run
is independent of local foreground execution. Save new status here and at the top
of START_HERE / PLAN10_LIVE when reaching an important new milestone.
