# PrefEval RGB memory: execution brief, 2026-09-17

## Iteration 6: full-action ranking supervision

ChatGPT plan `prefeval-rgb-semantic-ranking-supervision-06` keeps the complete
Plan05 payload and prompts unchanged. `semantic-ranking-v1/registration.json`
binds an alternative temperature-1 listwise objective using the existing Reader
candidate-NLL implementation. First calibrate 2,688 text/blank ranking decisions
across all rotations, with at most 10,752 candidate forwards and no generation.
Require text >=1,210/1,344, semantic-group macro gain over blank >=0.10, and
each overwrite contrast both correct on >=12/16 pairs. Only a passing new
calibration permits the planned paired 20,480-update continuation. Plan05's
failed generation gate remains failed. 49 focused semantic, Reader-loss, EOS,
resize and visual-policy tests pass. Actual calibration results are pending.

## Iteration 5: paired semantic supervision continuation

Plan `prefeval-rgb-semantic-query-mixture-05` freezes 28 preference values,
112 training scenarios and 56 reserved scenarios in `semantic-transfer-v1/`.
First run exactly 672 text/blank feasibility generations; text must reach
303/336 before optimization. If it passes, continue each of the 40 archived
endpoints for 256 updates in each arm, with fresh optimizers: A recovery only,
B alternating recovery and full-action application supervision. Both arms
share the same initial endpoint per state and equal slot-forward counts.
Reserved probes remain unread until all 80 endpoints are frozen. Final
evaluation adds 3,408 generations. No Writer optimization or final test use.
Registration and execution code are committed at `f3ae518`; 49 focused tests
passed. The 672-read feasibility run completed at `9097be7` and FAILED:
text 94/336, blank 153/336. No paired optimization or reserved-probe reads
were run. Of 242 text failures, 239 give the exact correct full action but
add a forbidden option label; this diagnosis does not change the failed gate.
Read `semantic-feasibility-results.md` before planning the next intervention.

Status: the 124-write unchanged-4f PNG baseline is complete and independently
verified. The fixed 40-state visual-teacher sentinel completed at `191d24a` and
FAILED its prospective allocation gate: K1 20/20, K2 0/4, K3 1/4, K4 1/12.
No shared-Writer optimization has started. The old v2 combined gate stays failed;
the prospective recovery allocation is bound in `visual-recovery-allocation-v1.json`.
The user's goal is active. ChatGPT planning/review is required by the user.
The user created the Project with project-only memory; connection and workspace
identity have been verified. C2C task c2c_7e19 plan `prefeval-rgb-pilot-01` was
received and executed through its first empirical allocation gate. See
`pilot-plan.md`, `registered/summary.json`, and `references-v1-summary.json`.
Do not claim the experiment completed.

## Iteration 4: fixed endpoint factorial diagnosis

Plan `prefeval-rgb-endpoint-factorial-diagnosis-04` (review base `04a8d21`)
allocates zero optimizer/Writer updates. `endpoint-diagnostic-v1.json` binds all
40 existing endpoints and current-value MCQ lineage before inference. The
four conditions cross float/archived-PNG values with CPU/CUDA preprocessing;
the frozen Reader remains on GPU. Five existing forms per addressed slot give
1,760 recovery generations and endpoint CE forwards. The 84 active occurrences
also receive original MCQs in PNG, complete-text, and blank conditions (252
generations). No query change, state filtering, historical rescoring, or final
test use. All 176 archived-path held-out generations must replay exactly.
64 focused tests pass. All diagnostic cells completed at `517475f`, and the
176 archived-path reads replay exactly. CPU/CUDA changes no generated tokens;
quantization causes only a few flips. PNG training recovery is 255/264 and
held-out recovery 117/176; whole states are 38/40 versus 22/40. Teacher PNG
original MCQ is 30/84, blank 31/84, complete-text 84/84. Read
`endpoint-diagnostic-results.md` and `endpoint-verified.json` before proposing
more target optimization or Writer training. The eight bounded prefix-logit
inspections at `d3b2f5f` reproduce all eight observed divergent tokens, with
small teacher-forced gold margins; no new answers were generated or rescored.
Full raw/processor evidence is archived in four `endpoint-diagnostic-v1-shard-*.tgz`
files and a readable bundle, and unpacked under `visual-recovery-v1-run/endpoint-diagnostic-v1/`.

## Iteration 3: actual visual memory evidence

Plan `prefeval-rgb-visual-recovery-sentinel-03` uses full-value recovery for
prospective allocation while retaining all yes/no diagnostics unchanged. See
`visual-execution-notes.md` for technical startup receipts and exact execution
commits. The baseline used `3f3bf10` and was not repeated. Verification found
124/124 writes, singleton recovery 0/192, MCQ 33/96, recurrent recovery 16/176,
zero complete recurrent states and zero complete episodes. Source/output PNG
hashes, 28 native steps, registered seeds, and full query coverage were checked.

`baseline-verified.json` is the original-runtime verification. All PNGs and raw
token-bearing reads are in `baseline-shard0-evidence.tgz` and
`baseline-other-shards-evidence.tgz`, also unpacked under
`visual-recovery-v1-run/baseline/` for connector review. Archived byte hashes
match the execution environment. Windows/Torch 2.11 noise regeneration differs
from Linux/Torch 2.7; bitwise cross-runtime RNG portability is not claimed.

The exact native-gray teacher run is
`visual-recovery-v1-run/sentinel-native-gray-r2`, with 256 registered latent
updates per target. Preserve the whole 40-state panel and failed dependencies.
Do not replace this outcome with training loss or stop once enough targets pass.

All 40 targets completed exactly 256 updates (10,240 latent updates; zero Writer
updates). Independent reconstruction of the same reopened PNG's two recovery
forms gives 117/176 individual correct reads, but only 22/40 complete states.
The K1, K2, K3, K4 read counts are respectively 40/40, 12/16, 15/24, 50/96.
The registered allocation needs K1 >=18/20, K2 >=3/4, K4 >=9/12 and fails.
Active/positive/negative diagnostics remain 16/88, 36/84, 84/84. Four full
teacher chains all fail. Failed predecessors were retained without replacements.

Read `visual-verified.json`, `visual-failure-analysis.json`, and
`visual-results.png`. All raw generations, PNGs, initial/final latents, optimizers
and 256-step traces are preserved in `sentinel-k2-k3-evidence.tgz` and
`sentinel-k1-k4-evidence.tgz`; `sentinel-readable-evidence.tgz` is the compact
text-only review bundle. All are unpacked under the run directory above.
The full verifier passed in the original execution runtime; local teacher
reconstruction independently matched the remote result exactly.

Of 59 failed held-out recovery reads, 22 exactly match another active slot,
3 incorrectly report absence, 2 other failures concern a cleared slot, and
32 have other full-string errors. Low last-step training CE does not establish
fixed-endpoint training-query generation success or isolate PNG quantization.
The next narrow diagnostic should separate training versus held-out query
binding from float-to-PNG loss on these existing endpoints, before more steps
or shared-Writer scaling. Teacher PNG original-MCQ performance is unmeasured.
No dev/test target was optimized and no usable shared Writer is claimed.

## Iteration 2: frozen interface repair and observed limitation

Plan `prefeval-rgb-format-and-visual-sentinel-02` stopped at its first failed gate.
The same 2,048 blank/text reads plus 1,040 preregistered sentinel-form reads
completed on four H200s at code `0c0a543`, exit 0. Original membership, state
targets, options, scorers and v1 evidence are unchanged. The revised prompt
restores the exact pinned upstream MCQ template; recovery separates scope
labels from verbatim values and preserves literal quotation marks.

Raw/token reconstruction independently confirms:

| Text-reference family | Correct / total | Gate |
|---|---:|---|
| Original MCQ | 92/96 | pass (77 required) |
| Derived full-state recovery | 926/928 | pass (836 required) |
| Teacher training recovery | 263/264 | pass (90%) |
| Positive complete-statement check | 80/84 | pass (90%) |
| Negative complete-statement check | 66/84 | **fail** |
| Active-status check | 72/88 | **fail** |

These auxiliary failures are content judgments, not missing tags: negative
statements sometimes elicit `no` to active-status, and near-equivalent candidate
statements sometimes elicit `yes` despite nonidentical text. Even pilot-only
distinct operational semantic groups can contain similar meaning. No regrouping,
dropping or relabeling was performed. Two dev recovery failures mix a cleared
travel-activities slot with the untouched hotel slot. All failures remain in
`references-v2-verified.json` and all 3,088 outputs in the adjacent JSONL.
The unpacked `references-v2/` is accessible locally for connector review;
`references-v2-evidence.tgz` preserves every full generation/token record.

`references-v1-verified.json` reconstructs v1 unchanged (28/96 and 783/928),
checks complete coverage, expected jobs, EOS IDs and token counts, and verifies
the archived SHA256. `five-case-source-audit.json` binds all five original
residual MCQs to pinned source explanations, retaining official denominators.

At the close of iteration 2, prepared but unexecuted: corrected FP32 AutoencoderTiny teacher loader,
official gray-encoding equality check, sealed registered queries, pilot-only
foils, initialization/optimizer/fixed-endpoint/PNG evidence; a 124-native-write
PNG-only 4f baseline. Launcher refuses to run either if v2 gate fails.
45 focused protocol/Reader/EOS/Base/RGB tests pass. No visual capacity outcome
or new usable model can yet be claimed. Next review should resolve whether
unreliable auxiliary judgments belong in the allocation gate, preserving their
failed results and prioritizing the registered visual-memory experiment.

## Inputs and scope

- `user-proposal-semantic-scaling.md`: first user attachment, preserved verbatim.
- `user-proposal-state-updates.md`: second user attachment, preserved verbatim.
- Historical handoff: `../official-alignment-results-20260913/NEXT_EXPERIMENT_HANDOFF.md`.
- Repository starting HEAD: `0cd46b7de367570d5cfd0e2e85df709d63d72d93`.
- New branch: `codex/dreamlite-prefeval-rgb-20260917`.
- Keep focus on shared DreamLite visual-form memory. Do not expand into a new
  benchmark framework, other datasets, Reader tuning, LoRA, or extensive old gates.

## First-principles experimental question

Can one shared Writer update an RGB-only persistent state from the previous PNG
and the current explicit preference event, then answer independently posed
queries about unseen preference content through a frozen Reader? Separately
measure semantic diversity, simultaneous capacity, and actual rewrite count.
All queries for one state must use the same fixed PNG; no query-conditioned write,
no text-history carry, no per-test-state latent optimization.

Proposed narrow execution sequence, pending ChatGPT inspection and concrete plan:

1. Reuse PrefEval explicit adapter. Verify official MCQ alignment; group by base
   preference (including forms/derived states/duplicates) before any split.
   Existing 16-topic adaptation/4-topic OOD boundary is available; verify inherited
   training exposure before describing any split as unseen. Seal final tests.
2. Pilot on 64 train semantic groups; separate dev groups. Compare blank, text
   reference, fixed 4f Writer, and full-state visual teachers. Teacher supervision
   recovers preference content, not the official final MCQ letter. Joint capacity
   check at 1, 2, 4 preference units, including selective clear and overwrite.
3. Small/Full nested semantic subsets (approximately 25%/100%), same fresh optimizer
   and equal compute. Proposed start: lr 1e-5, batch 4, 20,000 updates/arm; validate
   pilot timing/learnability first. Distinguish fixed-budget diversity from convergence.
   Brief Base-vs-4f pilot only if useful to identify inherited bias.
4. Report original MCQ separately from derived recovery/state-update metrics;
   whole-state and episode success, changed-item and untouched-item accuracy,
   retention after 0/2/5/10 real writes, and paired source-image intervention.
   Select checkpoint by development PNG performance, report fixed endpoint too;
   final unseen tests do not tune training. Cluster uncertainty by semantic group.
5. Add equal-budget rollout-source continuation only if canonical-vs-self-generated
   source results identify that gap. Preserve failed teachers and report coverage.
   Iterate on the measured bottleneck; goal completion requires actual usable-model
   evidence and stated capacity/domain boundaries, not loss decrease or infrastructure.

Keep Full U-Net, frozen Reader/VAE/text encoder, Base native 28-step CFG1,
native conditioning and official FM. Source is condition only, never the FM origin.
Fresh inference starts Gaussian; episode memory begins gray. RETAIN executes Writer.

## Historical paired evidence

`compare_parent_chains.py` read all 960 raw cells in the two full RGB chain matrices
from verified archives; `parent-chain-comparison.json` preserves every failed cell.

- Original: 470 retained-correct, 10 still wrong, zero fixed, zero regressions.
- Wording: 420 retained-correct, 50 still wrong, 10 regressions, zero fixed.
- All errors are failure to clear (`no active preference` expected, old jazz/ambient
  returned), with the following RETAIN carrying the same error. The extra 10 errors
  are one repetition's CLEAR and RETAIN, each read with five query forms.
- This is historical regression evidence, not unseen-semantic evaluation.

## Live resource verification

- User-designated instance `dl-source-aug-h200x4-20260914`: RUNNING in
  分布式训练空间 / 前沿课题探索 / 开发区-H200-3号机房-2-cuda13.2版本.
- Actual 4 x NVIDIA H200, each 143771 MiB; 0 MiB used / 0% utilization at inspection.
- CPU ingress `dl-align-cpu-20260914-r3`: RUNNING in CPU资源空间.
- Project root `/inspire/ssd/project/exploration-topic/czxs26210936`.
- GPU Python `envs/vlm-r3-ngc2502/bin/python` verified: Python 3.12.3,
  PyTorch 2.7.0a0+ecf3bae40a.nv25.02, CUDA build 12.8/runtime 12.8.61,
  CUDA available, four H200s. Current driver 595.58.03. CPU notebook resolves
  the shared environment's Python symlink to Python 3.10.12, so use it for
  transfer/download only; run model code on the GPU runtime.
- Model root `/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`:
  DreamLite-base-a9a0f15-20260907, DreamLite-mobile, Qwen3-VL-4B-Instruct exist.
- 4f checkpoint `runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt`:
  4,681,178,881 bytes; live SHA256
  `7294684170578dfc617b4fafcea97e6480c08642ca4f1e5967ff8966aa103182`, identical to archive.
- CLI is installed in WSL Ubuntu, `/home/zhangchenxi/.local/bin/inspire`, v7.1.6.
  Use `wsl -d Ubuntu -- bash -lc ...`; restricted GPU notebook exec requires
  a local PTY (`exec_command tty=true`) to avoid implicit stdin rejection.
- No historical instance stopped/deleted; no weights or past results overwritten.

Next: execute the finite ChatGPT PLAN on a separate remote checkout/output.

## Continuation findings

The official PrefEval checkout at `50795054b5ff5f418d2b768a331d71e480f93331`
passed the existing adapter's alignment checks. Existing split counts are
730 train / 82 dev / 188 OOD base pairs (not yet the new experiment's final split).
`data-inventory.json` binds all source files.

`duplicate-inventory.json` identifies 988 normalized preference strings among
1,000 base pairs, with 11 duplicate clusters. Two clusters cross train/dev:
`lifestyle_beauty:0016` with `:0020`, and `shop_home:0026` with `:0028`.
They must stay on the same side in the new experiment. This check covers exact
alphanumeric normalization only; semantic equivalence remains unproven. Do not
report the unmodified 730/82 split as free of preference-content leakage.

ChatGPT Project is bound and workspace_info returned this exact workspace name.
The first PLAN is received. The four-GPU Reader reference run completed at code
`8d95cdd`: 2,048 read records, no trainable parameters. Text-reference strict
MCQ is 28/96; recovery is 783/928. Of the failures, 63 MCQs have the correct
letter in wrong tags and all 145 recovery failures are scope-prefix/quote-only
differences. These diagnostics do not replace the original failed strict scores.
Preserve `references-v1-evidence.tgz` and compact raw records; review the first
failed gate before changing task formatting or allocating teacher/Writer training.
The official PrefEval source was cloned and checked out at the registered commit
on the shared disk at `data/PrefEval-50795054-20260917`. The independent experiment
checkout is ready at `repos/dreamlite-prefeval-rgb-20260917`. It uses sparse
checkout for runtime code and this experiment's reports, avoiding historical
archive materialization. The one-time incomplete checkout was repaired before
dispatch; `git status --short` was clean at dispatch.
