# PrefEval RGB memory: execution brief, 2026-09-17

Status: preparation complete/in progress; no new training or new-model result yet.
The user's goal is active. ChatGPT planning/review is required by the user.
Connection has been paired; waiting for the user to create/bind the ChatGPT
Project before the first planning request. Do not claim the experiment completed.

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

Next: finish ChatGPT Project binding and workspace_info validation, request a finite
PLAN from these files and current code; execute on a separate remote checkout/output.

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

ChatGPT session and current browser state still show no bound/new Project.
The first-time user Project creation request remains pending; no INIT or PLAN
has been sent, and there is no new model training process to monitor.
