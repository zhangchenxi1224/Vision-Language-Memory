# Plan13: within-state scope/value contrast

Plan received 2026-09-21 in the original C2C project conversation, following
EXECUTED12. C2C task `c2c_7e19`; execution iteration13; scientific plan ID
`prefeval-rgb-scope-contrast-13`. Review base `ad27821`.

## Scientific decision

Plan12 W is rejected for this recipe. M has 941/968 training-form and142/176
excluded-form recoveries; W has915/968 and130/176. All27 remaining M training
failures concern untouched slots.14/34 excluded M failures literally reproduce
another current slot value. This establishes a failure class, not the mechanism
of every error, and says nothing yet about recurrent Writer forgetting.

Run one paired32-update experiment from all40 Plan12 M endpoints, including
failed multi-slot and selective-clear states. Parent registration digest:
`cb325203bbe45020bd68b68e4a62ef7ec2037989da7f2dd4cc3d126c7e12774a`.
Inherited optimization `[256,128,64,64,64,32]`; new updates separate.
Fresh Adam per arm/state: lr .01, betas .9/.999, eps1e-8, weight decay0.

C13 uses Plan12 weighted mean recovery plus .25 slot-weighted application.
S13 adds .25 weighted mean scope-contrast loss. Both replay all11 training
forms for every addressed slot at every update. Preserve changed/untouched
weights, absence recovery, unallocated cleared-slot application weight, and
the exact32-draw old/compositional application schedule (half-fraction).

## Contrast definition

For a scope, rivals are other distinct current values in the same registered
state plus exactly `no active preference` when that is not the gold. Omit
duplicates and scorer-equivalent rivals; zero loss if none remain. Gold strings
stay verbatim. Rivals come only from state annotations, never observed failures,
qualification answers, outside preferences, or invented content.

Tokenize gold/rivals with the actual chat prefix and assistant terminator,
using the existing contextual tokenization contract. At their first differing
token d, compare gold and rival logits from the existing gold forward:
delta=logit(gold_d)-logit(rival_d). Deduplicate identical(d,rival_token) branches.
Per-query contrast is the mean max(0,2-delta), with EOS handled as a token.
Use slot_weight/11 to combine queries. Margin2 and coefficient.25 are fixed.
Compute diagnostics in both arms; only S13 includes their gradient.

Construct the full scalar per-query objective before its gradient. Stream
gradient buffers, then perform one optimizer step after recovery/application.
No negative-continuation forwards. Reader/VAE, precision, latent layout,
preprocessing, quantization, decoding and generation settings stay unchanged.
Rivals are offline annotations and never enter a generation prompt.

## Budget and evaluation

2,560 updates;83,456 gradient Reader forwards (41,728/arm),61,952 recovery
and21,504 application forwards. Same input-token counts expected; record actual
tokens/runtime. Exactly32 updates, no checkpoint selection or automatic extension.

Freeze all80 endpoints first. Save/reopen one PNG per state/arm. Reuse all
Plan12 panels:5,320 generations,2,256 rankings/9,024 candidate forwards,
2,288 recovery CE calls. During those CE calls record branch margins, token
NLL, gold margins and EOS. Excluded forms are evaluated after freeze only.
Reuse M/E historical results; no new blank/text references or final-test access.

Primary outcomes are free-generation recovery and same-PNG completeness.
Report C13/S13/M paired gains/regressions by capacity, changed/untouched status
and semantic group; exact requested-scope/returned-value confusion, incorrect
absence, unmatched content, wording and truncation; all clear states, chains,
MCQs, application panels and both counterfactual/overwrite types. Every active
slot must pass both excluded recovery and both reserved ranking queries on the
same PNG; cleared slots must recover absence. Also report conjunction with MCQ.

Unchanged gates:K1>=18/20,K2>=3/4,K4>=9/12,MCQ>=68/84,
reserved ranking>=135/168, historical overwrite>=6/8 and>=1/group.
Report K3 independently. Prior failed E-R comparison remains failed.
Compare application preservation against C13 and starting M. Better margins
alone do not qualify teachers. Writer updates remain zero pending independent
review of teacher gates/coverage; no recurrent-memory claim.

## Execution

Driver `scripts/experiments/prefeval_scope_contrast.py`; verifier
`scripts/reporting/verify_prefeval_scope_contrast.py`; paired report
`scripts/reporting/summarize_prefeval_scope_pairs.py`; focused tests
`tests/test_prefeval_scope_contrast.py`. Tests cover prefix/EOS, quotes,
duplicates/empty rivals, contextual tokenization, gradient equivalence and
query isolation; historical scorers and hashed functions remain untouched.

GPU work only on `dl-clear-retain-h200x4-20260914`. Prepare on CPU while
pending. Final registration requires a fresh platform RUNNING status and
actual four-GPU host binding; never manufacture that receipt or substitute an
instance. Do not stop unrelated jobs or change quotas.

Output `runs/dreamlite-prefeval-rgb-20260917/scope-contrast-v1-run`.
Launcher phase `scope-contrast` performs train/evaluate/verify/report and
writes `pipeline-terminal.txt`. Preserve both endpoints and all raw records.
Return EXECUTED with results, or completed preparation plus concrete quota
block. Neither prepared code nor a failed experiment completes the Writer goal.
