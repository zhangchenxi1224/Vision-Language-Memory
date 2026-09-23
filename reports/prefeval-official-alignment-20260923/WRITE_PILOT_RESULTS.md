# Fixed write pilot: full results and next diagnostic

2026-09-24. Both original pilot completion markers are present. All 1,360
evaluation condition files are complete: 128 teachers, 308 references, and
462 students per arm. Each condition has five MCQ and five natural-answer rows.
The 6,800 natural answers still await the official judge; no generation accuracy
or A/B method winner is claimed.

## Original-question T1 MCQ

| Input | 64 training preferences | 90 development preferences |
| --- | ---: | ---: |
| A teacher PNG | 44/64 (68.75%) | No dev teacher optimized |
| B teacher PNG | 64/64 (100%) | No dev teacher optimized |
| A shared Writer PNG, both seeds | 57/128 (44.53%) | 69/180 (38.33%) |
| B shared Writer PNG, both seeds | 57/128 (44.53%) | 71/180 (39.44%) |
| Blank image | 30/64 (46.88%) | 36/90 (40.00%) |
| Full text | 59/64 (92.19%) | 79/90 (87.78%) |

Both seeds refer to the same preferences and are correlated, not 128/180
independent examples. All T1 MCQ outputs parsed successfully. The complete
five-form matrix is in `write-evaluation-summary.json`; O1/O2 remain report-only.

For a single paired observation per preference, seed0 gives:

| Arm / split | Correct PNG | Blank | Mismatched PNG | Correct-only / mismatch-only successes |
| --- | ---: | ---: | ---: | ---: |
| A / train | 29/64 | 30/64 | 29/64 | 0 / 0 |
| B / train | 30/64 | 30/64 | 29/64 | 1 / 0 |
| A / dev | 35/90 | 36/90 | 35/90 | 0 / 0 |
| B / dev | 36/90 | 36/90 | 34/90 | 2 / 0 |

The shared Writer has not demonstrated useful memory under this protocol.
Its near-baseline accuracy and negligible matched-image advantage do not support
the claim that it transfers the readable teacher information into generated RGB.
B's perfect teacher MCQ fit establishes a feasible optimized single-state image
for those training questions; it does not establish a successful shared Writer.

## First-principles interpretation and fixed next step

The information path is disclosure -> shared Writer -> saved/reopened PNG ->
frozen Reader. Full text and B teacher PNGs show that the Reader can solve many
of these questions when supplied usable information. The missing evidence is
task-relevant information surviving shared generation. Lower velocity MSE on
target-noised training inputs does not guarantee that a trajectory starting from
pure Gaussian noise reaches a Reader-equivalent image.

There is one specific input difference to isolate before another training run:
the benchmark uses Reader-generated acknowledgments, whereas FM trained with
released SFT acknowledgments. All 64 disclosures are identical; all 64 full
exchanges differ. Consequently the table's train-content students also test
input generalization and cannot alone prove failure on exact training conditions.

Decision based only on completed T1 results: run the already prepared paired
`training-initial` diagnostic on all 64 training states per arm. Reuse the fixed
2048-update checkpoint, original SFT exchange, gray source, seed0 noise namespace,
native 28-step generation, and unchanged Reader evaluation. No teacher/FM update.
Keep its PNGs and scores separate from the benchmark and preserve the benchmark
acknowledgments. See `FM_DIAGNOSTIC_NOTES.md` for the pre-existing hypothesis.

If exact inputs work, investigate input generalization. If they also fail,
investigate free-running FM reconstruction and Reader sensitivity to reconstruction
errors. Do not proceed to retain/K2/K4 training until this single-write bottleneck
is located. Do not use OOD scores to select the correction or change scoring.

## Raw evidence

`write-complete-evidence.jsonl.gz` contains 1,398 raw JSON records including all
1,360 evaluation conditions, completion markers, dispatch/driver records and
preserved failures. Size 7,004,601 bytes; SHA256
`5956ab0ac0c7b8800fe4d83ef6180b41f5958096b1eaedd1f8ff3360042747d9`.
The same release already holds both complete Writer weights, all 616 benchmark
student PNGs, and both complete teacher banks:
[experiment assets](https://github.com/zhangchenxi1224/Vision-Language-Memory/releases/tag/prefeval-official-ab-teachers-20260924).
