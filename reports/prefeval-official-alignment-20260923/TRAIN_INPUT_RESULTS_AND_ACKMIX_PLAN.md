# Exact-input result and acknowledgment correction

Frozen before the corrective training launch, 2026-09-24.

## Completed diagnostic

All 64 training states per arm were generated with the original SFT initial
exchange, frozen 2048-update Writer, gray source, and the same seed0 Gaussian
noise as the benchmark. Native 28-step inference, saved/reopened PNGs, Reader,
questions, options and scoring were unchanged. All four diagnostic shards ended
successfully; no teacher or Writer optimizer updates occurred.

| T1 MCQ input | A | B |
| --- | ---: | ---: |
| Teacher PNG | 44/64 | 64/64 |
| Student, exact SFT exchange | 38/64 | 62/64 |
| Student, benchmark exchange, same seed0 | 29/64 | 30/64 |
| Blank | 30/64 | 30/64 |
| Full benchmark text | 59/64 | 59/64 |

Against benchmark-input students, A has 10 newly correct / 1 newly wrong;
B has 32 newly correct / 0 newly wrong. Against blanks the pairs are A9/1,
B32/0. All disclosures are identical; the changed component is the acknowledgment.
This is strong evidence that acknowledgment sensitivity is a major failure
mechanism, particularly for B. FM can generate readable single-state training
memories from pure noise; it is not universally failing to reproduce the targets.
A still has an additional teacher/reconstruction gap. This evidence does not
establish new-preference transfer, natural-answer quality, recurrence or K4.

Full five-form scores and pairs are in `training-input-evaluation-summary.json`.
O1/O2 are report-only and did not choose this correction. The 640 diagnostic
natural answers remain unjudged, for 7,440 pending across both completed stages.
The 273-record diagnostic archive is 800,238 bytes, SHA256
`2a7f6031b4b1e184d2aaea0982c998f0418d51b3d4cbb597004ace17c485c8eb`.

## First-principles correction

A memory update should preserve the same information when an acknowledgment
changes but the user's preference does not. With one condition string per target,
64 arbitrary image targets allow an overly specific text-to-image mapping. The
completed input intervention exposes that fragility. Train the same target under
two already available, semantically compatible exchanges to constrain it.

New stage: `write-ackmix`, independent A/B continuations. Start each arm from its
own frozen `writers/{arm}/write/checkpoint-final.pt`; retain all64 unchanged
teacher targets, including imperfect ones. Use a fresh AdamW with the original
lr5e-5, batch4, and **2048 additional updates per arm**. Keep official FM,
sigma/noise distributions, native Base conditions, frozen VAE/text encoder,
full U-Net, 28 inference steps, CFG1 and pure Gaussian inference unchanged.

Each training preference has exactly two initial exchanges: its released SFT
acknowledgment and its cached frozen-Reader acknowledgment. Alternate versions
by full64-state pass, with paired shuffled draws and noise across A/B. Thus each
state gets128 draws,64 per version. The user disclosure is byte-identical.
Only the first user/assistant exchange enters the Writer, with the gray source.
No query, option, future event, target answer, dev acknowledgment or state label
enters training. Both versions point to the same existing target latent.

This is a disclosed training augmentation, not an exact reproduction of the
official SFT input distribution. The official benchmark still uses its tested
Reader's acknowledgment; it is not replaced by a teacher acknowledgment.
The64 cached train-side benchmark acknowledgments now become training conditions;
subsequent scores on them must be labeled training fit, not unseen-input transfer.
The90 dev preferences remain outside optimization. No official180 evaluation
examples are used. Continuing training also adds compute, so this iteration alone
does not isolate augmentation's causal benefit from extra optimizer updates.

## Fixed readout and decision

After both2048 endpoints, generate the original fixed154-state benchmark, two
seeds per arm, and all462 student conditions per arm with the unchanged mismatch
donors. Same five question forms and two tasks. Reuse the completed write-stage
blank/full-text reference answers because their prompts and Reader are unchanged.
Store new weights/PNGs/readouts under `write-ackmix`; preserve all original files.

Read complete T1 paired results first. Better64-train fit alone means the added
conditions were learned. Evidence for transfer requires gains on90 unseen dev
preferences over blank and mismatched images, plus eventual official natural
answer judging. Do not tune from OOD or declare a winner from teacher/MCQ loss.
No further updates, retain or K2/K4 are automatically queued after this endpoint;
analyze the resulting bottleneck before the next correction.

Launch once with `launch_prefeval_official_ab.py ackmix --output RUN` on the
designated notebook. The two drivers use GPU0/1 for A and GPU2/3 for B: FM uses
GPU0/2; each arm then uses its two GPUs for rollout and Reader shards. Receipt
`dispatch-ackmix.json`, driver records `pipeline-ackmix/{A,B}`. Archive original
diagnostic PNGs and raw answers, then the new results, on the existing release.
