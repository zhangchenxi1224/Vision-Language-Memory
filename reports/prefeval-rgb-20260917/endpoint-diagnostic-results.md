# Fixed-endpoint diagnosis: query transfer and application remain the bottleneck

All 40 existing teacher endpoints completed the registered factorial diagnosis
at `517475f`, with 1,760 recovery generations, 1,760 no-gradient CE forwards,
252 MCQ generations, and zero optimizer updates. All 176 archived PNG-path
held-out token sequences replayed exactly. All endpoint reconstructions matched
the archived PNG pixels. Original visual qualification remains failed.

| Preprocessing / pixels | Training forms | Held-out forms |
|---|---:|---:|
| CPU / decoded float | 256/264 | 116/176 |
| CPU / reopened PNG | 255/264 | 117/176 |
| CUDA / decoded float | 256/264 | 116/176 |
| CUDA / reopened PNG | 255/264 | 117/176 |

CPU/CUDA placement changes **zero generated token sequences** at either pixel
representation. Float-to-PNG changes 3 training and 13 held-out sequences per
device. It causes one training correct-to-wrong flip; held-out has one
correct-to-wrong and two wrong-to-correct flips. Thus quantization has a small
measured effect, and preprocessing placement has no measured generation effect
on this panel. Neither explains the broad query-transfer failure.

| Addressed slots | PNG training forms: complete state | PNG held-out forms: complete state |
|---|---:|---:|
| 1 | 20/20 | 20/20 |
| 2 | 4/4 | 0/4 |
| 3 | 4/4 | 1/4 |
| 4 | 10/12 | 1/12 |
| Total | 38/40 | 22/40 |

Of the original 59 failed held-out PNG reads, 49 concern slots whose three
training forms all generate correctly on the same PNG. Only one original
failure becomes correct with float pixels. These indicators overlap; their
intersections are recorded in `endpoint-verified.json`. Held-out refers to
exclusion from teacher gradients, not fresh research holdouts.

## Applying the stored preference to its original task

All 84 active state-slot occurrences retain their exact current-value source
record, question, option permutation, and official label. Replacement values
use their own source questions; cleared slots receive no stale label. The
complete current state is supplied only to the text-reference condition.

| Condition | K1 | Multiple slots | All occurrences |
|---|---:|---:|---:|
| Archived teacher PNG | 4/20 | 26/64 | 30/84 |
| Blank image | 5/20 | 26/64 | 31/84 |
| Complete text reference | 20/20 | 64/64 | 84/84 |

There is no observed aggregate application improvement over blank in this
panel, even though every K1 teacher passes both held-out recovery forms.
Occurrences repeat semantic groups across states; they are not 84 independent
preferences. Group-level occurrence summaries are retained in the verifier.
This is training-side teacher evidence, not unseen Writer generalization.

## Eight bounded generation-divergence inspections

Eight factorial cells had every gold token argmax-correct under teacher forcing
but a different greedy continuation. The preregistered first-eight selection
therefore inspected all eight, at `d3b2f5f`, without generating replacement
answers or changing decoding. They concern four target/query pairs repeated
across pixel/device conditions. At the first divergence, original gold margins
were 0, 0.125, or 0.625 logits. Prefix-only and cached forward computations
changed the ranking; the cached prefix reproduced the observed wrong token in
all eight cases. Raw top logits, prefixes and identities are in
`first-divergence.json` in the unpacked evidence.

This supports numerical fragility near small token margins in these cells. It
does not establish a universal cache defect, justify changing the locked model
precision, or rescue the application result. No kernel sweep was performed.

## Scientific implication and next review

Optimizing an image to reproduce full statements under a few questions can
learn a narrow question-conditioned response code. The present evidence does
not establish that the image supplies reusable semantic content to other
questions. Shared-Writer distillation from this bank would inherit an
unresolved teacher limitation; scaling it now would not answer the user's
memory hypothesis.

The next controlled training intervention should address semantic/query
transfer using training-only supervision, while preserving these failed
held-out and application results. The next ChatGPT PLAN will specify that
intervention before execution. No third prompt revision, additional teacher
steps, final ID/OOD inference, or shared-Writer optimization occurred here.

The main diagnostic summed to 1804.6855 seconds after model loading across four
GPU lanes. This excludes startup and the small divergence probe, and is not
billed GPU time. 64 focused tests passed. Full raw records, processor tensors,
artifact hashes, replay checks and independent verification are retained.
