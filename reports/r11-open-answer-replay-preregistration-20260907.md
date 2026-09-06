# R11 old VAE latent: open-answer replay preregistration

Date: 2026-09-07 (Asia/Shanghai). This protocol is frozen before any new GPU replay output is inspected.

## Scope and question

Replay the eight existing old R11 step-256 endpoints with the original frozen Qwen Reader and VAE, using a question and decoded memory image with no candidate answers. This is a diagnostic of whether successful candidate ranking transfers to free answer generation and whether outputs depend on the matched image. It is not a new optimization run, a same-question multistart experiment, or an estimate of a latent convergence distribution. Existing R11 results and raw archives are retained unchanged.

The config is `configs/experiments/r11_open_answer_replay.json`, schema `vision_memory.r11-open-answer-replay-config.v1`. Its byte SHA-256 at preregistration is `d004c6f41cc6264e425eaecd2b8559722ea7640182376cc93d27a759971e3c12`. All eight original questions, candidate lists, gold positions, segment IDs and endpoints come from the archived old R11 manifests. Endpoint file bytes were rehashed against the prior offline audit; the canonical tensor SHA-256 values use `vision_memory.canonical_tensor.v1` and are carried from that verified audit. The runner must independently verify these values before inference. No later R11 optimized endpoint is included.

## Fixed design

There are exactly 48 open-answer generations: 8 targets × 3 image conditions × 2 prompts. For each target, keep the question fixed across the three image conditions:

- `matched`: that target's original step-256 latent decoded through the original VAE.
- `blank`: the shared original blank-source latent decoded through the same VAE, with canonical tensor hash `719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`. This controls for question priors and the shared starting background. An identical, hash-verified copy of this initial tensor is permissible; newly optimized tensors are not substitutes.
- `donor`: another old R11 target's endpoint. Starting at the next index and moving cyclically, select the first target whose gold text differs from the current target's. The assignment is fixed below, with no outcome-dependent reselection. This is a mismatched-memory intervention, not a clean counterfactual gold label, because entities and some attributes also differ.

| Target index | Segment ID | Gold, scorer only | Donor index |
|---|---|---|---|
| 00 | `r5-f1-8015bf53a4067aaa7e882288` | linen | 01 |
| 01 | `r5-f1-392d41fd097d069c42218e0a` | ambient | 02 |
| 02 | `r5-f1-1aee01c0f3e7684c05c9122c` | orange | 03 |
| 03 | `r5-f1-807090710dd4c077a97348ba` | yellow | 04 |
| 04 | `r5-f1-02c8aa9dc3523351c4d5f9c7` | blue | 05 |
| 05 | `r5-f1-11dddf29efebc033a995ab92` | juice | 06 |
| 06 | `r5-f1-de4899e05b0aa7f3d8373171` | pasta | 07 |
| 07 | `r5-f1-850189e424efde62468b2ef9` | green | 00 |

All eight gold answers happen to be different, so donor assignment is a one-step cycle. The config retains the original `no active preference` candidate text exactly; it is not a gold answer for these eight targets. It must never be silently recoded as an affirmative preference, a guessed value, or a synonym for uncertainty.

Each target has an `original_open` prompt that removes only `Choose exactly one option.` from the source query and adds the common short-answer instruction. The `paraphrase_open` prompt retains the source template prefix, exact entity identifier, requested attribute, and temporal relation (later check, before a subsequent update, or after applying the update). Neither prompt provides candidate answers, a gold answer, training-event text, or an answer prefix. Gold and any future prespecified aliases are confined to `scorer_metadata`; all aliases are empty in this run. `original` is archival and MCQ-anchor metadata and must not be used to construct the open-answer input beyond the frozen `inputs` strings.

Generation is greedy (`do_sample=false`), with at most 32 newly generated tokens and the original model's EOS behavior. Do not override EOS, ban tokens, retry with alternate prompts, or select a favorable sample. Freeze and record the model/processor snapshot, model generation config, chat template, attention implementation, dtype, VAE decoding and reader image preprocessing. Use the same numerical image path for matched, blank and donor; any alternate PNG-only replay must be identified as a separate diagnostic rather than silently replacing this path.

## Replay anchor

Before interpreting the open-answer comparison, replay the matched endpoints on all four original cyclic MCQ views: 8 targets × 4 views = 32 candidate-ranking evaluations, retaining the original candidate-mean-token-NLL and choice-CE path. Save all four scores, gold index, chosen index, margin, and CE per view. This is an anchor for model/decoder/preprocessing fidelity; it is not an additional open-answer generation condition. Record disagreement with the archived results rather than selecting endpoints, changing settings, or redefining success. If the old anchor cannot be reproduced, label the open-answer comparison technically unresolved until the discrepancy is explained.

## Scoring frozen before generation

Primary outcome is normalized exact match to the target's gold text. Normalize both prediction and gold by applying Unicode `casefold`, collapsing every whitespace run to one ASCII space and trimming, then repeatedly removing trailing whitespace and trailing Unicode punctuation (a character whose Unicode category begins with `P`). Stop when the last character is neither. No leading/internal punctuation deletion, prefix stripping, article removal, keyword extraction, token-set comparison, stemming, or semantic/synonym substitution is allowed. All aliases are empty, and aliases cannot be added after seeing generations.

For example, ` GREEN. ` and `green` match; `The answer is green.` and `green` do not match under the primary metric. `no active preference` remains that complete phrase. Extra explanation, empty output, truncation at 32 tokens and apparent semantic alternatives should be separately described from the preserved raw responses, without altering primary exact-match scores. Any later semantic adjudication must be labeled exploratory and keep the primary metric intact.

Report a 3 × 2 table of correct / 8, paired matched-minus-blank and matched-minus-donor differences for each prompt, and each target's six raw outputs with exact-match flags. Also record cross-prompt output agreement and whether a donor output matches the donor gold as exploratory diagnostics. The 48 outputs are repeated conditions on eight targets, not 48 independent samples. Do not claim population significance or latent convergence from this fixed small diagnostic set.

## Required artifacts and integrity

Save the frozen input config and its SHA-256; exact model-facing prompt/chat serialization and input token IDs; decoded output text with and without special-token removal; generated token IDs; generated length and stop/truncation reason; target and donor provenance; endpoint/initial canonical tensor hashes and file hashes; numerical image/preprocessing metadata; all MCQ anchor rows; per-generation errors; complete environment/runtime information; and summary metrics. Scorer-only fields must be excluded from serialized model inputs. Save outputs incrementally so an interrupted run is auditable; resume only missing deterministic rows from the same frozen run settings, and retain any failed attempt metadata.

A matched advantage over both controls with consistent paraphrase behavior supports transfer to open answers for these eight fixed cases. Strong matched MCQ accuracy with weak open-answer accuracy reveals a ranking-to-generation gap under this protocol; it does not by itself prove there is no information in the latent. Comparable blank performance raises a prior/template explanation. Donor sensitivity shows image intervention matters but does not establish human-readable or uniquely encoded information. No result here proves that distinct latent starts converge to the same endpoint.

## Frozen prompts

### Target 00

Original open:

```text
R3 Train Standard Templates 15: At this later check, what is the current material preference for the willow room train 000431?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 15: At this later check, which material does the willow room train 000431 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 01

Original open:

```text
R3 Train Standard Templates 03: At this later check, what is the current music preference for the indigo desk train 001123?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 03: At this later check, which music does the indigo desk train 001123 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 02

Original open:

```text
R3 Train Standard Templates 29: After applying this update first, what is the current color preference for the amber desk train 000797?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 29: Once this update has been applied first, which color does the amber desk train 000797 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 03

Original open:

```text
R3 Train Standard Templates 24: Before any later update, what is the current color preference for the willow chair train 000408?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 24: Before any subsequent update, which color does the willow chair train 000408 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 04

Original open:

```text
R3 Train Standard Templates 27: Before any later update, what is the current color preference for the linen lamp train 000987?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 27: Before any subsequent update, which color does the linen lamp train 000987 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 05

Original open:

```text
R3 Train Standard Templates 30: Before any later update, what is the current drink preference for the copper desk train 000478?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 30: Before any subsequent update, which drink does the copper desk train 000478 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 06

Original open:

```text
R3 Train Standard Templates 28: After applying this update first, what is the current meal preference for the linen backpack train 000540?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 28: Once this update has been applied first, which meal does the linen backpack train 000540 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```

### Target 07

Original open:

```text
R3 Train Standard Templates 03: At this later check, what is the current color preference for the silver backpack train 000003?
Use the memory image to answer. Answer with a short phrase only.
```

Paraphrase open:

```text
R3 Train Standard Templates 03: At this later check, which color does the silver backpack train 000003 currently prefer?
Use the memory image to answer. Answer with a short phrase only.
```
