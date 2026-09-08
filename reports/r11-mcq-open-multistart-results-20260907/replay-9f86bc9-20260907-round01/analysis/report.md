# Old R11 open-answer replay

The replay is technically complete: 48 open generations and 32/32 correct original MCQ anchor views.
No semantic credit has been added. Raw answers await manual review; the primary metric remains the preregistered normalized exact match.

## Exact-match accuracy

| Image condition | Prompt | Correct / 8 | Accuracy | Truncated | Gold-containing extra text | Empty |
|---|---|---:|---:|---:|---:|---:|
| matched | original_open | 3/8 | 37.5% | 0 | 0 | 0 |
| matched | paraphrase_open | 4/8 | 50.0% | 0 | 0 | 0 |
| blank | original_open | 0/8 | 0.0% | 0 | 0 | 0 |
| blank | paraphrase_open | 0/8 | 0.0% | 0 | 0 | 0 |
| donor | original_open | 0/8 | 0.0% | 0 | 0 | 0 |
| donor | paraphrase_open | 1/8 | 12.5% | 0 | 0 | 0 |

Gold-containing extra text means the expected string appears with additional content, including possible negation. A zero count does not mean the responses contain no unrelated sentences or extra words. This is not semantic correctness, and truncation is an independent flag.

## Paired matched-versus-control outcomes

| Prompt | Control | Both correct | Matched only | Control only | Neither | Matched − control |
|---|---|---:|---:|---:|---:|---:|
| original_open | blank | 0 | 3 | 0 | 5 | +37.5% |
| original_open | donor | 0 | 3 | 0 | 5 | +37.5% |
| paraphrase_open | blank | 0 | 4 | 0 | 4 | +50.0% |
| paraphrase_open | donor | 0 | 4 | 1 | 3 | +37.5% |

## Cross-prompt agreement

| Image condition | Raw strings identical / 8 | Normalized strings identical / 8 |
|---|---:|---:|
| matched | 5/8 | 5/8 |
| blank | 5/8 | 6/8 |
| donor | 3/8 | 5/8 |

Agreement can be agreement on an incorrect answer.

## Donor-string diagnostic

This only checks whether the answer under the donor image equals the donor question's gold string. Entities and attributes differ; a hit is not a semantic counterfactual score.

| Prompt | Donor-gold string matches / 8 | Query target indices |
|---|---:|---|
| original_open | 2/8 | [2, 3] |
| paraphrase_open | 2/8 | [2, 3] |

## Preserved raw answers

Cells show the entire original string in JSON notation: `\n` and spaces preserve whitespace; no text is shortened. `EM` is exact match, `T` is truncation, and `E` is extra wording containing the expected string. The CSV/JSON retain the raw strings directly.

| Target | Gold | Prompt | Matched answer | Blank answer | Donor answer |
|---:|---|---|---|---|---|
| 00 | linen | original_open | <code>"wood"</code><br>EM=0, T=0, E=0 | <code>"None"</code><br>EM=0, T=0, E=0 | <code>"wood"</code><br>EM=0, T=0, E=0 |
| 00 | linen | paraphrase_open | <code>"Linen"</code><br>EM=1, T=0, E=0 | <code>"None"</code><br>EM=0, T=0, E=0 | <code>"Willow Room Train 000431 currently prefers the material: **Willow**."</code><br>EM=0, T=0, E=0 |
| 01 | ambient | original_open | <code>"indigo"</code><br>EM=0, T=0, E=0 | <code>"No music preference recorded."</code><br>EM=0, T=0, E=0 | <code>"Yellow, not indigo."</code><br>EM=0, T=0, E=0 |
| 01 | ambient | paraphrase_open | <code>"R3 Train Standard Templates 03"</code><br>EM=0, T=0, E=0 | <code>"None"</code><br>EM=0, T=0, E=0 | <code>"Yellow, not indigo."</code><br>EM=0, T=0, E=0 |
| 02 | orange | original_open | <code>"Yellow"</code><br>EM=0, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"yellow"</code><br>EM=0, T=0, E=0 |
| 02 | orange | paraphrase_open | <code>"Yellow"</code><br>EM=0, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"yellow"</code><br>EM=0, T=0, E=0 |
| 03 | yellow | original_open | <code>"yellow"</code><br>EM=1, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"Blue"</code><br>EM=0, T=0, E=0 |
| 03 | yellow | paraphrase_open | <code>"yellow"</code><br>EM=1, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"blue"</code><br>EM=0, T=0, E=0 |
| 04 | blue | original_open | <code>"White"</code><br>EM=0, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"Yellow"</code><br>EM=0, T=0, E=0 |
| 04 | blue | paraphrase_open | <code>"White"</code><br>EM=0, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"Blue"</code><br>EM=1, T=0, E=0 |
| 05 | juice | original_open | <code>"Juice"</code><br>EM=1, T=0, E=0 | <code>"No drink preference recorded."</code><br>EM=0, T=0, E=0 | <code>"Coca-Cola"</code><br>EM=0, T=0, E=0 |
| 05 | juice | paraphrase_open | <code>"Juice"</code><br>EM=1, T=0, E=0 | <code>"None"</code><br>EM=0, T=0, E=0 | <code>"Coca-Cola"</code><br>EM=0, T=0, E=0 |
| 06 | pasta | original_open | <code>"With one, food sort."</code><br>EM=0, T=0, E=0 | <code>"none"</code><br>EM=0, T=0, E=0 | <code>"Vegetarian"</code><br>EM=0, T=0, E=0 |
| 06 | pasta | paraphrase_open | <code>"$11.00"</code><br>EM=0, T=0, E=0 | <code>"None"</code><br>EM=0, T=0, E=0 | <code>"Green Vegetables"</code><br>EM=0, T=0, E=0 |
| 07 | green | original_open | <code>"Green"</code><br>EM=1, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"blue"</code><br>EM=0, T=0, E=0 |
| 07 | green | paraphrase_open | <code>"Green"</code><br>EM=1, T=0, E=0 | <code>"gray"</code><br>EM=0, T=0, E=0 | <code>"Blue"</code><br>EM=0, T=0, E=0 |

## Technical and interpretation notes

- Eight fixed questions; conditions and prompts are repeated measures, not independent replications.
- Donor-gold matches and extra-text detection are string diagnostics, not semantic correctness.
- Prompt agreement may reflect the same wrong answer; it is not an accuracy metric.
- Truncation and extra text are separate flags and may overlap.
- No endpoint optimization, multistart convergence claim, semantic relabeling, or significance test is performed.
- MCQ gold-minus-best-wrong margin: min 6.58223, mean 8.68705, max 10.5381.
- Frozen source config SHA-256: `5420dafd73b1cabd9b0d77f1e8eda4d94eb9179f4ac157dc0d8993a1d10ef7a5`.
- The analysis independently rechecks row coverage, strict scores, image/latent provenance, EOS metadata, the saved technical gates and all 32 MCQ predictions. It does not reload models or deserialize tensors.
