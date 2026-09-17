# Plan06: ranking calibration completed; contrast criterion failed

The complete fixed score panel ran at `cd568be04262de9f1324f40551fef4217cc27afe`.
All 2,688 registered decisions are present. Exact-input reuse reduced candidate
forwards from the 10,752 maximum to 9,088, with complete occurrence mappings.
No new generated answers, latent updates, Writer updates or reserved-probe calls
were made. The conditional paired continuation was not run.

Registration digest:
`c559f8f9d598b161f8482e31273cc87fa7893ec738cf1cb053e7f6f53e86525a`.
The objective is the existing temperature-1 listwise loss on mean full-action
token NLL, without an application EOS term. Prompts and all scenarios are
unchanged from failed Plan05. No score normalization or temperature sweep ran.

| Prospective criterion | Result | Status |
|---|---|---|
| Text uniquely correct >=1,210/1,344 | 1,324/1,344 (98.51%) | Pass |
| Text minus blank semantic-group macro >=0.10 | 0.977679 - 0.642857 = 0.334821 | Pass |
| Each overwrite contrast both states correct >=12/16 | 16, 12, 16, **5** | **Fail** |

Blank occurrence accuracy is 881/1,344. There are no tied gold/best-alternative
scores. Rotations and repeated state occurrences are not independent samples;
macro results aggregate within the 28 semantic groups first.

| Overwrite contrast | Text both-correct | Blank both-correct |
|---|---:|---:|
| Avoid water -> avoid choreographed dance | 16/16 | 0/16 |
| Avoid EDM -> avoid superhero comics | 12/16 | 0/16 |
| Avoid wool -> avoid floral patterns | 16/16 | 0/16 |
| Avoid commercialized tourism -> avoid gambling | **5/16** | 0/16 |

Both sides use identical query and candidate order with different gold actions.
The verifier confirms byte-equivalent blank-input score records across each
pair, including pairs executed on different GPUs. The paired failure cannot
be inferred from the high aggregate text accuracy alone.

All 20 text ranking failures concern two values: superhero-comic avoidance
(27/32 correct) and gambling avoidance (17/32 correct). The full query, candidate
order, four scores and margins for every failure are preserved in
`semantic-ranking-text-failures.json`. This is descriptive localization, not
permission to remove or rewrite those cases or retrospectively pass the gate.

## Verification and cost

The independent verifier reconstructs coverage, candidate order, full actual
processor text, gold margins, ties, cached-input bindings, model identities,
token accounting, and all contrast decisions. Local reconstruction matches
the archived remote JSON semantically and after line-ending normalization.
Original LF archive receipts are restored in the unpacked evidence; all Plan05
artifacts compare byte-identical to their original archive. The verifier now
writes LF explicitly for portable receipts. No scoring semantics changed.

49 tests passed across `test_prefeval_semantic_transfer.py`,
`test_qwen_reader_loss.py`, `test_open_eos.py`, `test_open_answer.py`,
`test_prefeval_visual_qualification.py` and `test_base_alignment.py`.
Readable output: `.cache/prefeval-ranking-tests.txt` and
`.cache/prefeval-ranking-local-verification.txt`.

Actual processed input tokens: 2,713,824. Summed post-load shard time:
902.2137 seconds, excluding startup and not equivalent to billed GPU time.
Full archive: `semantic-ranking-calibration-v1.tgz`, 1,073,533 bytes,
SHA-256 `46606b58bcccee289f69c57b9dd12284f46eeb96e0d55d6e646f6b21e4d45ab8`.
Local and remote archive hashes match. Unpacked records are accessible under
`semantic-ranking-v1-run/`.

## Review question

The auxiliary ranking objective has a strong aggregate text signal but a
localized counterfactual failure. This does not establish that latent training
cannot improve original MCQ transfer; no such training has run. Keep the failed
calibration intact. The next plan should decide a bounded, informative training
experiment or one necessary targeted diagnosis, while prioritizing the user's
shared DreamLite RGB-memory main line over repeated broad calibration sweeps.
