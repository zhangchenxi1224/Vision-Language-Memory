# Plan10 R2: context/proposal augmentation completed; adoption failed

All training and evaluation ran on the user's latest designated instance,
`dl-clear-retain-h200x4-20260914`. The pipeline exited 0 with 80 endpoints,
5,120 latent optimizer updates, 5,320 generations, 2,256 ranking decisions,
9,024 candidate forwards and 528 recovery-CE forwards. Reader and VAE stayed
frozen. Writer updates remain **zero**, and this is not a usable shared Writer.

R and D start from the same Plan09 V endpoints and receive 64 updates per state,
fresh Adam at .01 and 38,400 gradient Reader forwards each. R uses the existing
application bank; D alternates that bank with counterfactual proposals and two
new contexts. The original evaluation and adoption thresholds remain fixed.

| Metric | Parent V | Control R | Augmented D |
|---|---:|---:|---:|
| Original MCQ | 47/84 | 48/84 | 46/84 |
| Semantic-group macro MCQ | 0.48095 | 0.48988 | 0.47202 |
| Reserved application ranking | 168/168 | 168/168 | 168/168 |
| New-context XML | — | 330/336 | 332/336 |
| New-context ranking | — | 290/336 | 304/336 |
| Same-PNG joint states | 26/40 | 27/40 | 26/40 |
| Joint states also passing MCQ | 8/40 | 8/40 | 8/40 |
| K1 complete recovery | 20/20 | 20/20 | 20/20 |
| K2 complete recovery | 2/4 | 1/4 | 2/4 |
| K3 complete recovery | 1/4 | 2/4 | 1/4 |
| K4 complete recovery | 3/12 | 4/12 | 3/12 |
| Historical overwrite pairs | 8/8 | 8/8 | 8/8 |
| New counterfactual XML pairs | — | 164/168 | 166/168 |
| New counterfactual ranking pairs | — | 130/168 | 143/168 |

D's semantic-group macro gain over R is **−0.01786**, below the registered
+0.10 target. Both fail the absolute recovery and original-MCQ gates. The
new-context ranking gain is limited to that panel; it does not transfer to the
original MCQs or improve multi-slot recovery. No endpoint was selected or
discarded based on these results.

## Scientific scope and construction deviation

The authoring implementation takes `attributes` directly from the preceding
clause-option audit and changes context text and named-proposal permutations.
This does **not** implement the substantive attribute expansion requested in
Plan10. Accordingly this run supports a conclusion about context/proposal
counterfactual augmentation only. Neither the new scenarios nor the repeatedly
observed original panel establish fresh held-out semantic generalization.

The new overwrite pairs share situation and proposal arrays, but their
`base_rotation` is hashed from a value-specific case ID. Therefore the generated
option order is not guaranteed identical before/after. This is a construction
deviation, not a reason to modify completed cases or discount a failure. The
unchanged historical overwrite panel remains separately reportable.

## Provenance and evidence

Runtime Git commit: `73bbcecc7c071fbdffb16022cd96c74a9d2c671f`.
Local implementation commit: `8951783`; the runtime has a separately recorded
Git lineage. Registration digest:
`865a46ee2d14b221ac912fa3805cb2747f4b0fcda3f4975f3bf00da7f777cc50`.
The first attempt failed before an optimizer update with a missing `targets`
payload. Its original registration and failed output remain preserved remotely.
R2 corrected loading and removed duplicate evaluation calls before launching.

Raw output is under `attribute-generalization-v1-r2-run/`. The original
`final-verified.json` is a partial outcome report: its `teacher_candidate`
already fails, but it does not represent every adoption gate. The additional
`same-png-gates.json` reconstructs the complete recovery, original MCQ, reserved
ranking and historical overwrite gates from reopened-PNG-bound raw records.
This addendum corrects the interpretation without rewriting historical scores.

Local reconstruction of both the core report and full outcome gates passed.
All three arms retain all eight historical overwrite pairs and at least one
correct pair for each contrast. D still fails the full adoption contract.
The new counterfactual counts require both variants correct, rather than
counting their individual answers separately.

Training processed 8,881,248 tokens in R and 9,365,344 in D. Summed per-state
training time was 9,909.16 and 9,918.62 seconds respectively. Evaluation used
3,510,222 input tokens and 4,824.83 summed seconds. These sums are not job wall
time. Full evidence has 250,582,512 bytes with SHA-256
`5b4950ae6501e875e464691d18e394268fdbce47eb7d228b624c382ef8b8f838`;
the three-part archive receipt is `attribute-generalization-r2-archive.json`.
Unpacked raw records and endpoint artifacts remain locally available.

The control-reuse addendum independently reconstructed all 5,320 expected
generation keys, 2,256 candidate mappings, 64-step query exposures and optimizer
states. It recalculated ranking flags from raw scores. Selective-clear complete
states are V 1/4, R 1/4 and D 0/4; complete offline teacher chains are V 0/4,
R 1/4 and D 0/4. No arm has an MCQ-inclusive complete chain. These offline
conjunctions do not represent recurrent Writer execution. R2's scalar training
logs cannot recover unrecorded per-component recovery CE or gradient norms;
that limitation is retained in `attribute-generalization-r2-reuse-addendum.json`.
