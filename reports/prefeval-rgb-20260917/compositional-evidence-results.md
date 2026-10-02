# Plan11: better original-task application, weaker joint retention

The fixed E treatment completed on `dl-clear-retain-h200x4-20260914` with
2,560 latent updates, 2,996 generations and 1,464 ranking decisions. The pipeline
exited 0. This treatment started from all 40 Plan09-V endpoints, using fresh
Adam and the same 64-step schedule as the archived R2-R control. The content
change was component-level factual evidence for the application task.

| Outcome | Parent V | Archived control R | Compositional E |
|---|---:|---:|---:|
| Original MCQ | 47/84 | 48/84 | **54/84** |
| Semantic-group macro MCQ | .48095 | .48988 | **.57738** |
| Observed transfer ranking | 168/168 | 168/168 | 168/168 |
| K1 complete recovery | 20/20 | 20/20 | 19/20 |
| K2 complete recovery | 2/4 | 1/4 | 2/4 |
| K3 complete recovery | 1/4 | 2/4 | 1/4 |
| K4 complete recovery | 3/12 | 4/12 | **2/12** |
| Same-PNG joint states | 26/40 | 27/40 | 24/40 |
| Joint states also passing MCQ | 8/40 | 8/40 | **11/40** |
| Complete selective-clear states | 1/4 | 1/4 | 1/4 |
| Complete offline teacher chains | 0/4 | 1/4 | 0/4 |

E's original-MCQ macro gain is **+.0875**, below the preregistered +.10 target.
It also fails the absolute original-MCQ and complete-recovery gates. All three
arms pass observed transfer ranking and the historical overwrite gates. Neither
the 54/84 score nor the increase in MCQ-inclusive joint states qualifies E as
a teacher bank or as a shared Writer. Writer updates remain **zero**.

| New compositional case outcomes | R | E |
|---|---:|---:|
| XML generation | 184/336 | 290/336 |
| Full-action candidate ranking | 163/336 | 253/336 |
| Both counterfactual variants correct, XML | 58/168 | 132/168 |
| Both variants correct, ranking | 52/168 | 98/168 |

These cases were training-fit cases for E and new-content comparisons for R.
Their gains establish learning on this bank. The original MCQ change provides
descriptive evidence of application transfer on the already observed development
panel; it is not a fresh held-out result or a replicated significance claim.
The two context types reuse the same component schema (single commitment versus
recurring package); this is a deliberately narrow compositional-content test.

The main next question is whether the application gain can be retained while
recovering multi-slot completeness. Simply extending the same treatment is not
justified by this run. No outcome-based exclusions or threshold changes were made.

Runtime commit: `cf665416bab12367c6236f2ea0a179db8a18911b`; local implementation:
`289f38e`. The registration binds execution-critical file bytes across the separate
Git lineages. Registration digest:
`e050b7bd9d807591dbb54987d8f58020af68eb9514e4dd7cdc040ecd6fa1b0ac`.
The remote verifier reconstructs exact record coverage, candidate mappings,
ranking flags, optimizer steps and per-step training exposures. Its compact
result is `compositional-evidence-final-verified.json`.

Full raw archive SHA-256:
`de02e955cf3a4053be5f5a22357df51dbed6ceb16f3b7ed29619228d25359477`.
The archive is fully downloaded and unpacked. Local exact reconstruction matches
the remote result. As in R2, the E training trace does not record
per-component recovery CE or gradient norms; these cannot be recovered from its
scalar training losses. The stored final PNGs, latent/optimizer checkpoints,
full query/candidate records and endpoint recovery CE remain available.

Paired E-versus-R MCQs rescue 9 cases and regress 3, net +6. Original training-form
recovery rescues 2 and regresses 7 (E 255/264, R 260/264). New recovery-form coverage
rescues 4 and regresses 19 (661/704 versus 676/704). Strict full-action generation
falls to 134/336 versus 154/336 on training cases and 55/168 versus 63/168 on
observed reserved cases. Old derived XML stays near ceiling, 335/336 versus
336/336. These generation results remain distinct from candidate ranking.

On the new byte-identical-question memory overwrite contrasts, E succeeds on
10/16 XML pairs and 8/16 ranking pairs, versus R 1/16 and 2/16. These new-case
contrasts are reported separately from the unchanged historical overwrite gates.
Full per-group comparisons and contrast counts are in
`compositional-evidence-paired-outcomes.json`.

E used 38,400 gradient Reader forwards and 10,106,872 input tokens, with
10,019.87 summed per-state training seconds. R used the same update/forward
budget but 8,881,248 tokens; this is not a token-compute-matched comparison.
Evaluation used 2,494,287 input tokens, 5,856 candidate forwards, 264 recovery CE
forwards and 2,692.50 summed seconds. Summed times are not job wall time.

The seven training-form recovery regressions are saved in
`compositional-evidence-recovery-regressions.json`. None is truncated. Three
return a different stored slot's value (vehicle versus home, or exercise versus
nutrition); other cases introduce unrelated content or paraphrase away the
complete constraint. Thus the retention decline cannot be explained solely by
strict output-format scoring. This is an observed failure pattern, not proof
of a particular gradient-conflict mechanism.
