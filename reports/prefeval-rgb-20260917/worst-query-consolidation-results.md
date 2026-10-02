# Plan12: mean consolidation helps slightly; worst-query mix regresses

Training and the complete fixed evaluation finished on
`dl-clear-retain-h200x4-20260914`. Train, evaluation and pipeline exit codes are
all zero. There were 2,560 updates, 5,320 generations, 2,256 rankings and 2,288
endpoint recovery CE calls. Each arm used 41,728 gradient forwards and exactly
8,671,260 input tokens. Both arms started from the same 40 frozen E endpoints.

| Outcome | Previous E | Mean M | Mean/worst W |
|---|---:|---:|---:|
| Original MCQ | 54/84 | 55/84 | 52/84 |
| Semantic-group macro MCQ | .57738 | .62798 | .54940 |
| K1 complete recovery | 19/20 | 20/20 | 19/20 |
| K2 complete recovery | 2/4 | 2/4 | 1/4 |
| K3 complete recovery | 1/4 | 1/4 | 1/4 |
| K4 complete recovery | 2/12 | 2/12 | 1/12 |
| Same-PNG joint recovery and reserved ranking | 24/40 | 25/40 | 22/40 |
| Joint states also passing MCQ | 11/40 | 13/40 | 11/40 |
| Observed reserved ranking | 168/168 | 168/168 | 168/168 |
| Complete selective-clear states | 1/4 | 1/4 | 0/4 |
| Complete offline teacher chains | 0/4 | 0/4 | 0/4 |
| Original training recovery forms | 255/264 | 260/264 | 253/264 |
| Additional training recovery forms | 661/704 | 681/704 | 662/704 |
| New compositional XML generation | 290/336 | 298/336 | 281/336 |

Neither arm passes the unchanged absolute recovery and original-MCQ gates.
K4 requires at least 9/12 and original MCQ requires at least 68/84. Historical
overwrite and observed reserved ranking gates pass. The earlier E-minus-R
macro gate remains a separate failed historical result.

The matched comparison does not support adopting W: it has three fewer joint
states and three fewer original MCQs than M. M improves the training recovery
bank and a few outcomes, but does not improve K4 completeness or offline chains.
These results concern already observed development examples and independently
optimized latent teachers. Writer updates remain zero; usable shared Writer
and true recurrent RGB transition training remain unfinished.

Source: `worst-query-consolidation-final-verified.json`, reconstructed by the
remote verifier from the full records. Full archive has 260,549,440 bytes and
SHA-256 `1503013c88d607f329f5a2283a409db64e9be8c35595ce39d9363df09c1d2880`.
The full archive is downloaded, its SHA matches the remote archive, and local
reconstruction exactly matches the remote result. Paired diagnostics and all
recovery failures are saved beside this report; no inference was repeated.

Read-only inspection of the two excluded recovery forms finds 34/176 failures
for M and 46/176 for W (142/176 and 130/176 correct). One M failure was truncated;
none of W's failures were. Several are substantive scope/value confusions:
`shop_motors` returns the stored home-decoration preference, and hotel/restaurant
queries exchange gambling and peanut-allergy preferences. W also has a case
whose answer preserves the car ground-clearance preference with changed wording.
Do not describe every strict failure as lost information or every failure as
mere wording. All failed generations will remain available for review. The next
scientific issue is reliable scope-to-value binding across read formulations,
with unchanged historical scoring and no outcome-based exclusions.

Across the complete eleven-form training bank, M recovers 941/968 queries
(97.21%) and W 915/968 (94.52%). On the two excluded formulations the corresponding
scores are 142/176 (80.68%) and 130/176 (73.86%). These are the same addressed
slots and fixed endpoint PNGs. High training-form recovery therefore does not
establish formulation-independent retrieval. The candidate-ranking ceiling also
does not establish reliable free generation or recursive Writer updates. This
is the central scientific distinction to carry into the next bounded experiment.

Paired original MCQs: M rescues6 and regresses5 relative to E; W rescues4 and
regresses6. W versus M rescues2 and regresses5. Thus the M improvement is small
and not uniform. For the excluded forms, literal answers from another active
scope account for14 of M's34 failures and22 of W's46; these are lower bounds
on semantic scope confusion because paraphrased wrong-slot answers are not
counted. E has17 such literal matches among38 failures.

New compositional counterfactual pairs (both variants correct): M XML137/168,
ranking106/168; W127/168 and93/168; E132/168 and98/168. New identical-question
memory overwrite: M XML9/16 and ranking10/16; W10/16 and7/16; E10/16 and8/16.
These stay separate from historical overwrite gates.

W reduces the worst training-bank total recovery CE (.1363 versus M .1965),
but increases mean answer CE (.02387 versus .01563) and the worst excluded-form
CE (5.4585 versus4.1667). Mean EOS CE stays below.001 in both arms and both
panels. A lower worst training-query loss therefore did not imply more robust
retrieval here. This describes the measured tradeoff, not proof of its mechanism.
