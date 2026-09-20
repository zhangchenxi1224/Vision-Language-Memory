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
Local archive reconstruction and paired diagnostic review are the next steps.
