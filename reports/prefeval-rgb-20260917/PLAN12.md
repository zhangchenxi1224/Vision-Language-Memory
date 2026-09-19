# Plan12: complete-bank worst-query consolidation

Approved in the original C2C conversation, iteration 12. Execute only on
`dl-clear-retain-h200x4-20260914`. Registration digest:
`cb325203bbe45020bd68b68e4a62ef7ec2037989da7f2dd4cc3d126c7e12774a`.

Both M and W start from the same 40 frozen Plan11 E latent endpoints. Fresh
Adam (.01, .9/.999, eps 1e-8, no weight decay), 32 updates per endpoint.
Every update replays all 11 existing training recovery forms per addressed
slot, including clear targets. M uses weighted mean recovery; W uses half
weighted mean and half the unweighted worst query. Exact ties use lexical
(scope, query id). Both retain the same .25 slot-weighted application term,
and the same fixed 32-step old/compositional case and format schedule.

Sequential detached gradient accumulation matches the explicit objective;
one optimizer update follows all queries at the same image. Four focused
tests passed, covering gradients, ties, bank membership and call budgets.
No qualification questions, MCQs or reserved cases enter gradients.

Budget: 2,560 updates, 83,456 gradient forwards, 80 fixed endpoint PNGs;
5,320 generations, 2,256 rankings (9,024 candidate forwards), 2,288 endpoint
recovery CE calls including 352 excluded-form diagnostic calls after freeze.
Endpoint CE retains token NLL, gold margins, argmax correctness and EOS.

Compare M/W/E complete and joint recovery, application, changed/untouched
slots, clear, offline chains, counterfactual pairs and memory overwrite.
Absolute gates remain K1>=18/20, K2>=3/4, K4>=9/12, MCQ>=68/84,
reserved ranking>=135/168, historical overwrite>=6/8 with one per group.
The earlier E-minus-R macro gate remains FAILED; it is not redefined here.
No automatic extension or adoption. Writer updates remain zero; these are
independent latent teachers, not recurrent shared Writer transitions.

Output: `runs/dreamlite-prefeval-rgb-20260917/worst-query-consolidation-v1-run`.
Launcher runs training, evaluation and final verification, then writes
`pipeline-terminal.txt`. Preserve all outcomes and obtain the next C2C review.
