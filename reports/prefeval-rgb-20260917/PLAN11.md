# Plan11: compositional evidence, one treatment

C2C `c2c_7e19`, PLAN_ID `prefeval-rgb-compositional-evidence-11`.
Instance: **dl-clear-retain-h200x4-20260914 only**.

E starts from every one of the 40 Plan09-V latents with a fresh Adam optimizer.
The archived Plan10 R2-R arm is the control. It has been reconstructed from the
complete archive, including exact queries/candidates, score-derived ranking,
optimizer state, 64-step exposures and PNG bindings. The new addendum records
the limitations of R2's less detailed training trace; no missing observations
were synthesized. R2-D remains a negative result, not the new initialization.

The only treatment change is the application content bank: old-bank draws
remain unchanged; new-bank draws use two-component factual descriptions authored
from all 28 original preferences and contrast membership, without original MCQs,
old attribute sentences or observed model errors as authoring inputs. Each value
has two situations and two factual variants, 112 assignments. Both mandatory
components must be considered. Factual edits change compatibility; overwrite
contrasts preserve complete question and candidate order. Gold/audit fields
are not included in Reader prompts. These are training-fit cases for E and
new-content comparisons for R, not new semantic holdouts.

Keep the same 64-draw schedule, optimizer, query families, slot weights, .25
application coefficient, Reader/VAE, latent geometry and saved RGB read path.
Exactly 2,560 updates, 16,896 recovery forwards and 21,504 application forwards.
Then freeze all E endpoints and run 2,996 generations, 1,464 ranking decisions
(5,856 candidate forwards) and 264 recovery CE calls. Old control outcomes are
reused; only the new cases need inference on archived R PNGs.

Keep all original recovery/MCQ/transfer/overwrite thresholds and require +.10
semantic-group macro MCQ over R. Report E/R/V state, selective-clear and offline
chain conjunctions. No outcome filtering, optimizer extension, final-test
access, shared Writer or actual RGB recurrence claim is authorized by this run.

Registration digest:
`e050b7bd9d807591dbb54987d8f58020af68eb9514e4dd7cdc040ecd6fa1b0ac`.
Constructor tests: 3 passed. All 896 scenario/format/rotation configurations
are frozen in `compositional-evidence-v1/construction-proof.json`.
