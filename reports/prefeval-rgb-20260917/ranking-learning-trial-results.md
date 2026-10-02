# Plan07: fixed paired ranking learning trial

Execution commit: `cb6a6c4`; instance: `dl-clear-retain-h200x4-20260914`.
All 40 archived endpoints received 128 additional updates in each arm with fresh Adam. A used recovery only; B alternated recovery with full-action listwise ranking. All 80 endpoints froze before reserved evaluation. Plan05/06 failed gates remain unchanged.

The fixed trial completed: 10,240 latent updates, 38,656 gradient forwards, 3,408 strict generations and 2,712 ranking decisions. No Writer updates or final-test calls.

## Application outcomes

| Panel | Parent | A | B |
|---|---:|---:|---:|
| Ranking application_training | 169/336 (50.3%) | 177/336 (52.7%) | 323/336 (96.1%) |
| Ranking application_reserved | 79/168 (47.0%) | 78/168 (46.4%) | 165/168 (98.2%) |
| Ranking application_training_rotations | 45/288 (15.6%) | 55/288 (19.1%) | 274/288 (95.1%) |
| Strict generation application_training | 55/336 (16.4%) | 58/336 (17.3%) | 141/336 (42.0%) |
| Strict generation application_reserved | 20/168 (11.9%) | 21/168 (12.5%) | 54/168 (32.1%) |
| Original MCQ | 30/84 (35.7%) | 30/84 (35.7%) | 45/84 (53.6%) |

Reserved ranking semantic-group macros: parent=0.486607, A=0.469643, B=0.988393, text=0.955357, blank=0.607143.

## Complete recovery

| Capacity | A | B |
|---|---:|---:|
| K1 | 19/20 (95.0%) | 15/20 (75.0%) |
| K2 | 3/4 (75.0%) | 0/4 (0.0%) |
| K3 | 1/4 (25.0%) | 0/4 (0.0%) |
| K4 | 2/12 (16.7%) | 0/12 (0.0%) |

Exact recovery on the training question forms: A=264/264 (100.0%), B=186/264 (70.5%).
Exact recovery on the held-out question forms: A=126/176 (71.6%), B=83/176 (47.2%).
Thus B loses recovery even on trained formulations; the deficit is not only wording generalization.

A: complete slot results `{"active": [51, 84], "all": [51, 88], "changed": [27, 40], "cleared": [0, 4], "untouched": [24, 48]}`; selective-clear states 0/4; offline teacher chains 0/4.

B: complete slot results `{"active": [28, 84], "all": [29, 88], "changed": [22, 40], "cleared": [1, 4], "untouched": [7, 48]}`; selective-clear states 0/4; offline teacher chains 0/4.

These are independently optimized endpoint teachers, not recurrent Writer rollouts.

## Overwrite and rotation behavior

- parent: reserved overwrite both-correct 0/8; training contrast cases correct in all four rotations 3/96.
- A: reserved overwrite both-correct 0/8; training contrast cases correct in all four rotations 4/96.
- B: reserved overwrite both-correct 8/8; training contrast cases correct in all four rotations 85/96.
- text: reserved overwrite both-correct 5/8; training contrast cases correct in all four rotations 83/96.
- blank: reserved overwrite both-correct 0/8; training contrast cases correct in all four rotations 10/96.

## Adoption targets

- recovery: FAIL
- mcq: FAIL
- mcq_macro_gain: PASS
- reserved_ranking: PASS
- reserved_ranking_macro_gain: PASS
- reserved_contrasts: PASS
- every_reserved_contrast: PASS

The application ranking gain is substantial within these authored semantic schemas. The complete-recovery and original-MCQ targets are still unmet, so this is not an adopted usable memory version. Strict full-action generation remains a distinct unchanged metric, including failures that add option labels. Reserved scenarios share authored preference schemas; this result does not establish general long-term-memory transfer.

## Compute and evidence

Gradient forwards A=11264, B=27392; actual evaluation candidate forwards=10640. Fixed updates do not imply equal compute.
Training processed tokens: `{"A": 1809792, "B": 6034912}`; evaluation processed tokens=3287522.
Validation: 51 focused tests passed. Both complete Plan07 receipts were reconstructed locally from raw records byte-identically to the remote receipts. Both historical Plan05/06 failed receipts also reconstructed byte-identically. No model calls were added by verification.
The archive preserves latents, fresh optimizer states, exact recovery checkpoints, PNGs, training traces, input-token-bearing generations, candidate scores and full condition-specific prompt proofs.
Archive SHA256: `abbf94cf787fbf835996acd542ee2fc36577fef9b1f4ffa51747f3d8ff5c37d1`.
The archive is distributed as three ordered binary parts; concatenate them before extracting. See `ranking-learning-trial-archive.json` for part hashes and reconstruction order.
