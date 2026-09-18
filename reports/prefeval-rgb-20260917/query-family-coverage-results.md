# Plan09 query-family coverage: completed, adoption failed

Training commit `c316456d34fe3b87264c26fdf90939529ee6d4d4`; evaluation
fix `c7aaca0`; immutable retry receipt commit `0b37b49`. Registration digest
`64406da59ad3f1a885720969b80943e7f8b2a24d68e6ca236b7a6f3c66d1d0af`.
The U/V continuation trained on `dl-clear-retain-h200x4-20260914`. Its first
evaluation stopped on a missing derived-XML scoring field. Per the user's
subsequent instance override, the preserved endpoints were evaluated on
`vlm-r11-trust-h200x4-20260907-r3`; no optimization occurred during the retry.

V establishes that broadening the supervised query forms changes what the same
saved RGB endpoint can expose to the frozen Reader. The improvement is real but
does not meet the registered adoption contract. This remains an independent
latent-teacher experiment, not a usable shared DreamLite Writer.

## Paired endpoint evidence

J is the frozen Plan08 parent. U continues J for 64 updates per state using the
unchanged three-form recovery plus full-action ranking objective. V uses one
original recovery form, one rotating new question form and one rotating new
instruction form; its application supervision alternates full-action ranking
and mechanically derived XML choice. Both arms use fresh Adam at learning rate
.01, 2,560 updates and 38,400 gradient Reader forwards. All endpoints froze
before evaluation.

| Metric | J | U | V |
|---|---:|---:|---:|
| Original training-form recovery | 260/264 | 262/264 | 251/264 |
| New recovery-family coverage | - | 488/704 | **650/704** |
| XML application generation | - | 325/336 | **336/336** |
| Original MCQ | 48/84 | 48/84 | 47/84 |
| Training application generation | 169/336 | 163/336 | 138/336 |
| Reserved application generation | 64/168 | 63/168 | 47/168 |
| Training ranking | 334/336 | 336/336 | 336/336 |
| Reserved ranking | 167/168 | 167/168 | 168/168 |
| Complete same-PNG joint states | 19/40 | 21/40 | **26/40** |
| Joint states also passing MCQ | 7/40 | 8/40 | 8/40 |

| Complete recovery | J | U | V | Registered target |
|---|---:|---:|---:|---:|
| K1 | 17/20 | 19/20 | **20/20** | 18/20 |
| K2 | 1/4 | 1/4 | **2/4** | 3/4 |
| K3 | 1/4 | 1/4 | 1/4 | reported |
| K4 | 0/12 | 0/12 | **3/12** | 9/12 |

V versus U rescues 176 new recovery-form generations while losing 14, a net
gain of 162. It rescues the 11 XML errors made by U with no XML regressions.
The benefit does not transfer uniformly: V loses 12 original recovery cases
while rescuing one, loses 34 training application generations while rescuing
9, and loses 16 reserved application generations with no rescues. Ranking and
all eight overwrite-pair contrasts remain intact. Formatting failures fall
from 58 in U to 34 in V, while truncations rise from 2 to 4.

The registered absolute targets require recovery, original MCQ, transfer
ranking, overwrite pairs and every overwrite. U and V pass the three ranking
and overwrite requirements. Both fail recovery and original MCQ. V's K4 3/12
is evidence that query-family coverage reaches multi-slot states, but it is far
below the 9/12 adoption target. No Writer update is authorized.

## Compute and independent verification

The run used 5,120 optimizer updates. Each arm used 38,400 gradient Reader
forwards; U processed 7,807,040 training tokens and V 8,881,248. Summed
per-target training time was 9,897.41 seconds for U and 9,904.33 for V.
Evaluation contained 4,648 generations, 1,584 ranking decisions, 6,336 actual
candidate forwards and 528 recovery-CE forwards, processing 2,490,654 tokens
in 2,919.65 summed seconds. These sums are not job wall time.

The verifier checks exact assignments and budgets, registered prompt families,
source and endpoint hashes, optimizer traces, shared reopened-PNG identities,
the preserved failed evaluation, retry receipt and new-instance runtime
identity. The raw archive SHA-256 is
`1a242987f8c3995d803d2baa92d441306b23271a7e4d2668123637f380c0e752`.
The tracked verified JSON and archive-part receipt provide the detailed
state/slot, paired, rotation and failure-class evidence.
