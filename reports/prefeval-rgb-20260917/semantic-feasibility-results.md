# Plan05: semantic-supervision feasibility failed before optimization

The one registered feasibility run completed all 672 generations at `9097be7`.
Text is **94/336**, below the fixed 303/336 threshold; blank is **153/336**.
The gate is failed. No paired continuation, Writer optimization, reserved-probe
inference, or final evaluation was run. New optimizer updates: **0**.

Registration digest: `df20224bf3c3ba231e3f90be4dea03154c5c9f39020a17a98f01ff435f15c0ac`.
The 28 values, 112 training scenarios, 56 reserved scenarios and clause audit
were authored from sanitized preference statements and frozen before inference.
No outcomes were used to replace scenarios, alter their targets or drop states.
The planned 20,480-update paired trial remains unexecuted.

## What the original answers show

| Original output category | Text | Blank |
|---|---:|---:|
| Registered strict correct | 94 | 153 |
| Exact correct full action, but forbidden A/B/C/D label added | 239 | 58 |
| Other strict failure | 3 | 125 |

These categories describe the original raw strings, without another model call.
They **do not change the score or pass the gate**. In particular, 239 of the
242 text failures are correct full action text prefixed by a label, despite
the explicit no-label instruction. The frozen output interface is therefore
the dominant obstacle to this feasibility test. The high blank action-match
count also warrants explicit reporting of preference-independent shortcuts;
these authored scenarios alone cannot establish general preference application.
`semantic-feasibility-failure-analysis.json` contains bounded raw examples.

## Evidence and cost

All four shards completed (152, 144, 216, 160 reads), with exit status 0.
The independent verifier reconstructed the same result locally from every raw
generation and EOS trace. Summed post-load shard time is 839.5643 seconds;
this excludes model startup and is not billed GPU time. 49 focused tests passed;
8 semantic-specific tests also passed after reporting additions.

`semantic-transfer-feasibility-v1.tgz` contains all raw reads, four identities,
blank PNGs, logs, completion receipts, original remote verification and execution
commit. Size: 131,164 bytes. SHA-256:
`f4f677ccd0e9dde726138972f06f42228cfc0692c1ece0c646252ced19c3b6d4`.
It is unpacked locally in `semantic-transfer-v1-run/` for connector inspection.
The paired-trial reporter includes slot completeness, selective clearing,
offline teacher chains, parent comparisons and semantic-group macro metrics;
none of those unrun endpoints should be presented as results.

Return this failed gate to ChatGPT for the next bounded plan. Preserve the
original MCQ and recovery evaluation interfaces and the user's RGB-memory main
line. Do not train merely by retrospectively ignoring option labels.
