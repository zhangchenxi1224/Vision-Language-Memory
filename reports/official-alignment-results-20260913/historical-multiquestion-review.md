# Historical multiquestion evidence rechecked on 2026-09-13

The previous local note covered only the first8 of16 questions. The full remote campaign is complete. The new CPU audit verified all128 runs,32768 optimizer updates,3456 raw generations, complete artifact inventories and checkpoint hashes, exact pinned historical source, preregistered target-panel SHA, matched A/B initialization hashes, all planned seed/arm combinations and independent semantic groups. It recomputed short-answer and token/EOS diagnostics using the historical scorer functions, rather than trusting summary counts.

| Step256,64 paired outputs per arm | A: answer CE | B: answer CE + EOS |
|---|---:|---:|
| Original question, exact answer + immediate EOS | 60/64 | 64/64 |
| First held-out rewrite | 62/64 | 64/64 |
| Second held-out rewrite | 56/64 | 61/64 |
| Blank, each of the three prompts | 0/64 | 0/64 |

The64 outputs are4 repeated seeds for16 independent questions, not64 independent tasks. A different-entity ambient donor yields4/64,8/64,0/64 for the three prompts, identically across A/B; these repeat the same donor image and are not semantic counterfactual successes.

This establishes a broader **direct latent oracle** positive control and supports keeping answer+EOS supervision when preparing future oracle targets. It does not establish shared Writer learning, event-conditioned updates, or recurrent memory. B remains incorrect on3/64 second-rewrite samples. The experiment decoded through BF16 VAE, whereas current official-FM targets use FP32 VAE; do not export these old endpoints into a new bank without fresh same-runtime readback and explicit provenance.

This audit verifies archived artifacts; no model inference or optimizer step was rerun. Model immutability was checked by the historical end audit, not by newly running its whole GPU environment. Source dataset provenance and panel remain recorded in the original experiment; this report does not pretend the old target panel supplies the five-prompt modern bank contract.

Source commit: `2c0e41c899910bf0641f16ee724f85bbe3491a7e`.
Panel SHA256: `d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672`.
Remote campaign: `/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos-multiquestion/paired-2c0e41c-20260908-r01`.

[Machine-readable audit and all128 inventory hashes](historical-multiquestion-audit.json).
Audit implementation: `scripts/reporting/audit_historical_multiquestion.py`, introduced in commit02532ea.
