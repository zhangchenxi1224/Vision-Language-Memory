# Registered execution plan: prefeval-rgb-pilot-01

C2C task `c2c_7e19`, iteration 0, ChatGPT review base `4b476bd`.
Plan received after independent inspection of both proposals, code and evidence.

Preserve Base/full U-Net FP32, frozen VAE/condition encoder/Reader, native
28-step CFG1, official FM target-side Gaussian interpolation and previous RGB
as condition only. Inference sees only previous PNG and current event.

1. Group exact duplicates and reviewed near-equivalents before splitting;
   preserve the four-topic OOD boundary. Other topics: approximately 75/12.5/12.5
   percent train/dev/ID test. Seal tests before model-dependent inspection.
   Pilot 64 train / 32 dev; nest pilot inside approximately 25% Small inside Full.
2. Register 64 singleton, eight K2 and eight K4 training episodes with genuine
   SET/OVERWRITE/selective CLEAR/RETAIN. Scope updates explicitly; replacements
   are other original preferences from the same topic and same split. Query
   multiple units from one fixed PNG. No old MCQ labels after changed preference.
3. Measure blank and text references, plus unchanged 4f. Teacher targets only
   on train states, never dev/test. Recover full preference statements, including
   conditions and negation; MCQ application is an independent downstream metric.
4. Teacher sentinel: 16 singleton targets (one per topic), four complete K4
   chains including K2 prefixes. One FP32 latent, Adam lr .05, 256 updates,
   three training query forms, independently normalized changed/untouched loss.
   Fixed endpoint, no seed selection, qualify saved/reopened PNG with held-out
   queries and retain all failed attempts. Roots initialize from gray VAE mean;
   successor initialization uses registered predecessor teacher latent.
5. Allocation gates: text MCQ >=80%, derived checks >=90%; complete PNG teacher
   state qualification >=90% K1 and >=75% K2/K4. First failed empirical gate
   goes back for independent review; no silent extra updates or score relaxation.
6. If gates pass, one pilot Writer: fresh AdamW lr1e-5, global batch4, 2048
   updates. Curriculum 256 K1, 512 K1/K2, 1280 K1/K2/K4. Eval 0/256/768/2048.
   Qualified canonical PNG sources, balanced capacity/operation/topic/group.
   Canonical-source targets >=80% K1 and >=60% K2/K4 before direct scaling.
7. Evaluate actual PNG recurrence, r=0/2/5/10 retention, two fresh paired seeds,
   unchanged 4f baseline, new content vs new combinations separately; fixed-source
   vs self-generated-source diagnostic and directional source swap. Report
   changed, untouched, joint, whole-state, whole-episode and first failure.
   Shared semantic components define uncertainty clusters, not individual reads.
8. After pilot review: Small/Full both restart 4f; 20,000 updates × batch4 each,
   same curriculum and selection protocol. Report teacher cost separately from FM.
   No rollout-source arm unless a canonical/self-source gap motivates it.

This plan is not a successful model result. The user goal remains unfinished.
