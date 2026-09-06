# R11_new canonical-latent bridge: post-step-128 cosine LR preregistration

Status: preregistered before any result from this variant.

## Why this experiment exists

The parent canonical-latent bridge for target 01 completed with a valid technical
gate but failed both scientific diagnostics:

- MSE ratio at raw step 128: 0.7737632777116902.
- MSE ratio at raw step 256: 0.8105615501236992.
- Raw-step-256 Reader accuracy: 0/4 fixed reverse-cyclic views.
- Formal success: false; Phase 2 remained blocked.

The trajectory contracted through the middle of training and then rebounded.
This is evidence consistent with optimizer overshoot, but it does not establish
that learning rate is the cause. This experiment changes one solver factor to
test that hypothesis.

## Single changed factor

Only the Adam optimizer learning-rate schedule changes. This is explicitly not
the DreamLite diffusion scheduler, sigma schedule, or denoising-step count.

For one-indexed optimizer update u:

    lr(u) = 0.05                                      for 1 <= u <= 128
    lr(u) = 0.025 * (1 + cos(pi * (u - 128) / 128)) for 129 <= u <= 256

Thus updates 1--128 are exactly the parent constant-lr run, update 129 is the
first intervention, update 192 uses lr=0.025, and update 256 uses lr=0.0.
Update 256 must still execute forward, backward, and Adam.step; its gradient
must be finite and nonzero and the internal Adam counter must equal 256. A zero
parameter update at this final zero-lr step is expected.

## Locked causal prefix

Before update 129, the newly generated step-128 checkpoint must match the parent
run on all of the following:

- x_T, endpoint z_t, and all five trajectory tensor hashes;
- canonical hash of the complete Adam state;
- rendered PNG hash;
- optimizer step exactly 128.

The trainer blocks before update 129 if any anchor differs. The locked parent
anchors are:

- x_T: 482724b2a7ac88624c054a543c8267ae0a318f67177206c1a7d9a2c3ac364ddb
- z_t: 34125c8426d0eccf9eca70578c164673d263f592e05bbd2a8b4161be239a5f93
- Adam state: b26457294d573890ba6d7ebf220eb0b4e7dec844338be48186893a6b17ad8004
- PNG: ccfff48bf0bc8cb2ff38cfeb8cbefb845a90ac80deeab3de26326f7b0faf7bd0

## Everything else remains fixed

- Target 01 and its canonical R11 latent teacher.
- Train/dev files, selected segment, event text, and condition tensors.
- Frozen DreamLite and frozen Reader snapshots.
- Full source-anchored four-step DreamLite path with sigmas
  [0.5, 0.375, 0.25, 0.125].
- Only x_T_fp32 is trainable.
- Dense FP32 endpoint-latent MSE objective.
- Adam, weight decay 0, seed 0, strict determinism, 256 updates.
- No gradient clipping.
- Checkpoints at raw steps 0, 64, 128, 192, and 256.
- Raw step 256 is the only primary endpoint; best-checkpoint rescue is forbidden.
- Four fixed reverse-cyclic Reader views and the existing causal reset controls.

The machine-readable lock is
`configs/experiments/r11_new_canonical_latent_bridge_target01_post128_cosine.json`.
Its file SHA-256 is
`c9794f5197f6c62f2f84af3cf0db9aee0ff225b4649004967f52d1424a247dc5`;
its canonical JSON SHA-256 is
`5e6110d8bc02d3495f4ce621ed01dacd8fe9922b4b4a23cfd8c7f2dc7b51985d`.

## Gates and decisions

The original primary bridge gates are unchanged:

- distance: MSE ratio <= 0.01, L2 ratio <= 0.1, and teacher-normalized
  RMSE <= 0.1 at raw step 256;
- Reader transfer: 4/4 reverse-cyclic views correct at raw step 256;
- technical and teacher-replay gates must pass.

This variant also has a secondary, non-rescuing schedule audit. It is eligible
only when the technical, teacher-replay, and step-128 prefix-parity gates pass
and both primary distance and Reader gates fail. It passes only if:

- raw-step-256 MSE ratio <= this run's exact step-128 ratio; and
- raw-step-256 MSE ratio < the parent endpoint ratio 0.8105615501236992.

The secondary audit cannot turn a failed primary bridge gate into scientific
success. The original four-way distance/Reader decision has precedence.
Formal success remains false and Phase 2 remains blocked for every outcome of
this diagnostic.

## Deployment and audit

- Pinned instance: vlm-r3-h200x2-live-20260717.
- A fresh output root and a clean detached checkout are mandatory.
- The implementation commit must be pushed before remote checkout.
- Technical preflight runs before formal optimization.
- Independent aggregation recomputes receipts, tensor distances, Reader logits,
  optimizer-state hashes/counters, prefix parity, gates, and the decision.
- Local readiness at preregistration: 64 focused tests passed.

No result from this new schedule was inspected when these rules were fixed.
