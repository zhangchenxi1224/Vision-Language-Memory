# R6-SourceAnchor-Bottleneck preregistration

## First-principles diagnosis

The frozen Reader and the end-to-end gradient path are not dead: a single SET can be learned,
R5 full gradients reach earlier states, and the visible-event-prefix semantic visual code reaches
the Reader.  R5 nevertheless fails at fixed multi-step dev endpoints.  The updater law is the
remaining structural mismatch:

```text
R5 event update: fresh pure noise -> four-step full-image generation,
                 while the old memory is only a condition image.
```

Thus the nominal latent state is not actually carried through the diffusion trajectory.  Every
event asks DreamLite to redraw all memory, so preservation and modification are entangled.  The
post-hoc `tau=0.5` rescue partially improved the controlled mechanism suite but not formal dev;
that is evidence worth testing, not a successful result.

DreamLite's own flow-matching training law provides a minimal, principled intervention:

```text
x_sigma = (1-sigma) * previous_state + sigma * fixed_event_noise
```

R5 is exactly `sigma=1`.  R6 compares it with `sigma=0.5`, then integrates the same four-step
DreamLite vector field to zero.  No Reader, data, loss, LoRA, optimizer, or evaluation change is
allowed.

## Paired experiment

Two H200 instances run concurrently:

| Arm | Start state | Meaning |
| --- | --- | --- |
| `legacy-pure-noise` | `x_1 = noise` | exact R5 update-law control |
| `source-anchored` | `x_0.5 = 0.5*old + 0.5*noise` | preserve old state inside the pretrained flow path |

Both repeat the identical hard eight-segment batch: two each from F2 retention, F3 overwrite,
F5 and F6 cross-slot composition.  The purpose is a bottleneck diagnostic: can the architecture
first overfit a small multi-step state algebra, and does that transfer to the unchanged
formal-select-32/mechanism-select-32 suites?

The machine-readable contract and exact gates are in
`configs/experiments/r6_source_anchor_bottleneck.json`.

## Decision boundary

- If only source anchoring passes the hard8 gate, advance it to a fixed full-data pilot.
- If both pass, recurrence is locally learnable; fixed-dev transfer and gradient conflict decide.
- If neither passes, source anchoring is insufficient; use the already-recorded per-segment
  gradients to test gradient balancing next.
- If only the legacy arm passes, reject source anchoring.

Regardless of outcome, this repeated-subset, one-seed diagnostic can never be labeled formal
training success.  Formal success still requires fixed full data, ID/OOD, causal resets/swaps,
paired endpoint-versus-M0 uncertainty, and at least two consistent seeds.

