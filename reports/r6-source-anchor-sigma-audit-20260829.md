# R6 source-anchor scheduler-sigma audit

## Outcome

The first R6 launch from commit `a6652400e2dfd038d1f10a168b25df9a6585caac` was intentionally
terminated and is **not valid scientific evidence**.  No endpoint, selection gate, or success
claim exists for either partial arm.

## Root cause

DreamLite uses Diffusers `FlowMatchEulerDiscreteScheduler` with dynamic exponential timestep
shifting.  Its `set_timesteps` method shifts even caller-supplied sigma values.  The original
source-anchor implementation mixed the state as

```text
x = 0.5 * previous_state + 0.5 * event_noise
```

but passed raw sigma `0.5` to the scheduler.  At the fixed 1024x1024 contract the dynamic shift
maps that raw value to an effective first sigma of approximately `0.76`.  The latent state and
the denoiser timestep therefore represented different points on the flow path, invalidating the
claimed pretrained-manifold-consistent intervention.

## Preserved partial evidence

Remote root:

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r6/r6-source-anchor-a665240-20260829
```

| Arm | Completed steps | Last loss | Clip rate | Terminal SHA-256 |
| --- | ---: | ---: | ---: | --- |
| legacy-pure-noise | 27/128 | 16.4655 | 1.000 | `a86d4f4236787cb1b5a879168a14fc1eed8218247b69c9949f8348f35d03ebb1` |
| source-anchored | 24/128 | 20.8336 | 0.000 | `dfcfeac43d0ecbdf0b76f938267f965b1578e8b1af018bf663f657ae8317f26c` |

Both controller terminals record child exit code `-15`, `passed=false`, and no summary.  These
rows are useful only for engineering diagnosis.  They must never be compared as completed
training results.

## Correction and restart rule

The corrected wrapper treats configured edit sigmas as effective post-shift flow sigmas.  It
analytically inverts the scheduler shift, calls `set_timesteps`, verifies the resulting effective
schedule to tolerance, and only then forms the source/noise mixture.  The sigma=1 legacy path
keeps its original raw schedule exactly.  Both arms must restart from the same corrected commit,
fresh output directories, identical selected segments, data hashes, seeds, and evaluation gates.
