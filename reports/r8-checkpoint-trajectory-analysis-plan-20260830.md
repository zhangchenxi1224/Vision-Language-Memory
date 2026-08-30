# R8 checkpoint-trajectory analysis plan — locked before outcomes

This analysis is descriptive and cannot select a checkpoint or alter the preregistered R8 decision.

## Purpose

The R8 controllers save raw training checkpoints at optimizer steps 0, 32, 64, 96, and 128. Each checkpoint also contains the exact EMA state. After both arms reach a valid terminal state, evaluate the EMA states at all five fixed steps on the identical hard8 causal suite.

## Fixed analysis

- Arms: `raw-mean-control` and `common-descent-projected-norm-matched`.
- Training lineage: commit `82e983743be73919f257d441f1cacb2b7f601288`.
- Hard8 selection SHA-256: `eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6`.
- Checkpoint steps: 0, 32, 64, 96, 128; no adaptive checkpoint choice.
- Weights: exact EMA state embedded in each raw training checkpoint.
- Evaluation: all eight fixed segments, four reverse-cyclic choice views, and `normal`, `reset`, `cross_episode_swap`, and `temporal_swap` controls.
- Existing trainer rows are reused for M0 and EMA step128; EMA steps 32/64/96 are evaluated post hoc with the same frozen Reader and deterministic CUDA contract.
- Outputs: raw evaluation rows, aggregated CE/accuracy/control summaries, paired checkpoint-versus-M0 estimates, normal/reset difference-in-differences, checkpoint hashes, endpoint binding, source inventory validation, and an output inventory.

## Integrity and interpretation

The evaluator must fail closed unless both source terminal and artifact inventory pass, all five checkpoints share the exact training manifest, every checkpoint cursor matches its filename, and `endpoint_ema.pt` is tensor-identical to the EMA state embedded at step128.

EMA step128 remains the only primary endpoint. Intermediate points may explain learning onset, oscillation, or degradation, but they cannot rescue a failed endpoint, define early stopping, or support a formal-success claim.
