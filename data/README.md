# Data directory

Generated synthetic JSONL and external dataset snapshots are not committed. Recreate them with:

```bash
python scripts/data/generate_synthetic.py --output-dir data/synthetic_v2 --seed 2026
python scripts/data/generate_synthetic.py \
  --output-dir data/synthetic_v2_set_only --seed 2026 --transition-profile set-only
python scripts/bootstrap/fetch_datasets.py --data-root data/external
```

Every generated split and imported PrefEval source file is content-addressed in its manifest.
The `set-only` curriculum is generated independently; it is never produced by deleting
overwrite/clear turns from the full episodes.

Synthetic schema v2 records both metadata IDs and the actual controlled surfaces:

- `entity_surface` is the exact entity string present in state-changing events and queries.
- `template_family` is a literal marker present in every event and query. Entity surfaces,
  template families, and normalized model-visible template skeletons are split-disjoint and
  hashed under `manifest.json -> surface_partitions`.
- `distractor_variant`, `distractor_pair_id`, and `distractor_episode_id` identify reciprocal
  clean/distractor streams. `query.comparison_id` matches corresponding queries without
  encoding the answer. The clean member contains no `noop` event.
- Mixed queries remain explicit `mixed` turns; their count is recorded per split in the
  manifest and validation report. The router/update code receives turn text only, never these
  analysis fields or `target_index`.

Each semantic counterfactual is crossed with clean/distractor variants in groups of four.
For the full transition profile, all four semantic counterfactual reads that share a visible
query and candidate set also share one deterministic ordered choice tuple. Each candidate is
the target exactly once before the clean/distractor duplication. This makes the target share
conditional on the complete query-only payload exactly 25%, including groups that contain
`no active preference`; the validator fails closed if either the order or target balance drifts.
The formal Qwen sanity gate scores the complete 500-episode dev split after deduplicating
clean/distractor copies by `comparison_id`. For the full profile this produces 500 comparison
reads but only 125 exact model-visible payloads; each payload is paired with all four targets.
The gate saves every comparison read and all four choice NLLs for audit. The independently
generated set-only curriculum does not have that single-factor counterfactual certificate, so
it is explicitly treated as a curriculum-transfer experiment and must pass its own empirical
oracle/query-only Qwen sanity gate before training.

Because each 250-example OOD stratum leaves a two-example residue, those eight episodes are
marked `unpaired` and excluded from matched distractor damage while remaining valid semantic
counterfactual pairs. Length-OOD clean streams use target-consistent overwrite reaffirmations
to preserve the 9–16-turn requirement; the manifest records this definition explicitly.
