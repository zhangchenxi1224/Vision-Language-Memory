# Model snapshots

This directory is intentionally excluded from Git, except for this note.

Reconstruct snapshots from models.lock.json with scripts/bootstrap/fetch_models.py. Set `VLM_MODEL_ROOT` or pass
`--model-root` when cluster weights belong on shared storage. A metadata-only fetch and a complete snapshot use
different completion markers; cluster preflight accepts only the complete snapshot marker plus weight files.
Training and probes must load these paths with local_files_only=True so a branch update cannot silently change an
experiment.

DreamLite weights retain their non-commercial research license. Never commit tokens or model files to this repo.
