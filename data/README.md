# Data directory

Generated synthetic JSONL and external dataset snapshots are not committed. Recreate them with:

```bash
python scripts/data/generate_synthetic.py --output-dir data/synthetic_v1 --seed 2026
python scripts/bootstrap/fetch_datasets.py --data-root data/external
```

Every generated split and imported PrefEval source file is content-addressed in its manifest.
