# Fixed transition endpoint collection

The 9628d71 fixed2880-update run has completed. Its final collector, `scripts/reporting/collect_transition_endpoint.py`, has actually verified the sealed endpoint:880/900 matched answers with immediate EOS,176/180 images passing all five prompts, and all11520 deterministic draws. Development failed; see `official-alignment-results-20260913/transition-wording-endpoint-review.md` for the preserved failures and full evidence.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/reporting/collect_transition_endpoint.py --run /path/to/9628d71-transition-wording-full2880-20260913 --bank /path/to/9628d71-transition-wording-bank/manifest.json --output-prefix /path/to/transition-wording-endpoint
```

It binds the source commit, bank bytes, update/accumulation/evaluation budget, official FM, full-U-Net scope and native CFG1/28-step runtime. It checks the final checkpoint SHA, all 180 tensor files plus raw/summary files in each phase, exact 45-group/four-seed/five-question coverage, fixed gold tokenization from the sealed Reader's positive controls, and identical images across question variants. Blank/donor images and raw token outputs must remain unchanged between paired phases.

Every one of the 11,520 training draws is replayed through the fixed CPU generator: condition, teacher, noise seed and sigma must match exactly. Each condition must receive 256 draws. Reporting retains all source/operation/target/wording cells, raw answer counts, paired changes and images passing all five questions. It does not treat training loss as accuracy or a development success as RGB-chain/unseen-entity success.

The text archive preserves the completed run's JSON/JSONL/logs. Large PT files remain on the shared project disk. An explicit `--text-only` local rerun lists every absent PT rather than claiming local weight/tensor verification. Run the full collector remotely first, compare archive hashes after transfer, then use the local mode on downloaded text.

The phase validator was exercised against the actual sealed 1,350-row baseline:0/900 matched and0/180 images passing all five questions. Two CPU tests also reject missing/duplicated rows, forged gold-token scoring and changing the image between question variants. Actual final collection and local text/draw replay have now completed; local verification explicitly lists361 remote-only PT/checkpoint files.

`scripts/reporting/transition_state_geometry.py --run RUN --bank BANK --output NEW_JSON` adds a CPU diagnostic on the completed endpoint. It requires all180 generated tensors and all900 matched raw answers, verifies the phase seal, teacher file/tensor hashes and each generated tensor/Reader image binding, and compares every output latent with the three unchanged teacher states. It retains source, operation, event wording, noise, all raw answers and exact EOS scores. Actual remote analysis has completed:all four failed images are closest to the old jazz teacher. Latent distance does not establish semantic correctness or prove a causal mechanism. The tool does not select images for further training or alter any scoring.
