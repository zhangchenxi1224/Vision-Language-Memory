# R13 technical preflight failure and source-hash correction

## Classification

- Run commit: `4bed40d73bb10bbe3a4cafa1b1a57db1a0ea62c9`
- Remote run root: `/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r13/r13-centered-residual-4bed40d-20260904`
- Started: `2026-09-04T11:17:32.020649+00:00`
- Finished: `2026-09-04T11:18:35.433964+00:00`
- Status: technical failure; no scientific outcome.

The fail-closed controller stopped before any Reader inference, optimizer step, checkpoint, or fresh-final model output. It therefore permits a source-only correction without changing or contaminating the preregistered scientific test.

## Observed boundary

Data selection, schedule construction, model loading, and event embedding reconstruction completed. All `192` R12 source event embeddings matched the frozen cache exactly, and the `24` fresh-final event embeddings were merely cached without Reader evaluation. The run then stopped while algebraically reconstructing the frozen R12 common base:

- preregistered fixed-base hash: `11281993533d2db0fcab6b890908bdddc986996552034fe57c8c4f5a432825e8`
- strict-controller fixed-base hash: `51fcc191ac3914ebdbfe07a914d95a51c20458de9a4352de999d7cb8c3595fb5`

## Root cause and falsification

The source-only derivation was rerun with identical tensors, code, Python environment, and PyTorch build under two CPU thread settings:

| Setting | Fixed-base hash | Repetition |
| --- | --- | --- |
| default CPU thread count | `11281993533d2db0fcab6b890908bdddc986996552034fe57c8c4f5a432825e8` | reproduces old preregistration |
| `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | `51fcc191ac3914ebdbfe07a914d95a51c20458de9a4352de999d7cb8c3595fb5` | 2/2 bitwise identical |

The training controller intentionally locks both thread variables to `1`. CPU matrix multiplication therefore used a different floating-point reduction order from the earlier source-only derivation. Train IDs, train feature tensor hash, R11 blank latent, and every R11/R12 source artifact hash remained identical. This is a bit-level reproducibility mismatch, not a scientific or data change.

## Correction boundary

Only deterministic source-derived values are re-locked to the exact controller environment:

- source mean coefficient hash: `93f7de196f335429dfb3c8aaef44a835bb734926b5b1143f7f7cb6036b241f66`
- common latent delta hash: `8292303879f522d0fb590e7f8045c69bec67e9ac04ccdc476177747912c27d7c`
- frozen base latent hash: `51fcc191ac3914ebdbfe07a914d95a51c20458de9a4352de999d7cb8c3595fb5`

The data, frozen model snapshots, event-only information boundary, writer architecture, R12 initialization, optimizer, learning rates, `4,608` micro-forwards, `1,152` fixed optimizer steps, checkpoints, sole endpoint, four causal conditions, target gates, and required pass counts are unchanged.

## Preserved artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `launch.json` | `1d61be87c4af99ed0542448b55ce178173ca8ead90cfcade4ba02af3dc5bab8d` |
| `terminal.json` | `77db34ac9143c9383580ca7312f0876fd835ca994423bc72e3004faa36684d0b` |
| `stderr.log` | `1b7df61a82b19195a9f9918bc600be64a9b1f0b6ff69e5ae5096f548daec8b90` |
| `artifact_inventory.json` | `c60baa03e8a6fb24e07f2f113ce39b2028e39e3a63497cfc8e69f5bc6b9cb7fb` |
| `run/schedule_audit.json` | `5a746f30b4c6ddb44a1838867914e9aebcd646201c2184432c4eb447dcb7c701` |
| `run/selection_audit.json` | `888a0fcd04f19afcab9ec254b33bbf3f6b9b2b0262658818ae305e758ff2193e` |
| `run/event_embedding_audit.json` | `85779d8ec1f6c1269096e5e1f7adc3d19004d8cee32831bb431332be78d98215` |
| `run/event_embedding_cache.pt` | `1d1fec24a7f74bc610ca4af3f1e7f24db0ca4c173245e3da599c7576a86d90d9` |
| `run/feature_audit.json` | `1972ce111f2cfdda928aedfc8cce77d27523425e9c9d6f5f3d35337e55396d6b` |

The failed run remains immutable at the path above. The corrected experiment must use a new Git commit, a new frozen source directory, and a fresh output root.

Complete remote archive: `/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r13/r13-centered-residual-4bed40d-20260904-technical-failure.tar.gz` (`SHA-256 5192a0cd68eff9f9aaad53be12d2ddc5f2cdc9810ec1bb74d466176f55b59a07`).
