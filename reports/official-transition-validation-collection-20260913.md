# Independent transition validation collection

`scripts/reporting/collect_transition_validation.py` handles the fixed **0f4076788bf4125bbca8851b6b270d5f8418d534** probe schema. The earlier CFG1 collector is bound to the earlier model and cannot be substituted for this one.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/reporting/collect_transition_validation.py --run /path/to/new-probe --parent /path/to/9628d71-transition-wording-full2880-20260913 --bank /path/to/9628d71-transition-wording-bank/manifest.json --plan reports/official-transition-validation-plan-20260913.json --output-prefix /path/to/new-validation-evidence
```

The collector checks exact parent result/checkpoint bindings, immutable probe/plan identities, all new literal event expressions, all source links, all artifact hashes, and the raw Reader token/EOS scores. It independently recomputes all 900 parent development cells. If development failed, the downstream interpretation must remain `diagnostic_after_development_failure` even if the new test succeeds.

Single writes require all390 rows (360 generated-image answers plus30 controls),72 generated images and152 sealed artifacts. RGB chains require480 rows,96 images,194 sealed artifacts and all16 six-write sequences. A failure on any of the30 questions in a sequence prevents that sequence passing. Category summaries retain original versus new event expressions, initial writes, later writes, no-ops and expected states. Query variants must see the same image. An oracle reset or swapped previous-image reference is rejected.

Two new CPU tests verify control exclusion, absent rows, substituted events, a failed no-op breaking the full chain, and rejected source resets. Together with the completed-endpoint tests, this run reports4 passed. These fixtures validate reporting behavior; **actual new probe results do not exist yet** because the parent training is ongoing.

Full remote collection archives JSON/JSONL and all PNGs, with the original preregistered plan bytes. Local `--text-only` explicitly lists missing remote PT/checkpoint files and still requires all PNGs. Use the plan extracted from the archive for local verification so Windows line-ending conversion does not change its byte hash. The existing `scripts/reporting/verify_rgb_chain_tensors.py` additionally applies to the new96-write schema for real PNG/tensor, independent-noise and29-state trajectory checks; it must actually run after the new chain completes.
