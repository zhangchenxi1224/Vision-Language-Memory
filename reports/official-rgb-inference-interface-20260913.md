# Standalone RGB memory inference

The inference interface consumes a sealed parameter export, the pinned official Base snapshot/source, an optional frozen Reader, and a JSONL command file. It does not load a teacher bank, oracle tensor, training optimizer, event-to-answer table, or answer labels. A write supplies only event text and an explicit noise seed. A read supplies only a question; it cannot call the Writer or change stored RGB pixels. The persistent memory is the actual generated1024×1024 RGB uint8 image, re-encoded by the official VAE for the next event.

Export a completed official Base full-U-Net run:

```bash
python scripts/inference/export_rgb_writer.py --parent-run /path/to/completed-run --output /path/to/new-package
```

The exporter verifies the completed result/checkpoint hash, cursor, full parameter count and exact training manifest. It writes only FP32 U-Net parameters plus snapshot/protocol metadata. The package remains labelled `experimental_endpoint_requires_independent_functional_validation`; export completion and successful loading do not certify memory functionality or complete the goal. A model already known to fail can be exported for engineering checks without changing that interpretation.

Each command is one JSON object. Additional fields, including gold/answer labels, are rejected:

```jsonl
{"op":"write","event":"For the indigo desk train 001123, remember that the preferred music is ambient.","seed":123}
{"op":"read","query":"What is the current music preference for the indigo desk train 001123?\nUse the memory image to answer.\nAnswer with a short phrase only."}
```

Run an inference session:

```bash
python scripts/inference/rgb_memory.py --package /path/to/package --base-model /path/to/exact-Base-snapshot --official-source /path/to/pinned-DreamLite --reader-model /path/to/exact-Reader-snapshot --commands /path/to/commands.jsonl --output /path/to/new-session
```

Use `--initial-image /path/to/previous-memory.png` to continue from saved memory. The image must already be RGB1024×1024; no implicit resizing or conversion occurs. The Reader is loaded lazily for the first read and uses the same bf16/SDPA, fixed256×256 processor contract and greedy32-token generation as the experiment. The Writer uses FP32, native28 steps and the exact guidance scale recorded in the package. Determinism environment settings apply in a fresh worker process.

Each successful write saves a PNG. `results.jsonl` contains output image hashes or the raw Reader generation; the final PNG is stored as `memory-final.png`. Logs are audit outputs and are never fed back to the Writer. A failed edit does not replace the previous in-memory state. The interface writes a completion record only after all commands finish and their input/package identities remain unchanged.

Current evidence:three unit checks cover parameter-only export/loading without a bank, refusal before any mutation on an invalid final parameter, hash tampering, rejection of label-bearing commands, and repeated read-only queries. These are interface checks. Real exported-weight/native-output parity and full functional validation remain required before a usable-version claim.
