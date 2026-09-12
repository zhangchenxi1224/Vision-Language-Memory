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

## Real parameter export check

Codea333cf8 exported the already completed c90896c full1536 endpoint for engineering verification, using CPU execution with CUDA hidden and oneCPU thread while the new GPU training kept running. This old endpoint remains a known functional failure; its original nativeCFG7.5 setting is preserved in this engineering package.

The exported file contains1075 tensors/389,968,388FP32 parameter values. Every value was loaded back and compared bitwise against the sealed original checkpoint; all match. Size is1,560,240,719 bytes versus4,680,971,012 bytes for the training checkpoint. Export weights SHA6a202dce122dde37877fbaa5c30989754b9ee9295fe99a4945c5cf9fc7ab17b8; package manifest SHA7b8fc78458b155c9ec0e17c8c311a4280bb82e1b50ab021818291ac020529184. The package stays on the shared project disk at `runs/dreamlite-official-alignment/a333cf8-export-engineering-full1536`; large weights are not copied into Git.

See the [exact-value verification](official-alignment-results-20260913/rgb-writer-export-verification.json), [inference manifest](official-alignment-results-20260913/rgb-writer-export-manifest.json) and [completion seal](official-alignment-results-20260913/rgb-writer-export-complete.json). Downloaded manifest bytes match the remote seal. Actual bank-independent native inference/Reader parity has not yet run; this check establishes serialization integrity only.

## Actual CPU pipeline loading

The real `load_writer_package` path has now also completed on the H200 host with CUDA hidden and one CPU thread. Probe4c2f0fd used the unchanged loader in checkout3c335a2 and the same old, known-failing c90896c engineering export. Inputs were only that parameter package, the sealed Base snapshot and pinned official source; no training bank or parent checkpoint was supplied.

All six official pipeline components loaded. The resulting U-Net's1075 tensors/389,968,388 values were compared against the export and were bitwise equal. The U-Net, VAE (2,445,063 parameters) and text encoder (2,127,532,032 parameters) were all frozen FP32 CPU modules. `OfficialRGBMemory` accepted them and constructed its RGB1024×1024 state. CUDA was never initialized. This took31.63 seconds; the separate training GPU process remained active throughout.

The [actual loading result](official-alignment-results-20260913/rgb-writer-real-cpu-loading.json) has verified local/remote SHA2a3055f5a4f0787382ea3cf0be38c19beab5a801fbd6bfbadc658b2445cd4ec1. Writer calls and Reader calls were both zero. This establishes actual package/pipeline integration on CPU, while CUDA sampling parity and functional memory behavior remain unverified for the new endpoint.

## Fixed native inference parity protocol

`scripts/probes/rgb_package_parity.py` prepares a replay from the **first preregistered six-write RGB sequence** of the new 45-condition endpoint, irrespective of whether it succeeds. It requires a completed chain, verifies its artifact hashes, and binds the package to the exact parent checkpoint/result and CFG setting. Preparation emits a separate command stream containing only six event/seed writes and thirty query reads. The inference process receives no reference results, expected answers, bank path, or training checkpoint.

```bash
python scripts/probes/rgb_package_parity.py prepare --reference /path/to/completed-new-rgb-chains --package /path/to/new-package --output /path/to/replay
python scripts/inference/rgb_memory.py --package /path/to/new-package --base-model /path/to/exact-Base-snapshot --official-source /path/to/pinned-DreamLite --reader-model /path/to/exact-Reader-snapshot --commands /path/to/replay/commands.jsonl --output /path/to/new-inference
python scripts/probes/rgb_package_parity.py verify --prepared /path/to/replay --inference /path/to/new-inference
```

Verification compares every PNG byte hash, raw answer, input/output token sequence, EOS/termination field and final persistent image. It rejects missing or reordered operations and altered artifacts. A parity result retains the reference's functional failure flag: two implementations making the same mistake does not become functional success. Two new CPU tests exercise one-token differences, missing reads, image tampering and that distinction; both passed. **The actual GPU replay remains pending the new endpoint and complete chain evidence.**
