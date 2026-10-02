# Image-only chained memory validation

The runtime component src/vision_memory/dreamlite/rgb_memory.py stores only the previous generated uint8 RGB image. Every event invokes the official Base editor with that exact image, a new fixed Gaussian noise seed and the event text. It re-encodes the image through the official VAE rather than passing an unquantized latent around. Queries only read a copy of the image; they never invoke the Writer. Failed edits leave the previous memory intact. No teacher, gold answer, event parser, state classifier or external answer ledger is accepted by this runtime API.

The component has unit coverage for previous-output propagation, read-only image access, failed-edit atomicity and frozen-weight rejection. These tests use a small fake pipeline and do not establish real-model chain success. The real-model probe scripts/probes/official_memory_chains.py is registered before inspecting the current three-state endpoint; execution remains pending a concrete verified endpoint and the appropriate diagnostic results.

Fixed native inference protocol: FP32 official Base28-step sampler, textCFG7.5/imageCFG1, frozen checkpoint and Reader, greedy32-token answers with immediate EOS. Use four state orders: ambient→jazz→clear; jazz→ambient→clear; clear→ambient→jazz; clear→jazz→ambient. Insert the identical delivery/keep-preference no-op event after every write. Four repetitions per order produce16 six-event chains,96 generated images, and480 raw answers across the unchanged five query variants. Together the orders cover all six directed transitions between the three states.

Reset to gray only at the beginning of each six-event chain. Continue with actual generated images even after an incorrect answer; never reset to an oracle or expected state. Read the actual quantized PNG pixels at every step and verify queries leave memory unchanged. Preserve all raw tokens, source/output links, latents, noise, complete trajectories and image hashes. Success requires every state/position/question cell, with four repetitions per cell, to answer correctly and end immediately. Report failures and their propagation, not only final answers.

Noise namespace official-rgb-memory-chain-noise-v1 contains24 distinct seeds; repetition/step seeds are paired across the four orderings. Before execution reject overlap with the parent's complete training noise set and all earlier development, prompt-control, fresh-noise and source-parity namespaces. Bind the exact parent result/checkpoint/model/runtime/bank identities, verify all models remain frozen, and record the full fixed plan. This is still one entity and one semantic question; it does not establish unseen entities, multiple simultaneous facts, or general memory usefulness.

Minimal runtime use with an already loaded and frozen FP32 official Base pipeline:

```python
from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory

memory = OfficialRGBMemory(pipe)
written = memory.write(event_text, seed=explicit_noise_seed)
written.image.save("memory.png")
# Pass memory.image and a query to the frozen Reader separately.
# A later event uses this same session's generated image as its source.
```

The runtime is implemented; trained chain behavior has not yet been measured. Do not advertise this example as a completed usable model.
