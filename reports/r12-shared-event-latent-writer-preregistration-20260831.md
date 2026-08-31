# R12 shared event-to-latent writer preregistration

Status: locked after the complete R11 `8/8` VAE-latent oracle and before any R12 model outcome.

## First-principles question

R11 proved that the frozen DreamLite VAE can decode images that causally change the frozen Reader's answer, but it optimized a different answer-supervised latent for every item. R12 asks the next necessary question only:

> Can one shared function receive visible event text, write a VAE latent image, and generalize to held-out entities—without seeing the query, choices, answer index, item ID, or a per-item parameter?

This is deliberately still one-SET F1. Recurrence, overwrite, clear, and interference cannot be interpreted until this shared writing boundary passes.

## Fixed data and information boundary

- Formal data hashes remain unchanged: train `24327ed...d184`, dev `8b167d...7303`.
- Train: 144 unique entities, exactly one item for every `36 target values × 4 target positions`; payload SHA `07ce170b...36d62`.
- Train audit: 36 items, one per target value and `9/9/9/9` positions; SHA `a6de7819...abc4f`.
- Dev select: 24 values, 24 unique entities, `6/6/6/6` positions; SHA `e744484c...901c`.
- Sealed dev final: the same 24-value vocabulary but 24 different entities, `6/6/6/6` positions; SHA `0bd95da0...f0b8`.
- Train, dev-select, and dev-final entity overlaps are exactly zero.
- The writer receives only `event_text`. Query text, choices, target index, answer extracted from `QuerySpec`, segment ID, entity lookup embeddings, and per-item latents are forbidden inputs.
- Dev-select and dev-final never contribute gradients. Dev-final is opened only at the fixed step-1152 endpoint; no best-checkpoint rescue is allowed.

## Shared writer

1. The frozen DreamLite internal Qwen3-VL-2B encodes each event in text-only `generate` mode. Its token states are cached; no query information enters this cache.
2. One learned attention query pools the token states. A shared `2048→512→48` MLP emits bounded coefficients `2*tanh(raw/2)`.
3. The output latent is the exact R11 blank-image latent plus a coefficient-weighted 48-vector latent dictionary. Every dictionary vector is L2-normalized and the output norm scale is fixed at `80`, matching the observed R11 code scale.
4. Dictionary vectors 0–7 initialize from the eight exact R11 endpoint-minus-step0 deltas. Vectors 8–47 use seed-0 Gaussian initialization orthogonalized against preceding vectors. All 48 vectors and the shared pooling/MLP are trainable.
5. The unchanged frozen VAE decodes the latent; the unchanged frozen Qwen3-VL-4B Reader supplies MCQ CE. DreamLite U-Net, semantic edit prompt, event noise, LoRA, and per-target latent optimization are not executed.

The matched constant control uses the identical parameters and optimizer but replaces every event token state with zero. It measures whether one universal image or the query prior can mimic learning.

## Fixed optimization

- AdamW, constant LR `1e-3`; network weight decay `1e-4`, dictionary weight decay `0`.
- 32 epochs × 144 segments = 4,608 micro-forwards; accumulation `4`; exactly 1,152 optimizer steps.
- Every segment sees each forward-cyclic choice view exactly eight times.
- Objective: Reader CE + `0.10·relu(delta_latent_rms−0.50)^2 + 1e-4·mean(coefficient²)`.
- No gradient clipping; any non-finite loss or gradient is a technical failure.
- Raw checkpoints at steps `0/288/576/864/1152`; raw step 1152 is the only primary endpoint.

## Causal evaluation and gates

Each train-audit, dev-select, and dev-final target is evaluated on four disjoint reverse-cyclic views under:

- `normal`: image from its own event;
- `reset`: exact blank `127/255` image;
- `donor`: image from a fixed different-value event in the same split.

The existing R10/R11 target gate is unchanged: at least 20% CE reduction from M0, all four views improve, accuracy rises by at least 0.25, and normal/reset difference-in-differences is negative. R12 adds the same-strength donor gate: normal must beat donor CE by at least 20%, win all four views, improve accuracy by at least 0.25, and have negative normal/donor difference-in-differences.

The conditioned arm passes only at `36/36` train audit, `24/24` dev select, and `24/24` sealed dev final, with the constant-control dev-final arm gate false. Any partial count remains diagnostic.

## Locked interpretation

| Outcome | Interpretation and next action |
| --- | --- |
| Conditioned passes all; constant fails | Shared causal one-SET writing on held-out entities is established. Advance to recurrent SET/OVERWRITE/CLEAR state algebra using this writer. |
| Train passes; dev fails | The writer memorizes held-in events/values or the frozen event representation does not support the shared map. Repair event-to-code generalization before recurrence. |
| Train fails | The shared dictionary, pooling, or optimization cannot fit the known F1 boundary. Attribute gradient flow, coefficient collapse, and basis movement before increasing scope. |
| Normal/reset passes; donor fails | Universal-image/query-prior shortcut: false positive, no advance. |
| Technical failure | Rerun only the invalid arm in a fresh root; make no scientific conclusion. |

Even a full R12 pass is not final Picture Memory success. The project goal still requires recurrent composition, overwrite/clear/interference, fixed full ID/OOD evaluation, multiple seeds, and causal state controls.
