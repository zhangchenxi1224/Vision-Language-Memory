# Official Base input-format control, fixed before execution

The existing raw event asks the model to remember a preference and includes an entity containing the word `train`. Native Base generations visibly contain trains; this observation alone does not prove a prompt-interface cause.

Test the pretrained Base, with no trained checkpoint and no optimizer updates, on a fixed 2×2×4 panel:

- Events: the original ambient event and a deliberate synthetic intervention replacing only its value with jazz; same entity, slot, query and source image.
- Prompt forms: unchanged event, or the fixed prefix `Create a clean, legible memory note on the image. Use large black text on a plain white background. Record the information in this update: ` followed by that event.
- Four hash-fixed fresh Gaussian noise seeds, shared across event/form pairs; namespace `official-prompt-control-noise`, master seed20260913. Check disjointness against all14000 Base3500 training noise seeds and the8 observed benchmark seeds before generation.
- Native official28 steps, CFG7.5, imageCFG1, exact sealed Base revision, FP32 pipeline/bf16 frozen Reader, five original question variants. Record80 raw greedy32 generations and immediate EOS, plus all16 generated latent/image tensors and PNG previews.

The prefix changes a model instruction; it does not parse gold from a query, draw answer text, consult the oracle latent at inference, or change denoising. The model must generate every pixel. This is a zero-update interface diagnostic and cannot replace the authorized official-FM training or establish recurrent/shared-memory usability.

Run as a separate process on cuda:1, where the primary training process has its frozen Reader loaded but performs its3500 optimization updates on cuda:0. Require70GiB free on the probe device. The training checkout/output remains untouched; this is not a throughput comparison. Inference-only loading freezes all parameters and permits colocating VAE/Writer/Reader for this bounded probe; normal training still requires separate devices.
