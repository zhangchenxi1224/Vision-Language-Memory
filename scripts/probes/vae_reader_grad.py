from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import decode_model_latents_unit_interval, freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_target_only_ce  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DreamLite TinyVAE decode -> Qwen Reader CE gradient probe")
    parser.add_argument("--dreamlite", type=Path, default=ROOT / "models" / "DreamLite-mobile")
    parser.add_argument("--reader", type=Path, default=ROOT / "models" / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--query", default="What is the dominant color? Answer briefly.")
    parser.add_argument("--target", default="red")
    parser.add_argument("--state-resolution", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real VAE/Reader probe.")

    from diffusers import AutoencoderTiny
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    vae = AutoencoderTiny.from_pretrained(
        args.dreamlite,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    freeze_module(vae)
    scale_factor = 2 ** (len(vae.config.encoder_block_out_channels) - 1)

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    processor_name = type(processor.image_processor).__name__
    if "Fast" not in processor_name:
        raise RuntimeError(f"Expected a fast tensor image processor, got {processor_name}")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False

    latent_hw = args.state_resolution // scale_factor
    state = torch.randn(
        1,
        int(vae.config.latent_channels),
        latent_hw,
        latent_hw,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    image = decode_model_latents_unit_interval(vae, state)
    result = qwen3vl_target_only_ce(
        model=reader,
        processor=processor,
        image=image[0],
        query=args.query,
        target=args.target,
        device=device,
    )
    if not torch.isfinite(result.loss):
        raise RuntimeError("The Qwen target CE is NaN or Inf.")
    result.loss.backward()

    if state.grad is None or not torch.isfinite(state.grad).all() or state.grad.norm().item() == 0:
        raise RuntimeError("CE did not produce a finite, non-zero latent gradient.")
    if any(parameter.grad is not None for parameter in vae.parameters()):
        raise RuntimeError("Frozen VAE accumulated parameter gradients.")
    if any(parameter.grad is not None for parameter in reader.parameters()):
        raise RuntimeError("Frozen Reader accumulated parameter gradients.")

    print(
        json.dumps(
            {
                "loss": result.loss.item(),
                "state_shape": list(state.shape),
                "state_grad_norm": state.grad.norm().item(),
                "decoded_image_shape": list(image.shape),
                "reader_processor": processor_name,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
