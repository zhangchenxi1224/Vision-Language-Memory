from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import decode_model_latents_unit_interval, freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_target_only_ce  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    cuda_peak_memory_report,
    emit_json_report,
    probe_provenance,
    reset_cuda_peak_memory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="DreamLite TinyVAE decode -> Qwen Reader CE gradient probe")
    parser.add_argument("--dreamlite", type=Path, default=ROOT / "models" / "DreamLite-mobile")
    parser.add_argument("--reader", type=Path, default=ROOT / "models" / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--query", default="What is the dominant color? Answer briefly.")
    parser.add_argument("--target", default="red")
    parser.add_argument("--state-resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-out-of-range-fraction",
        type=float,
        default=0.99,
        help="Fail closed when clamp would discard gradients for at least this fraction of decoded pixels.",
    )
    parser.add_argument(
        "--max-zero-gradient-fraction",
        type=float,
        default=0.99,
        help="Fail closed when at least this fraction of pre-clamp image-gradient elements are zero.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    for name in ("max_out_of_range_fraction", "max_zero_gradient_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1].")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real VAE/Reader probe.")

    from diffusers import AutoencoderTiny
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reset_cuda_peak_memory([device])
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
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state = torch.randn(
        1,
        int(vae.config.latent_channels),
        latent_hw,
        latent_hw,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    unclamped_image = decode_model_latents_unit_interval(vae, state, clamp=False)
    unclamped_image.retain_grad()
    out_of_range_fraction = (
        ((unclamped_image < 0.0) | (unclamped_image > 1.0)).float().mean().item()
    )
    image = unclamped_image.clamp(0.0, 1.0)
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
    if unclamped_image.grad is None or not torch.isfinite(unclamped_image.grad).all():
        raise RuntimeError("The pre-clamp image gradient is absent or non-finite.")
    zero_image_grad_fraction = (unclamped_image.grad == 0).float().mean().item()
    if out_of_range_fraction >= args.max_out_of_range_fraction:
        raise RuntimeError(
            "Decoded-image clamp saturation exceeds the registered G1 threshold: "
            f"{out_of_range_fraction:.6f} >= {args.max_out_of_range_fraction:.6f}."
        )
    if zero_image_grad_fraction >= args.max_zero_gradient_fraction:
        raise RuntimeError(
            "Pre-clamp image-gradient sparsity exceeds the registered G1 threshold: "
            f"{zero_image_grad_fraction:.6f} >= {args.max_zero_gradient_fraction:.6f}."
        )
    frozen_gradients = assert_no_frozen_parameter_grads(
        {"vae": vae, "reader": reader},
        fully_frozen={"vae", "reader"},
    )

    report = {
        "probe": "vae_reader_grad",
        "loss": result.loss.item(),
        "state_shape": list(state.shape),
        "state_grad_norm": state.grad.norm().item(),
        "decoded_image_shape": list(image.shape),
        "unclamped_out_of_range_fraction": out_of_range_fraction,
        "unclamped_zero_grad_fraction": zero_image_grad_fraction,
        "max_out_of_range_fraction": args.max_out_of_range_fraction,
        "max_zero_gradient_fraction": args.max_zero_gradient_fraction,
        "reader_processor": processor_name,
        "frozen_gradients": frozen_gradients,
        "cuda_peak_memory": cuda_peak_memory_report([device]),
        "provenance": probe_provenance(
            root=ROOT,
            arguments=args,
            models={"dreamlite": args.dreamlite, "reader": args.reader},
        ),
    }
    emit_json_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
