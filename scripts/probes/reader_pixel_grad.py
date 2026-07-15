from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_target_only_ce  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    cuda_peak_memory_report,
    emit_json_report,
    probe_provenance,
    reset_cuda_peak_memory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL float-image -> target-only CE gradient probe")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--query", default="What is the dominant color? Answer briefly.")
    parser.add_argument("--target", default="red")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-small-gpu", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real 4B Reader probe.")
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory_gib < 16 and not args.allow_small_gpu:
        raise SystemExit(f"Only {memory_gib:.1f} GiB VRAM detected; run this probe on the cluster.")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reset_cuda_peak_memory([device])
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    processor_name = type(processor.image_processor).__name__
    if "Fast" not in processor_name:
        raise RuntimeError(f"Expected a fast tensor image processor, got {processor_name}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(model)
    model.config.use_cache = False

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = torch.rand(
        3,
        256,
        256,
        generator=generator,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    result = qwen3vl_target_only_ce(
        model=model,
        processor=processor,
        image=image,
        query=args.query,
        target=args.target,
        device=device,
    )
    if not torch.isfinite(result.loss):
        raise RuntimeError("The Qwen target CE is NaN or Inf.")
    result.loss.backward()

    if image.grad is None or not torch.isfinite(image.grad).all() or image.grad.norm().item() == 0:
        raise RuntimeError("Reader image gradient is absent, non-finite, or zero.")
    frozen_gradients = assert_no_frozen_parameter_grads(
        {"reader": model},
        fully_frozen={"reader"},
    )

    report = {
        "probe": "reader_pixel_grad",
        "loss": result.loss.item(),
        "image_grad_norm": image.grad.norm().item(),
        "pixel_values_shape": list(result.pixel_values.shape),
        "processor": processor_name,
        "dtype": str(dtype),
        "frozen_gradients": frozen_gradients,
        "cuda_peak_memory": cuda_peak_memory_report([device]),
        "provenance": probe_provenance(
            root=ROOT,
            arguments=args,
            models={"reader": args.model},
        ),
    }
    emit_json_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
