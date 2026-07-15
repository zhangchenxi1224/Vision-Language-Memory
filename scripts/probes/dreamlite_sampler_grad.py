from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler, freeze_module  # noqa: E402
from vision_memory.dreamlite.conditioning import official_mobile_edit_prompt  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    cuda_peak_memory_report,
    emit_json_report,
    load_source_image,
    lora_trainable_parameters,
    probe_provenance,
    reset_cuda_peak_memory,
    seed_adapter_initialization,
)


def grad_stats(parameters) -> tuple[float, int, int]:
    squared = 0.0
    tensors_with_grad = 0
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            tensors_with_grad += 1
            nonfinite += int((~torch.isfinite(gradient)).sum().item())
            squared += gradient.nan_to_num().square().sum().item()
    return squared**0.5, tensors_with_grad, nonfinite


def main() -> int:
    parser = argparse.ArgumentParser(description="DreamLite-mobile 4-step sampler/LoRA gradient probe")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "DreamLite-mobile")
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Optional RGB input; omitted uses the versioned deterministic fixture.",
    )
    parser.add_argument("--event", default="the background is a quiet blue room")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--noise-seed", "--seed", dest="noise_seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--checkpoint-unet", action="store_true")
    parser.add_argument("--allow-small-gpu", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real DreamLite probe.")
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory_gib < 20 and not args.allow_small_gpu:
        raise SystemExit(f"Only {memory_gib:.1f} GiB VRAM detected; run this probe on the cluster.")

    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reset_cuda_peak_memory([device])
    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    seed_adapter_initialization(args.adapter_seed)
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.eval()
    trainable = lora_trainable_parameters(pipe.unet)

    source_pil, source_metadata = load_source_image(args.source_image, resolution=args.resolution)
    image_tensor = pipe.image_processor.preprocess(
        source_pil,
        height=args.resolution,
        width=args.resolution,
    )
    with torch.no_grad():
        source_latents = pipe.prepare_image_latents(image_tensor, dtype=dtype, device=device)
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            mode="edit",
            prompts=[official_mobile_edit_prompt(args.event)],
            image=source_pil,
            device=device,
            dtype=dtype,
        )
    source_latents = source_latents.detach().requires_grad_(True)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    noise_latents = torch.randn(
        source_latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    sampler = DifferentiableDreamLiteMobileSampler.from_pipeline(
        pipe,
        checkpoint_unet=args.checkpoint_unet,
    )
    output = sampler(
        source_latents=source_latents,
        noise_latents=noise_latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        return_trajectory=True,
    )
    loss = output.latents.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("The sampler surrogate loss is NaN or Inf.")
    loss.backward()

    source_grad = source_latents.grad
    lora_norm, lora_tensors_with_grad, lora_nonfinite = grad_stats(trainable)
    if source_grad is None or not torch.isfinite(source_grad).all() or source_grad.norm().item() == 0:
        raise RuntimeError("The sampler did not preserve a finite source-latent gradient.")
    if lora_tensors_with_grad == 0 or lora_norm == 0 or lora_nonfinite:
        raise RuntimeError(
            "The sampler surrogate loss produced absent, zero, or non-finite LoRA gradients: "
            f"tensors={lora_tensors_with_grad}, norm={lora_norm}, nonfinite={lora_nonfinite}."
        )
    trajectory_length = len(output.trajectory or ())
    if trajectory_length != 5:
        raise RuntimeError(f"Expected initial state plus four denoising states, got {trajectory_length}.")
    frozen_gradients = assert_no_frozen_parameter_grads(
        {
            "base_unet": pipe.unet,
            "vae": pipe.vae,
            "internal_qwen": pipe.text_encoder,
        },
        fully_frozen={"vae", "internal_qwen"},
    )

    report = {
        "probe": "dreamlite_sampler_grad",
        "loss": loss.item(),
        "source_latent_shape": list(source_latents.shape),
        "source_grad_norm": source_grad.norm().item(),
        "lora_grad_norm": lora_norm,
        "lora_tensors_with_grad": lora_tensors_with_grad,
        "lora_nonfinite_elements": lora_nonfinite,
        "trainable_lora_parameters": sum(parameter.numel() for parameter in trainable),
        "trajectory_length": trajectory_length,
        "adapter_seed": args.adapter_seed,
        "event_noise_seeds": [args.noise_seed],
        "frozen_gradients": frozen_gradients,
        "cuda_peak_memory": cuda_peak_memory_report([device]),
        "provenance": probe_provenance(
            root=ROOT,
            arguments=args,
            models={"dreamlite": args.model},
            source_image=source_metadata,
        ),
    }
    emit_json_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
