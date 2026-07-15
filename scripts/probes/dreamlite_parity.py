from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Official DreamLite-mobile vs differentiable sampler trajectory")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "DreamLite-mobile")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--event", default="the background is a quiet blue room")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real DreamLite parity probe.")

    from diffusers import DreamLiteMobilePipeline

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    image = Image.open(args.source_image).convert("RGB")

    captured: dict[str, object] = {"steps": []}
    original_prepare_latents = pipe.prepare_latents
    original_prepare_image_latents = pipe.prepare_image_latents
    original_encode_prompt = pipe.encode_prompt
    original_scheduler_step = pipe.scheduler.step

    def capture_prepare_latents(*positional, **keywords):
        value = original_prepare_latents(*positional, **keywords)
        captured["initial"] = value.detach().clone()
        return value

    def capture_prepare_image_latents(*positional, **keywords):
        value = original_prepare_image_latents(*positional, **keywords)
        captured["source"] = value.detach().clone()
        return value

    def capture_encode_prompt(*positional, **keywords):
        embeds, mask = original_encode_prompt(*positional, **keywords)
        captured["prompt_embeds"] = embeds.detach().clone()
        captured["prompt_mask"] = mask.detach().clone()
        return embeds, mask

    def capture_scheduler_step(*positional, **keywords):
        value = original_scheduler_step(*positional, **keywords)
        captured["steps"].append(value[0].detach().clone())
        return value

    pipe.prepare_latents = capture_prepare_latents
    pipe.prepare_image_latents = capture_prepare_image_latents
    pipe.encode_prompt = capture_encode_prompt
    pipe.scheduler.step = capture_scheduler_step
    try:
        with torch.no_grad():
            official = pipe(
                prompt=args.event,
                image=image,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=4,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                output_type="latent",
            ).images
    finally:
        pipe.prepare_latents = original_prepare_latents
        pipe.prepare_image_latents = original_prepare_image_latents
        pipe.encode_prompt = original_encode_prompt
        pipe.scheduler.step = original_scheduler_step

    required = ["initial", "source", "prompt_embeds", "prompt_mask"]
    missing = [key for key in required if key not in captured]
    if missing or len(captured["steps"]) != 4:
        raise RuntimeError(f"Failed to capture official trajectory; missing={missing}, steps={len(captured['steps'])}")

    sampler = DifferentiableDreamLiteMobileSampler.from_pipeline(pipe)
    with torch.no_grad():
        custom = sampler(
            source_latents=captured["source"],
            noise_latents=captured["initial"],
            prompt_embeds=captured["prompt_embeds"],
            prompt_attention_mask=captured["prompt_mask"],
            return_trajectory=True,
        )

    official_trajectory = [captured["initial"], *captured["steps"]]
    custom_trajectory = list(custom.trajectory or ())
    if len(official_trajectory) != len(custom_trajectory):
        raise RuntimeError("Official and custom trajectory lengths differ.")

    per_step = []
    all_close = True
    for index, (reference, candidate) in enumerate(zip(official_trajectory, custom_trajectory, strict=True)):
        difference = (reference.float() - candidate.float()).abs()
        close = torch.allclose(reference.float(), candidate.float(), atol=args.atol, rtol=args.rtol)
        all_close = all_close and close
        per_step.append(
            {
                "index": index,
                "allclose": close,
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
            }
        )

    final_difference = (official.float() - custom.latents.float()).abs()
    report = {
        "allclose": all_close,
        "atol": args.atol,
        "rtol": args.rtol,
        "dtype": str(dtype),
        "shape": list(custom.latents.shape),
        "final_max_abs": final_difference.max().item(),
        "steps": per_step,
    }
    print(json.dumps(report, indent=2))
    return 0 if all_close else 3


if __name__ == "__main__":
    raise SystemExit(main())
