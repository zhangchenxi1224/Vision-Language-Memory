from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import (  # noqa: E402
    DifferentiableDreamLiteMobileSampler,
    decode_model_latents_raw,
    freeze_module,
)
from vision_memory.dreamlite.conditioning import encode_latent_path_condition  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_listwise_choice_ce,
    qwen3vl_target_only_ce,
)
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    canonical_json_sha256,
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    cuda_peak_memory_report,
    emit_json_report,
    load_source_image,
    lora_trainable_parameters,
    probe_provenance,
    reset_cuda_peak_memory,
    seed_adapter_initialization,
)
from vision_memory.training import format_mcq_query  # noqa: E402


def parameter_grad_stats(parameters) -> tuple[float, int, int]:
    total = 0.0
    tensors_with_grad = 0
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            tensors_with_grad += 1
            nonfinite += int((~torch.isfinite(gradient)).sum().item())
            total += gradient.nan_to_num().square().sum().item()
    return total**0.5, tensors_with_grad, nonfinite


def cuda_compute_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}.")
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def main() -> int:
    parser = argparse.ArgumentParser(description="One/two-event DreamLite -> VAE -> Qwen CE gradient probe")
    parser.add_argument("--dreamlite", type=Path, default=ROOT / "models" / "DreamLite-mobile")
    parser.add_argument("--reader", type=Path, default=ROOT / "models" / "Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Optional RGB input; omitted uses the versioned deterministic fixture.",
    )
    parser.add_argument("--event", action="append", required=True, help="Repeat once or twice")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--reader-loss-mode",
        choices=("legacy-target-only", "listwise-choice"),
        default="legacy-target-only",
        help="R3 scientific gates require listwise-choice; legacy mode is retained for historical probes only.",
    )
    parser.add_argument("--target", help="Required only by legacy-target-only mode")
    parser.add_argument("--choice", action="append", help="Repeat exactly four times for listwise-choice")
    parser.add_argument("--target-index", type=int, help="Correct choice index for listwise-choice")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--noise-seed", "--seed", dest="noise_seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument(
        "--checkpoint-unet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use non-reentrant U-Net activation checkpointing (enabled by default).",
    )
    parser.add_argument("--detach-between-events", action="store_true")
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default=None)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if len(args.event) not in (1, 2):
        raise SystemExit("This gated probe deliberately supports exactly one or two events.")
    if args.reader_loss_mode == "legacy-target-only":
        if not args.target or args.choice is not None or args.target_index is not None:
            raise SystemExit("legacy-target-only requires --target and forbids --choice/--target-index.")
    else:
        if args.target is not None:
            raise SystemExit("listwise-choice forbids --target; the label is selected only by --target-index.")
        if args.choice is None or len(args.choice) != 4 or len(set(args.choice)) != 4:
            raise SystemExit("listwise-choice requires exactly four distinct --choice values.")
        if args.target_index is None or not 0 <= args.target_index < 4:
            raise SystemExit("listwise-choice requires --target-index in [0, 3].")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real end-to-end probe.")

    strict_determinism = configure_strict_cuda_determinism(seed=args.adapter_seed)

    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device or args.dreamlite_device)
    updater_dtype = cuda_compute_dtype(updater_device)
    reader_dtype = cuda_compute_dtype(reader_device)
    reset_cuda_peak_memory([updater_device, reader_device])

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=updater_dtype,
    ).to(updater_device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    seed_adapter_initialization(args.adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()
    lora_parameters = lora_trainable_parameters(pipe.unet)

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
        torch_dtype=reader_dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.eval()
    reader.config.use_cache = False

    source_pil, source_metadata = load_source_image(args.source_image, resolution=args.resolution)
    source_image = pipe.image_processor.preprocess(
        source_pil,
        height=args.resolution,
        width=args.resolution,
    )
    with torch.no_grad():
        state = pipe.prepare_image_latents(source_image, dtype=updater_dtype, device=updater_device)

    sampler = DifferentiableDreamLiteMobileSampler.from_pipeline(
        pipe,
        checkpoint_unet=args.checkpoint_unet,
    )
    intermediate_states = []
    event_noise_seeds = [args.noise_seed + index for index in range(len(args.event))]
    for index, event in enumerate(args.event):
        condition = encode_latent_path_condition(pipe, state, event)
        source_for_update = state.detach() if index > 0 and args.detach_between_events else state
        generator = torch.Generator(device=updater_device).manual_seed(event_noise_seeds[index])
        noise = torch.randn(
            source_for_update.shape,
            generator=generator,
            device=updater_device,
            dtype=updater_dtype,
        )
        state = sampler(
            source_latents=source_for_update,
            noise_latents=noise,
            prompt_embeds=condition.prompt_embeds,
            prompt_attention_mask=condition.attention_mask,
        ).latents
        if index < len(args.event) - 1:
            state.retain_grad()
            intermediate_states.append(state)

    state.retain_grad()
    final_state_sha256 = canonical_tensor_sha256(state)
    decoded_raw = decode_model_latents_raw(pipe.vae, state)
    unclamped_image = decoded_raw * 0.5 + 0.5
    unclamped_image.retain_grad()
    out_of_range_fraction = ((unclamped_image < 0.0) | (unclamped_image > 1.0)).float().mean().item()
    image = unclamped_image.clamp(0.0, 1.0)
    if args.reader_loss_mode == "listwise-choice":
        assert args.choice is not None and args.target_index is not None
        reader_result = qwen3vl_listwise_choice_ce(
            model=reader,
            processor=processor,
            image=image[0].to(reader_device),
            query=format_mcq_query(args.query, args.choice),
            choices=args.choice,
            target_index=args.target_index,
            device=reader_device,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=True,
        )
    else:
        assert args.target is not None
        reader_result = qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image[0].to(reader_device),
            query=args.query,
            target=args.target,
            device=reader_device,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=True,
        )
    if not torch.isfinite(reader_result.loss):
        raise RuntimeError("The Qwen target CE is NaN or Inf.")
    reader_result.loss.backward()

    lora_norm, lora_tensors_with_grad, lora_nonfinite = parameter_grad_stats(lora_parameters)
    intermediate_stats = []
    for item in intermediate_states:
        gradient = item.grad
        intermediate_stats.append(
            {
                "norm": None if gradient is None else gradient.detach().float().nan_to_num().norm().item(),
                "nonfinite_elements": None
                if gradient is None
                else int((~torch.isfinite(gradient.detach())).sum().item()),
            }
        )
    image_gradient = unclamped_image.grad
    final_state_gradient = state.grad
    if (
        final_state_gradient is None
        or not torch.isfinite(final_state_gradient).all()
        or final_state_gradient.norm().item() == 0
    ):
        raise RuntimeError("Qwen CE did not produce a finite, non-zero final latent gradient.")
    if image_gradient is None or not torch.isfinite(image_gradient).all() or image_gradient.norm().item() == 0:
        raise RuntimeError("Qwen CE did not produce a finite, non-zero gradient through the VAE decode.")
    zero_image_grad_fraction = (image_gradient == 0).float().mean().item()
    if lora_tensors_with_grad == 0 or lora_norm == 0 or lora_nonfinite:
        raise RuntimeError(
            "Qwen CE produced absent, zero, or non-finite DreamLite LoRA gradients: "
            f"tensors={lora_tensors_with_grad}, norm={lora_norm}, nonfinite={lora_nonfinite}."
        )
    if not args.detach_between_events and any(
        value["norm"] is None or value["norm"] == 0 or value["nonfinite_elements"]
        for value in intermediate_stats
    ):
        raise RuntimeError(
            "Latent-path BPTT with stop-gradient conditioning did not produce a finite, non-zero "
            "intermediate-state gradient."
        )
    if args.detach_between_events and any(value["norm"] is not None for value in intermediate_stats):
        raise RuntimeError("Detach negative control unexpectedly preserved an intermediate-state gradient.")

    frozen_gradients = assert_no_frozen_parameter_grads(
        {
            "base_unet": pipe.unet,
            "vae": pipe.vae,
            "internal_qwen": pipe.text_encoder,
            "reader": reader,
        },
        fully_frozen={"vae", "internal_qwen", "reader"},
    )
    provenance = probe_provenance(
        root=ROOT,
        arguments=args,
        models={"dreamlite": args.dreamlite, "reader": args.reader},
        source_image=source_metadata,
    )
    pair_metadata = {
        "schema_version": 1,
        "git": provenance["git"],
        "models": provenance["models"],
        "source_image": source_metadata,
        "event": list(args.event),
        "query": args.query,
        "reader_loss_mode": args.reader_loss_mode,
        "target": args.target,
        "choices": args.choice,
        "target_index": args.target_index,
        "resolution": args.resolution,
        "adapter_seed": args.adapter_seed,
        "event_noise_seeds": event_noise_seeds,
        "lora_rank": args.lora_rank,
        "checkpoint_unet": args.checkpoint_unet,
        "dreamlite_device": str(updater_device),
        "reader_device": str(reader_device),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "updater_dtype": str(updater_dtype),
        "reader_dtype": str(reader_dtype),
        "strict_determinism": strict_determinism,
    }

    report = {
        "probe": "e2e_episode_grad",
        "events": len(args.event),
        "detach_between_events": args.detach_between_events,
        "pair_id": canonical_json_sha256(pair_metadata),
        "pair_metadata": pair_metadata,
        "loss": reader_result.loss.item(),
        "reader_loss_mode": args.reader_loss_mode,
        "choice_mean_nll": (
            None
            if args.reader_loss_mode != "listwise-choice"
            else [float(value) for value in reader_result.choice_mean_nll.detach().cpu().tolist()]
        ),
        "final_state_shape": list(state.shape),
        "final_state_sha256": final_state_sha256,
        "final_state_gradient": {
            "norm": final_state_gradient.detach().float().norm().item(),
            "nonfinite_elements": int((~torch.isfinite(final_state_gradient.detach())).sum().item()),
        },
        "intermediate_gradients": intermediate_stats,
        "lora_grad_norm": lora_norm,
        "lora_tensors_with_grad": lora_tensors_with_grad,
        "lora_nonfinite_elements": lora_nonfinite,
        "trainable_lora_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "unclamped_out_of_range_fraction": out_of_range_fraction,
        "unclamped_zero_grad_fraction": zero_image_grad_fraction,
        "unclamped_image_grad_norm": image_gradient.detach().float().norm().item(),
        "adapter_seed": args.adapter_seed,
        "event_noise_seeds": event_noise_seeds,
        "updater_device": str(updater_device),
        "reader_device": str(reader_device),
        "updater_dtype": str(updater_dtype),
        "reader_dtype": str(reader_dtype),
        "reader_processor": processor_name,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "strict_determinism": strict_determinism,
        "frozen_gradients": frozen_gradients,
        "cuda_peak_memory": cuda_peak_memory_report([updater_device, reader_device]),
        "provenance": provenance,
    }
    emit_json_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
