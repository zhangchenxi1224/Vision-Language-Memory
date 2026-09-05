"""R11_new Phase 1A: optimize only x_T through a frozen full DreamLite path.

This is deliberately not canonical R11.  Canonical R11 optimizes the final VAE
model-space latent and bypasses DreamLite.  Here an FP32 initial/noise latent is
the only trainable tensor; the frozen DreamLite conditioner, four-step denoiser,
scheduler, VAE decoder, and Qwen Reader are all retained in the executed path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding  # noqa: E402
from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from vision_memory.dreamlite.conditioning import encode_latent_path_condition  # noqa: E402
from vision_memory.dreamlite.differentiable_mobile import (  # noqa: E402
    DifferentiableDreamLiteMobileSampler,
)
from vision_memory.dreamlite.latent_codec import (  # noqa: E402
    decode_model_latents_unit_interval,
    freeze_module,
)
from vision_memory.event_noise import make_event_generator  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTION_SEED,
    select_f1_targets,
)
from vision_memory.training.r11_new_oracle import (  # noqa: E402
    R11_NEW_CHECKPOINT_STEPS,
    R11_NEW_DIFFUSION_STEPS,
    R11_NEW_DREAMLITE_INPUTS,
    R11_NEW_EFFECTIVE_SIGMA_SCHEDULE,
    R11_NEW_EFFECTIVE_START_SIGMA,
    R11_NEW_LEARNING_RATE,
    R11_NEW_NOISE_KEY,
    R11_NEW_OPTIMIZER_STEPS,
    R11_NEW_PRIMARY_ENDPOINT,
    R11_NEW_PROTOCOL,
    R11_NEW_READER_LOSS_INPUTS,
    R11_NEW_SEED,
    R11_NEW_TARGETS_PAYLOAD_SHA256,
    R11_NEW_TARGET_IDS,
    R11_NEW_WEIGHT_DECAY,
    build_phase1a_schedule,
    phase1a_effective_sigmas_match,
    phase1a_target_gate,
    phase1a_target_statistics,
    phase1a_technical_gate,
    validate_information_boundary,
    validate_phase1a_config,
)


PROTOCOL = R11_NEW_PROTOCOL
IMPLEMENTATION_REVISION = "full-frozen-dreamlite-x-t-fp32-v1"
MANIFEST_SCHEMA = "vision_memory.r11-new-phase1a-manifest.v1"
METRICS_SCHEMA = "vision_memory.r11-new-phase1a-metrics.v1"
CHECKPOINT_SCHEMA = "vision_memory.r11-new-phase1a-checkpoint.v1"
CHECKPOINT_HASH_SCHEMA = "vision_memory.r11-new-phase1a-checkpoint-hashes.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r11-new-phase1a-technical-gate.v1"
SUMMARY_SCHEMA = "vision_memory.r11-new-phase1a-summary.v1"
SUITE = "r11_new_f1_frozen_dreamlite_oracle"

SEED = R11_NEW_SEED
RESOLUTION = 1024
OPTIMIZER_STEPS = R11_NEW_OPTIMIZER_STEPS
LEARNING_RATE = R11_NEW_LEARNING_RATE
WEIGHT_DECAY = R11_NEW_WEIGHT_DECAY
CHECKPOINT_STEPS = R11_NEW_CHECKPOINT_STEPS
NUM_DENOISING_STEPS = R11_NEW_DIFFUSION_STEPS
EDIT_START_SIGMA = R11_NEW_EFFECTIVE_START_SIGMA
EFFECTIVE_SIGMAS = R11_NEW_EFFECTIVE_SIGMA_SCHEDULE
EXPECTED_VIEW_COUNTS = {0: 64, 1: 64, 2: 64, 3: 64}
CONFIG_PATH = ROOT / "configs" / "experiments" / "r11_new_frozen_dreamlite_oracle_phase1a.json"


def _posterior_mean(output: Any) -> Tensor:
    direct = getattr(output, "latents", None)
    if isinstance(direct, Tensor):
        return direct
    distribution = getattr(output, "latent_dist", None)
    if distribution is None and isinstance(output, (tuple, list)) and output:
        distribution = output[0]
    if isinstance(distribution, Tensor):
        return distribution
    if distribution is None:
        raise TypeError(f"Unsupported VAE posterior output: {type(output)!r}")
    mode = getattr(distribution, "mode", None)
    if callable(mode):
        return mode()
    mean = getattr(distribution, "mean", None)
    if isinstance(mean, Tensor):
        return mean
    raise TypeError("VAE posterior exposes neither mode() nor mean.")


def encode_model_latent(vae: nn.Module, unit_image: Tensor) -> Tensor:
    """Encode RGB [0,1] with the same scale/shift contract as canonical R11."""

    posterior = vae.encode(unit_image.mul(2.0).sub(1.0), return_dict=True)
    latent = _posterior_mean(posterior)
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return (latent - shift) * scaling


def blank_source_rgb(*, device: torch.device, dtype: torch.dtype, resolution: int = RESOLUTION) -> Tensor:
    """The preregistered blank source is exactly RGB 127/255, not an image-file default."""

    return torch.full(
        (1, 3, resolution, resolution),
        127.0 / 255.0,
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class OracleForward:
    z_t: Tensor
    image: Tensor
    trajectory: tuple[Tensor, ...]
    effective_sigmas: tuple[float, ...]


class FrozenDreamLiteOracle(nn.Module):
    """One FP32 x_T driving a fully frozen DreamLite-mobile edit trajectory."""

    def __init__(
        self,
        *,
        unet: nn.Module,
        scheduler: Any,
        vae: nn.Module,
        text_encoder: nn.Module,
        source_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        initial_x_t: Tensor,
        compute_dtype: torch.dtype,
        checkpoint_unet: bool = True,
        vae_scale_factor: int = 8,
    ) -> None:
        super().__init__()
        if source_latents.ndim != 4 or source_latents.shape[0] != 1:
            raise ValueError(f"R11_new source latent must be batch-one BCHW: {tuple(source_latents.shape)}")
        if initial_x_t.shape != source_latents.shape:
            raise ValueError("R11_new x_T and source latent shapes must match.")
        if not torch.isfinite(initial_x_t).all():
            raise ValueError("R11_new initial x_T is non-finite.")
        for module in (unet, vae, text_encoder):
            freeze_module(module)
        self.sampler = DifferentiableDreamLiteMobileSampler(
            unet=unet,
            scheduler=scheduler,
            vae_scale_factor=vae_scale_factor,
            checkpoint_unet=checkpoint_unet,
        )
        self.vae = vae
        self.text_encoder = text_encoder
        self.compute_dtype = compute_dtype
        self.x_T_fp32 = nn.Parameter(initial_x_t.detach().float().clone())
        self.register_buffer("source_latents", source_latents.detach().clone(), persistent=False)
        self.register_buffer("prompt_embeds", prompt_embeds.detach().clone(), persistent=False)
        self.register_buffer(
            "prompt_attention_mask",
            prompt_attention_mask.detach().clone(),
            persistent=False,
        )
        self.register_buffer("initial_x_T_fp32", initial_x_t.detach().float().clone(), persistent=False)

    @property
    def unet(self) -> nn.Module:
        return self.sampler.unet

    def forward(self) -> OracleForward:
        result = self.sampler(
            source_latents=self.source_latents,
            noise_latents=self.x_T_fp32.to(dtype=self.compute_dtype),
            prompt_embeds=self.prompt_embeds,
            prompt_attention_mask=self.prompt_attention_mask,
            num_steps=NUM_DENOISING_STEPS,
            return_trajectory=True,
            gradient_mode="full",
            selected_step_indices=None,
            edit_start_sigma=EDIT_START_SIGMA,
        )
        if result.trajectory is None or len(result.trajectory) != NUM_DENOISING_STEPS + 1:
            raise RuntimeError("R11_new requires an exact five-point, four-step DreamLite trajectory.")
        if not phase1a_effective_sigmas_match(result.effective_sigmas):
            raise RuntimeError(
                f"R11_new effective schedule drifted: expected={EFFECTIVE_SIGMAS}, observed={result.effective_sigmas}"
            )
        image = decode_model_latents_unit_interval(self.vae, result.latents, clamp=True)
        return OracleForward(
            z_t=result.latents,
            image=image,
            trajectory=result.trajectory,
            effective_sigmas=result.effective_sigmas,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("technical-preflight", "formal"), required=True)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    # These are protocol constants, intentionally not CLI controls.
    args.seed = SEED
    args.adapter_seed = SEED
    args.pairing_seed = 0
    args.split_seed = 20260730
    args.schedule_seed = R10_SELECTION_SEED
    args.bootstrap_iterations = 10_000
    args.resolution = RESOLUTION
    args.lora_rank = 0
    args.selected_step_count = 0
    args.gradient_mode = "full"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not CONFIG_PATH.is_file():
        raise ValueError(f"R11_new preregistered config is missing: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_phase1a_config(config)
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R11_new {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R11_new {name} path is not a directory: {getattr(args, name)}")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise ValueError("R11_new DreamLite and Reader must use distinct devices.")
    if not args.strict_determinism:
        raise ValueError("R11_new locks strict determinism for preflight and formal runs.")
    fixed_data = config["fixed_data"]
    observed_data = {"train_sha256": r5.sha256_file(args.train), "dev_sha256": r5.sha256_file(args.dev)}
    if observed_data != fixed_data:
        raise ValueError(f"R11_new data binding drifted: expected={fixed_data}, observed={observed_data}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise ValueError("R11_new output path exists and is not a directory.")
        unexpected = [path.name for path in args.output_dir.iterdir() if path.name != ".r11_new_output_owner.json"]
        if unexpected:
            raise ValueError("R11_new refuses a non-empty output directory.")
    if r8.git_value("status", "--porcelain") and not args.allow_dirty:
        raise ValueError("R11_new refuses a dirty source tree unless --allow-dirty is explicit.")


def _atomic_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _claim_output_dir(path: Path) -> Path:
    """Atomically reserve a fresh trainer root before writing any artifact."""

    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("R11_new refuses a pre-existing non-empty output directory.")
    else:
        path.mkdir(parents=True, exist_ok=False)
    owner = path / ".r11_new_output_owner.json"
    payload = {
        "schema": "vision_memory.r11-new-phase1a-output-owner.v1",
        "pid": os.getpid(),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with owner.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return owner


def _append_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
            handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _save_image(path: Path, image: Tensor) -> None:
    value = image.detach().float().cpu()
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError(f"R11_new image artifact must be batch-one RGB: {tuple(value.shape)}")
    array = value[0].clamp(0.0, 1.0).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _tensor_stats(value: Tensor) -> dict[str, Any]:
    fp32 = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": float(fp32.min()),
        "maximum": float(fp32.max()),
        "rms": float(fp32.square().mean().sqrt()),
        "norm": float(fp32.double().norm()),
    }


def _image_stats(value: Tensor) -> dict[str, Any]:
    result = _tensor_stats(value)
    fp32 = value.detach().float()
    result["saturation_fraction"] = float(((fp32 <= 0.0) | (fp32 >= 1.0)).double().mean())
    return result


def _trajectory_stats(values: Sequence[Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        row = {"trajectory_index": index, **_tensor_stats(value)}
        if index:
            row["delta_from_previous_norm"] = float(
                (value.detach().float() - values[index - 1].detach().float()).double().norm()
            )
        rows.append(row)
    return rows


def _condition_record(
    path: Path,
    *,
    prompt_embeds: Tensor,
    attention_mask: Tensor,
    event_text: str,
    recompute_matches: bool,
) -> dict[str, Any]:
    payload = {
        "schema": "vision_memory.r11-new-phase1a-condition.v1",
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "tensor_sha256": {
            "prompt_embeds": canonical_tensor_sha256(prompt_embeds.detach().cpu()),
            "attention_mask": canonical_tensor_sha256(attention_mask.detach().cpu()),
        },
        "event_text_sha256": hashlib.sha256(event_text.encode("utf-8")).hexdigest(),
        "recompute_matches": bool(recompute_matches),
    }
    _atomic_torch_save(path, payload)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "tensor_sha256": dict(payload["tensor_sha256"]),
        "event_text_sha256": payload["event_text_sha256"],
        "recompute_matches": payload["recompute_matches"],
    }


def _verify_condition_record(record: Mapping[str, Any]) -> bool:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _sha256(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hashes = payload.get("tensor_sha256", {})
    return bool(
        payload.get("schema") == "vision_memory.r11-new-phase1a-condition.v1"
        and hashes == record.get("tensor_sha256")
        and bool(payload.get("recompute_matches"))
        and bool(record.get("recompute_matches"))
        and payload.get("event_text_sha256") == record.get("event_text_sha256")
        and hashes.get("prompt_embeds") == canonical_tensor_sha256(payload["prompt_embeds"])
        and hashes.get("attention_mask") == canonical_tensor_sha256(payload["attention_mask"])
    )


def _writer_information_boundary(segment: Any) -> dict[str, Any]:
    event = segment.events[0]
    core_audit = validate_information_boundary(
        dreamlite_inputs=R11_NEW_DREAMLITE_INPUTS,
        noise_key=R11_NEW_NOISE_KEY,
        reader_loss_inputs=R11_NEW_READER_LOSS_INPUTS,
    )
    return {
        "schema": "vision_memory.r11-new-phase1a-information-boundary.v1",
        **core_audit,
        "conditioner_input_names": ["source_latent", "event_text"],
        "sampler_input_names": list(R11_NEW_DREAMLITE_INPUTS),
        "oracle_initialization_key_names": list(R11_NEW_NOISE_KEY),
        "event_source_episode_id_sha256": hashlib.sha256(event.source_episode_id.encode()).hexdigest(),
        "event_source_turn_id": event.noise_turn_id,
        "query_used_only_by_frozen_reader_loss": True,
        "choices_used_only_by_frozen_reader_loss": True,
        "target_index_used_only_by_frozen_reader_loss": True,
        "forbidden_writer_key_violations": [],
    }


def _load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Any:
    """Load the base pipeline directly; PEFT and R5's LoRA loader are forbidden here."""

    from diffusers import DreamLiteMobilePipeline

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    freeze_module(pipe.unet)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    return pipe


def _load_reader(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[Any, nn.Module]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("R11_new requires the fast tensor-native Qwen image processor.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    return processor, reader


def _initial_x_t(source_latents: Tensor, event: Any) -> Tensor:
    generator = make_event_generator(
        device=source_latents.device,
        global_seed=SEED,
        episode_id=event.source_episode_id,
        turn_id=event.noise_turn_id,
    )
    return torch.randn(
        source_latents.shape,
        generator=generator,
        device=source_latents.device,
        dtype=torch.float32,
    )


def _item(segment: Any, state: Tensor) -> Any:
    cpu = state.detach().to(device="cpu", copy=True)
    return r5.EvalItem(
        item_id=segment.segment_id,
        pair_unit=segment.segment_id,
        episode_id=segment.query_source_episode_id,
        query_id=f"{segment.query_source_episode_id}:{segment.query_turn_index}",
        query=segment.query,
        normal_state=cpu,
        temporal_state=cpu,
        family=segment.family,
        target_event_kind=segment.target_event_kind,
        query_gap=segment.query_gap,
        updater_count=segment.updater_count,
        cross_slot_interference=segment.cross_slot_interference,
        stale_target_text=segment.stale_target_text,
    )


def _evaluation_rows(
    *,
    reader_fn: Any,
    segment: Any,
    normal_image: Tensor,
    reset_image: Tensor,
    checkpoint: str,
) -> list[dict[str, Any]]:
    item = _item(segment, normal_image)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for condition, image in (("normal", normal_image), ("reset", reset_image)):
            for view_index, permutation in enumerate(r5.REVERSE_CYCLIC4):
                ordered = tuple(segment.query.choices[index] for index in permutation)
                target_index = permutation.index(segment.query.target_index)
                output = reader_fn(
                    image,
                    r5.format_mcq_query(segment.query.text, ordered),
                    ordered,
                    target_index,
                )
                rows.append(
                    r5._choice_row(
                        reader_output=output,
                        item=item,
                        checkpoint_label=checkpoint,
                        suite=SUITE,
                        condition=condition,
                        permutation=permutation,
                        view_index=view_index,
                        donor_item_id=None,
                    )
                )
    return rows


def _checkpoint_payload(
    *,
    step: int,
    oracle: FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    output: OracleForward,
    manifest_sha256: str,
    condition_sha256: str,
) -> dict[str, Any]:
    trajectory = tuple(value.detach().float().cpu() for value in output.trajectory)
    if len(trajectory) != 5:
        raise RuntimeError("R11_new checkpoint must contain five trajectory tensors.")
    x_t = oracle.x_T_fp32.detach().float().cpu()
    z_t = output.z_t.detach().float().cpu()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "optimizer_step": step,
        "x_T_fp32": x_t,
        "z_t_fp32": z_t,
        "trajectory_fp32": trajectory,
        "effective_sigmas": list(output.effective_sigmas),
        "optimizer": optimizer.state_dict(),
        "manifest_sha256": manifest_sha256,
        "condition_artifact_sha256": condition_sha256,
        "tensor_sha256": {
            "x_T_fp32": canonical_tensor_sha256(x_t),
            "z_t_fp32": canonical_tensor_sha256(z_t),
            "trajectory_fp32": [canonical_tensor_sha256(value) for value in trajectory],
        },
    }


def _save_checkpoint(
    *,
    step: int,
    oracle: FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    manifest_sha256: str,
    condition_sha256: str,
    output_dir: Path,
    output: OracleForward | None = None,
) -> tuple[dict[str, Any], OracleForward]:
    if output is None:
        with torch.no_grad():
            output = oracle()
    pt_path = output_dir / "checkpoints" / f"step-{step:03d}.pt"
    png_path = output_dir / "images" / f"step-{step:03d}.png"
    payload = _checkpoint_payload(
        step=step,
        oracle=oracle,
        optimizer=optimizer,
        output=output,
        manifest_sha256=manifest_sha256,
        condition_sha256=condition_sha256,
    )
    _atomic_torch_save(pt_path, payload)
    _save_image(png_path, output.image)
    record = {
        "schema": CHECKPOINT_HASH_SCHEMA,
        "optimizer_step": step,
        "checkpoint_path": str(pt_path),
        "checkpoint_bytes": pt_path.stat().st_size,
        "checkpoint_sha256": _sha256(pt_path),
        "png_path": str(png_path),
        "png_bytes": png_path.stat().st_size,
        "png_sha256": _sha256(png_path),
        "trajectory_points": len(output.trajectory),
        "effective_sigmas": list(output.effective_sigmas),
        "tensor_sha256": payload["tensor_sha256"],
    }
    _atomic_json(output_dir / "checkpoint_hashes" / f"step-{step:03d}.json", record)
    return record, output


def _verify_checkpoint_record(record: Mapping[str, Any], *, expected_step: int) -> bool:
    pt_path = Path(str(record.get("checkpoint_path", "")))
    png_path = Path(str(record.get("png_path", "")))
    if not pt_path.is_file() or not png_path.is_file():
        return False
    if (
        pt_path.stat().st_size != record.get("checkpoint_bytes")
        or png_path.stat().st_size != record.get("png_bytes")
        or _sha256(pt_path) != record.get("checkpoint_sha256")
        or _sha256(png_path) != record.get("png_sha256")
    ):
        return False
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = payload.get("tensor_sha256", {})
    trajectory = payload.get("trajectory_fp32")
    return bool(
        payload.get("schema") == CHECKPOINT_SCHEMA
        and payload.get("optimizer_step") == expected_step
        and record.get("optimizer_step") == expected_step
        and isinstance(trajectory, (tuple, list))
        and len(trajectory) == 5
        and tensors == record.get("tensor_sha256")
        and tensors.get("x_T_fp32") == canonical_tensor_sha256(payload["x_T_fp32"])
        and tensors.get("z_t_fp32") == canonical_tensor_sha256(payload["z_t_fp32"])
        and tensors.get("trajectory_fp32") == [canonical_tensor_sha256(value) for value in trajectory]
        and phase1a_effective_sigmas_match(payload.get("effective_sigmas"))
        and phase1a_effective_sigmas_match(record.get("effective_sigmas"))
        and payload.get("effective_sigmas") == record.get("effective_sigmas")
    )


def _train_step(
    *,
    step_zero: int,
    oracle: FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    reader_fn: Any,
    segment: Any,
) -> dict[str, Any]:
    try:
        target_index = R11_NEW_TARGET_IDS.index(segment.segment_id)
    except ValueError as exc:
        raise ValueError(f"R11_new received an unlocked target: {segment.segment_id}") from exc
    schedule_step = build_phase1a_schedule(target_index)[step_zero]
    view_index = schedule_step.forward_cyclic_training_view
    permutation = schedule_step.permutation
    ordered = tuple(segment.query.choices[index] for index in permutation)
    target_index = permutation.index(segment.query.target_index)
    optimizer.zero_grad(set_to_none=True)
    output = oracle()
    reader_output = reader_fn(
        output.image,
        r5.format_mcq_query(segment.query.text, ordered),
        ordered,
        target_index,
    )
    loss = reader_output.loss
    if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("R11_new Reader returned an invalid scalar loss.")
    loss.backward()
    gradient = oracle.x_T_fp32.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("R11_new x_T received no finite gradient.")
    gradient_norm = float(gradient.double().norm())
    nonzero_fraction = float((gradient != 0).double().mean())
    if gradient_norm <= 0.0 or nonzero_fraction <= 0.0:
        raise RuntimeError("R11_new x_T received a zero gradient.")
    if any(
        parameter.grad is not None
        for module in (oracle.unet, oracle.vae, oracle.text_encoder)
        for parameter in module.parameters()
    ):
        raise RuntimeError("A frozen DreamLite parameter unexpectedly received a gradient.")
    x_before = oracle.x_T_fp32.detach().clone()
    optimizer.step()
    x_after = oracle.x_T_fp32.detach()
    optimizer.zero_grad(set_to_none=True)
    return {
        "schema": METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step_zero + 1,
        "target_segment_id": segment.segment_id,
        "forward_cyclic_training_view": view_index,
        "permutation": list(permutation),
        "loss_before_step": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "gradient_nonzero_fraction": nonzero_fraction,
        "x_T_before_step": _tensor_stats(x_before),
        "x_T_after_step": _tensor_stats(x_after),
        "x_T_update_norm": float((x_after - x_before).double().norm()),
        "z_t_before_step": _tensor_stats(output.z_t),
        "image_before_step": _image_stats(output.image),
        "trajectory_before_step": _trajectory_stats(output.trajectory),
        "trajectory_points": len(output.trajectory),
        "dreamlite_denoising_steps": len(output.trajectory) - 1,
        "effective_sigmas": list(output.effective_sigmas),
        "full_dreamlite_forward_executed": True,
        "denoiser_steps_executed": len(output.trajectory) - 1,
        "effective_sigma_schedule": list(output.effective_sigmas),
        "gradient_mode": "full",
        "gradient_clipping_applied": False,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }


def _training_view_counts(metrics: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = {index: 0 for index in range(4)}
    for row in metrics:
        if row.get("kind") == "optimizer_step":
            counts[int(row["forward_cyclic_training_view"])] += 1
    return counts


def _all_frozen(module: nn.Module) -> bool:
    return all(not parameter.requires_grad and parameter.grad is None for parameter in module.parameters())


def _technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
    oracle: FrozenDreamLiteOracle,
    reader: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_records: Sequence[Mapping[str, Any]],
    condition_record: Mapping[str, Any],
    snapshots_unchanged: bool,
    information_boundary_passed: bool = True,
    source_contract_verified: bool = True,
    event_noise_contract_verified: bool = True,
) -> dict[str, Any]:
    expected_steps = list(range(1, OPTIMIZER_STEPS + 1))
    trainable = [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    finite = all(
        math.isfinite(float(row[field]))
        for row in metrics
        for field in ("loss_before_step", "gradient_norm", "gradient_nonzero_fraction", "x_T_update_norm")
    )
    schedule_valid = all(
        phase1a_effective_sigmas_match(row.get("effective_sigma_schedule"))
        and int(row.get("denoiser_steps_executed", -1)) == NUM_DENOISING_STEPS
        and int(row.get("trajectory_points", -1)) == NUM_DENOISING_STEPS + 1
        and bool(row.get("full_dreamlite_forward_executed"))
        for row in metrics
    )
    checkpoint_map = {int(row.get("optimizer_step", -1)): row for row in checkpoint_records}
    checkpoint_hashes_valid = set(checkpoint_map) == set(CHECKPOINT_STEPS) and all(
        _verify_checkpoint_record(checkpoint_map[step], expected_step=step) for step in CHECKPOINT_STEPS
    )
    optimizer_valid = bool(
        type(optimizer) is torch.optim.Adam
        and len(optimizer.param_groups) == 1
        and float(optimizer.param_groups[0]["lr"]) == LEARNING_RATE
        and float(optimizer.param_groups[0]["weight_decay"]) == WEIGHT_DECAY
        and len(optimizer.param_groups[0]["params"]) == 1
        and optimizer.param_groups[0]["params"][0] is oracle.x_T_fp32
    )
    views = _training_view_counts(metrics)
    frozen_gradients_absent = all(
        _all_frozen(module) for module in (oracle.unet, oracle.vae, oracle.text_encoder, reader)
    )
    core_audit = {
        "checkpoint_steps_observed": sorted(checkpoint_map),
        "latent_checkpoint_steps_observed": sorted(checkpoint_map),
        "image_checkpoint_steps_observed": sorted(checkpoint_map),
        "trainable_parameter_names": trainable,
        "trainable_parameter_dtypes": {
            name: str(parameter.dtype) for name, parameter in oracle.named_parameters() if parameter.requires_grad
        },
        "frozen_components": {
            "dreamlite_unet": _all_frozen(oracle.unet),
            "condition_encoder": _all_frozen(oracle.text_encoder),
            "vae": _all_frozen(oracle.vae),
            "reader": _all_frozen(reader),
        },
        "frozen_gradients_absent": frozen_gradients_absent,
        "full_model_snapshots_unchanged": snapshots_unchanged,
        "information_boundary_passed": information_boundary_passed,
        "source_contract_verified": source_contract_verified,
        "event_noise_contract_verified": event_noise_contract_verified,
        "optimizer": "Adam" if type(optimizer) is torch.optim.Adam else type(optimizer).__name__,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "gradient_clip": None,
        "primary_endpoint": R11_NEW_PRIMARY_ENDPOINT,
    }
    core_gate = phase1a_technical_gate(
        metrics,
        target_segment_id=target_segment_id,
        audit=core_audit,
    )
    local_valid = bool(
        finite
        and schedule_valid
        and checkpoint_hashes_valid
        and _verify_condition_record(condition_record)
        and optimizer_valid
        and all(not bool(row.get("gradient_clipping_applied")) for row in metrics)
    )
    return {
        **core_gate,
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": bool(core_gate["passed"] and local_valid),
        "optimizer_step_records": len(metrics),
        "optimizer_steps_exact": [int(row["optimizer_step"]) for row in metrics] == expected_steps,
        "training_view_counts": views,
        "expected_training_view_counts": EXPECTED_VIEW_COUNTS,
        "four_step_schedule_exact": schedule_valid,
        "effective_sigmas_expected": list(EFFECTIVE_SIGMAS),
        "checkpoint_steps_observed": sorted(checkpoint_map),
        "checkpoint_hashes_valid": checkpoint_hashes_valid,
        "condition_artifact_valid": _verify_condition_record(condition_record),
        "core_audit": core_audit,
        "trainable_parameter_names": trainable,
        "only_x_T_fp32_trainable": trainable == ["x_T_fp32"],
        "unet_frozen": _all_frozen(oracle.unet),
        "vae_frozen": _all_frozen(oracle.vae),
        "text_encoder_frozen": _all_frozen(oracle.text_encoder),
        "reader_frozen": _all_frozen(reader),
        "optimizer_contract_valid": optimizer_valid,
        "gradient_clipping_forbidden_and_absent": all(
            not bool(row.get("gradient_clipping_applied")) for row in metrics
        ),
        "snapshots_unchanged": snapshots_unchanged,
        "finite_metrics": finite,
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in metrics),
        "minimum_gradient_nonzero_fraction": min(float(row["gradient_nonzero_fraction"]) for row in metrics),
    }


def _runtime_versions() -> dict[str, Any]:
    distributions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "peft"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": distributions,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _write_environment(path: Path) -> None:
    installed = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(installed) + "\n", encoding="utf-8")


def _snapshot_bindings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dreamlite_mobile": r5._verified_snapshot_payload(
            model_dir=args.dreamlite,
            model_key="dreamlite_mobile",
            env_name="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
        "qwen_reader": r5._verified_snapshot_payload(
            model_dir=args.reader,
            model_key="qwen_reader",
            env_name="VLM_READER_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
    }


def _manifest(
    *,
    args: argparse.Namespace,
    data: Any,
    selected: Sequence[Any],
    target: Any,
    source_rgb: Tensor,
    source_latents: Tensor,
    initial_x_t: Tensor,
    condition_record: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config_sha = _sha256(CONFIG_PATH) if CONFIG_PATH.is_file() else None
    information_boundary = _writer_information_boundary(target)
    if not information_boundary["passed"]:
        raise RuntimeError(f"R11_new information boundary failed: {information_boundary}")
    expected_source = blank_source_rgb(
        device=source_rgb.device,
        dtype=source_rgb.dtype,
        resolution=source_rgb.shape[-1],
    )
    source_contract_verified = bool(torch.equal(source_rgb, expected_source))
    expected_x_t = _initial_x_t(source_latents, target.events[0])
    event_noise_contract_verified = bool(torch.equal(initial_x_t.float(), expected_x_t.float()))
    if not source_contract_verified or not event_noise_contract_verified:
        raise RuntimeError("R11_new source or event-noise initialization contract drifted.")
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": args.mode,
        "git_commit": r8.git_value("rev-parse", "HEAD"),
        "git_dirty": bool(r8.git_value("status", "--porcelain")),
        "preregistered_config_path": str(CONFIG_PATH),
        "preregistered_config_sha256": config_sha,
        "target_index": args.target_index,
        "target_segment": target.to_dict(),
        "target_segment_id": target.segment_id,
        "selected_segment_ids": [segment.segment_id for segment in selected],
        "selected_segments_sha256": R11_NEW_TARGETS_PAYLOAD_SHA256,
        "train_sha256": r5.sha256_file(args.train),
        "dev_sha256": r5.sha256_file(args.dev),
        "family_pool_audit": data.pool_audit,
        "dev_split_audit": data.split_audit,
        "model_snapshot_payloads_start": dict(snapshot_bindings),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "source_rgb": {
            "value": "127/255",
            "shape": list(source_rgb.shape),
            "dtype": str(source_rgb.dtype),
            "sha256": canonical_tensor_sha256(source_rgb.detach().cpu()),
        },
        "source_latents": {
            "shape": list(source_latents.shape),
            "dtype": str(source_latents.dtype),
            "sha256": canonical_tensor_sha256(source_latents.detach().float().cpu()),
        },
        "initial_x_T_fp32": {
            "distribution": "standard Gaussian N(0,I)",
            "noise_key": [SEED, target.events[0].source_episode_id, target.events[0].noise_turn_id],
            "shape": list(initial_x_t.shape),
            "dtype": str(initial_x_t.dtype),
            "sha256": canonical_tensor_sha256(initial_x_t.detach().float().cpu()),
        },
        "condition_artifact": dict(condition_record),
        "information_boundary": information_boundary,
        "source_contract_verified": source_contract_verified,
        "event_noise_contract_verified": event_noise_contract_verified,
        "fixed_contract": {
            "resolution": RESOLUTION,
            "only_trainable": "x_T_fp32",
            "dreamlite_pipeline_load": "DreamLiteMobilePipeline.from_pretrained directly; no PEFT",
            "unet_frozen": True,
            "vae_frozen": True,
            "text_encoder_frozen": True,
            "reader_frozen": True,
            "dreamlite_unet_executed": True,
            "official_full_conditioner_precomputed": True,
            "gradient_mode": "full",
            "checkpoint_unet": True,
            "num_denoising_steps": NUM_DENOISING_STEPS,
            "edit_start_sigma": EDIT_START_SIGMA,
            "effective_sigmas": list(EFFECTIVE_SIGMAS),
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer_steps": OPTIMIZER_STEPS if args.mode == "formal" else 0,
            "preflight_backward_calls": 1 if args.mode == "technical-preflight" else 0,
            "gradient_clipping": None,
            "checkpoint_steps": list(CHECKPOINT_STEPS) if args.mode == "formal" else [0],
            "training_views": "forward cyclic",
            "endpoint_views": "reverse cyclic",
            "primary_endpoint": R11_NEW_PRIMARY_ENDPOINT,
            "best_checkpoint_selection_forbidden": True,
            "reset": "decode(blank_source_latent)",
        },
        "query_level_diagnostic_only": True,
        "formal_success_gate": False,
    }


def _load_runtime(
    args: argparse.Namespace,
) -> tuple[Any, Any, nn.Module, FrozenDreamLiteOracle, Tensor, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("R11_new Phase 1A requires CUDA.")
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = r5.compute_dtype(updater_device)
    reader_dtype = r5.compute_dtype(reader_device)
    data = r5._load_data(args, optimizer_steps=0)
    selected = select_f1_targets(data.train_pools)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    selected_ids = tuple(segment.segment_id for segment in selected)
    if selected_ids != R11_NEW_TARGET_IDS or selected_sha != R11_NEW_TARGETS_PAYLOAD_SHA256:
        raise RuntimeError(f"R11_new F1 selection drifted: {selected_sha}")
    target = selected[args.target_index]
    if target.family != "F1" or len(target.events) != 1:
        raise RuntimeError("R11_new Phase 1A requires exactly one F1 event.")
    pipe = _load_pipeline(args, updater_device, updater_dtype)
    processor, reader = _load_reader(args, reader_device, reader_dtype)
    source_rgb = blank_source_rgb(device=updater_device, dtype=updater_dtype)
    with torch.no_grad():
        source_latents = encode_model_latent(pipe.vae, source_rgb)
        condition = encode_latent_path_condition(pipe, source_latents, target.events[0].event_text)
        repeated_condition = encode_latent_path_condition(
            pipe,
            source_latents,
            target.events[0].event_text,
        )
    condition_recompute_matches = bool(
        torch.equal(condition.prompt_embeds, repeated_condition.prompt_embeds)
        and torch.equal(condition.attention_mask, repeated_condition.attention_mask)
    )
    if not condition_recompute_matches:
        raise RuntimeError("R11_new official conditioner is not repeatable for the locked source/event.")
    condition_record = _condition_record(
        args.output_dir / "condition" / "official_full_condition.pt",
        prompt_embeds=condition.prompt_embeds,
        attention_mask=condition.attention_mask,
        event_text=target.events[0].event_text,
        recompute_matches=condition_recompute_matches,
    )
    oracle = FrozenDreamLiteOracle(
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        source_latents=source_latents,
        prompt_embeds=condition.prompt_embeds,
        prompt_attention_mask=condition.attention_mask,
        initial_x_t=_initial_x_t(source_latents, target.events[0]),
        compute_dtype=updater_dtype,
        checkpoint_unet=True,
        vae_scale_factor=int(pipe.vae_scale_factor),
    )
    context = {
        "data": data,
        "selected": selected,
        "selected_sha": selected_sha,
        "target": target,
        "pipe": pipe,
        "source_rgb": source_rgb,
        "source_latents": source_latents,
        "condition_record": condition_record,
        "updater_device": updater_device,
        "reader_device": reader_device,
    }
    return processor, pipe, reader, oracle, source_latents, context


def _preflight(
    *,
    args: argparse.Namespace,
    oracle: FrozenDreamLiteOracle,
    reader: nn.Module,
    reader_fn: Any,
    target: Any,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    schedule_step = build_phase1a_schedule(args.target_index)[0]
    permutation = schedule_step.permutation
    ordered = tuple(target.query.choices[index] for index in permutation)
    target_index = permutation.index(target.query.target_index)
    optimizer.zero_grad(set_to_none=True)
    output = oracle()
    result = reader_fn(
        output.image,
        r5.format_mcq_query(target.query.text, ordered),
        ordered,
        target_index,
    )
    loss = result.loss
    if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("R11_new preflight Reader loss is invalid.")
    loss.backward()  # Exactly one preflight backward; optimizer.step() is forbidden.
    gradient = oracle.x_T_fp32.grad
    gradient_valid = bool(
        gradient is not None
        and torch.isfinite(gradient).all()
        and float(gradient.double().norm()) > 0.0
        and float((gradient != 0).double().mean()) > 0.0
    )
    frozen_valid = all(_all_frozen(module) for module in (oracle.unet, oracle.vae, oracle.text_encoder, reader))
    trainable = [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    manifest_sha = _sha256(args.output_dir / "manifest.json")
    checkpoint, _ = _save_checkpoint(
        step=0,
        oracle=oracle,
        optimizer=optimizer,
        manifest_sha256=manifest_sha,
        condition_sha256=str(manifest["condition_artifact"]["sha256"]),
        output_dir=args.output_dir,
        output=OracleForward(
            z_t=output.z_t.detach(),
            image=output.image.detach(),
            trajectory=tuple(value.detach() for value in output.trajectory),
            effective_sigmas=output.effective_sigmas,
        ),
    )
    observed_bindings = {name: verify_snapshot_binding(binding) for name, binding in snapshot_bindings.items()}
    snapshots_unchanged = observed_bindings == dict(snapshot_bindings)
    _atomic_json(
        args.output_dir / "model_snapshot_verification_end.json",
        {
            "schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1",
            "passed": snapshots_unchanged,
            "bindings": observed_bindings,
        },
    )
    checkpoint_valid = _verify_checkpoint_record(checkpoint, expected_step=0)
    passed = bool(
        gradient_valid
        and frozen_valid
        and trainable == ["x_T_fp32"]
        and checkpoint_valid
        and snapshots_unchanged
        and bool(manifest["information_boundary"]["passed"])
        and bool(manifest["source_contract_verified"])
        and bool(manifest["event_noise_contract_verified"])
        and _verify_condition_record(manifest["condition_artifact"])
    )
    report = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_technical",
        "mode": "technical-preflight",
        "passed": passed,
        "backward_calls": 1,
        "optimizer_steps": 0,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient.double().norm()) if gradient is not None else None,
        "gradient_nonzero_fraction": (float((gradient != 0).double().mean()) if gradient is not None else None),
        "trajectory_points": len(output.trajectory),
        "dreamlite_denoising_steps": len(output.trajectory) - 1,
        "effective_sigmas": list(output.effective_sigmas),
        "trainable_parameter_names": trainable,
        "all_models_frozen": frozen_valid,
        "snapshots_unchanged": snapshots_unchanged,
        "condition_artifact_valid": _verify_condition_record(manifest["condition_artifact"]),
        "checkpoint": checkpoint,
        "scientific_gate_evaluated": False,
        "phase1a_reachability_gate": None,
        "gates": {
            "technical_gate": passed,
            "phase1a_query_level_reachability_gate": None,
            "formal_success_gate": False,
        },
        "formal_success_gate": False,
        "query_level_diagnostic_only": True,
    }
    _atomic_json(args.output_dir / "technical_preflight.json", report)
    return report


def _formal(
    *,
    args: argparse.Namespace,
    oracle: FrozenDreamLiteOracle,
    reader: nn.Module,
    train_reader: Any,
    eval_reader: Any,
    target: Any,
    source_latents: Tensor,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_sha = _sha256(args.output_dir / "manifest.json")
    reset_image = decode_model_latents_unit_interval(oracle.vae, source_latents, clamp=True).detach()
    with torch.no_grad():
        initial_output = oracle()
    rows_path = args.output_dir / "target_evaluation_rows.jsonl"
    m0_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=initial_output.image.detach(),
        reset_image=reset_image,
        checkpoint="m0",
    )
    _append_jsonl(rows_path, m0_rows)
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint, _ = _save_checkpoint(
        step=0,
        oracle=oracle,
        optimizer=optimizer,
        manifest_sha256=manifest_sha,
        condition_sha256=str(manifest["condition_artifact"]["sha256"]),
        output_dir=args.output_dir,
        output=initial_output,
    )
    checkpoint_records.append(checkpoint)
    metrics: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    endpoint_output: OracleForward | None = None
    for step_zero in range(OPTIMIZER_STEPS):
        metric = _train_step(
            step_zero=step_zero,
            oracle=oracle,
            optimizer=optimizer,
            reader_fn=train_reader,
            segment=target,
        )
        metric["elapsed_seconds"] = time.monotonic() - started
        metrics.append(metric)
        _append_jsonl(metrics_path, (metric,))
        step = step_zero + 1
        if step in CHECKPOINT_STEPS:
            checkpoint, snapshot = _save_checkpoint(
                step=step,
                oracle=oracle,
                optimizer=optimizer,
                manifest_sha256=manifest_sha,
                condition_sha256=str(manifest["condition_artifact"]["sha256"]),
                output_dir=args.output_dir,
            )
            checkpoint_records.append(checkpoint)
            if step == OPTIMIZER_STEPS:
                endpoint_output = snapshot
        print(
            json.dumps(
                {
                    "milestone": "r11_new_phase1a_optimizer_step",
                    "target_index": args.target_index,
                    "optimizer_step": step,
                    "loss": metric["loss_before_step"],
                    "gradient_norm": metric["gradient_norm"],
                    "elapsed_seconds": metric["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if endpoint_output is None:
        raise RuntimeError("R11_new raw step-256 endpoint was not materialized.")
    endpoint_label = R11_NEW_PRIMARY_ENDPOINT
    endpoint_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=endpoint_output.image.detach(),
        reset_image=reset_image,
        checkpoint=endpoint_label,
    )
    _append_jsonl(rows_path, endpoint_rows)
    endpoint_checkpoint = args.output_dir / "checkpoints" / "step-256.pt"
    endpoint_png = args.output_dir / "images" / "step-256.png"
    shutil.copyfile(endpoint_checkpoint, args.output_dir / "endpoint_raw.pt")
    shutil.copyfile(endpoint_png, args.output_dir / "endpoint_raw.png")

    observed_bindings = {name: verify_snapshot_binding(binding) for name, binding in snapshot_bindings.items()}
    snapshots_unchanged = observed_bindings == dict(snapshot_bindings)
    snapshot_end = {
        "schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1",
        "passed": snapshots_unchanged,
        "bindings": observed_bindings,
    }
    _atomic_json(args.output_dir / "model_snapshot_verification_end.json", snapshot_end)
    technical = _technical_gate(
        metrics,
        target_segment_id=target.segment_id,
        oracle=oracle,
        reader=reader,
        optimizer=optimizer,
        checkpoint_records=checkpoint_records,
        condition_record=manifest["condition_artifact"],
        snapshots_unchanged=snapshots_unchanged,
        information_boundary_passed=bool(manifest["information_boundary"]["passed"]),
        source_contract_verified=bool(manifest["source_contract_verified"]),
        event_noise_contract_verified=bool(manifest["event_noise_contract_verified"]),
    )
    _atomic_json(args.output_dir / "technical_gate.json", technical)
    statistics = phase1a_target_statistics(
        m0_rows + endpoint_rows,
        suite=SUITE,
        target_segment_id=target.segment_id,
        endpoint=endpoint_label,
    )
    reachability_gate = phase1a_target_gate(statistics, technical_gate=True) if bool(technical["passed"]) else None
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "optimizer_steps": OPTIMIZER_STEPS,
        "primary_endpoint": endpoint_label,
        "technical_gate": technical,
        "target_statistics": statistics,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "phase1a_query_level_reachability_gate": reachability_gate,
            "formal_success_gate": False,
        },
        "query_level_diagnostic_only": True,
        "state_level_reachability_not_tested": True,
        "shared_writer_not_trained": True,
        "formal_success_gate": False,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": [row["optimizer_step"] for row in checkpoint_records],
        "artifacts": {
            "manifest_sha256": manifest_sha,
            "metrics_sha256": _sha256(metrics_path),
            "evaluation_rows_sha256": _sha256(rows_path),
            "endpoint_raw_sha256": _sha256(args.output_dir / "endpoint_raw.pt"),
            "endpoint_png_sha256": _sha256(args.output_dir / "endpoint_raw.png"),
            "snapshot_end_sha256": _sha256(args.output_dir / "model_snapshot_verification_end.json"),
            "technical_gate_sha256": _sha256(args.output_dir / "technical_gate.json"),
        },
        "wall_clock_seconds": time.monotonic() - started,
    }


def _artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "vision_memory.r11-new-phase1a-artifact-inventory.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _report_artifact_rows(root: Path) -> list[dict[str, Any]]:
    """Hash already-materialized evidence without making REPORT self-referential."""

    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name in {"REPORT.md", "artifact_inventory.json"}:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _markdown_scalar(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _write_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Render a stable report from machine-readable evidence only."""

    mode = str(summary.get("mode"))
    if mode not in {"technical-preflight", "formal"}:
        raise ValueError(f"R11_new report received an invalid mode: {mode!r}")
    gates = summary.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("R11_new report requires machine-readable gates.")
    technical = bool(gates.get("technical_gate"))
    reachability = gates.get("phase1a_query_level_reachability_gate")
    if bool(gates.get("formal_success_gate")) or bool(summary.get("formal_success_gate")):
        raise ValueError("R11_new Phase 1A can never set formal_success_gate=true.")
    if mode == "technical-preflight":
        if reachability is not None or bool(summary.get("scientific_gate_evaluated")):
            raise ValueError("R11_new preflight must not contain a scientific/diagnostic judgment.")
        diagnostic_text = "未评估：preflight 只执行一次 backward，不执行 optimizer step。"
    elif not technical:
        diagnostic_text = "不可判定：工程技术门未通过，query-level reachability 无有效科学解释。"
    elif bool(reachability):
        diagnostic_text = "通过：达到预注册的 query-level full-chain reachability 诊断门。"
    else:
        diagnostic_text = "未通过：在锁定预算下未达到 query-level reachability 门；不等于证明数学上不可达。"

    target_id = manifest.get("target_segment_id", summary.get("target_segment_id", "unknown"))
    endpoint = summary.get("primary_endpoint", manifest.get("fixed_contract", {}).get("primary_endpoint"))
    artifact_rows = _report_artifact_rows(output_dir)
    lines = [
        "# R11_new Phase 1A 运行报告",
        "",
        "> 本报告仅由本次运行的机器可读 summary、manifest 与落盘 artifact 复算生成。",
        "",
        "## 运行身份",
        "",
        f"- 模式：`{_markdown_scalar(mode)}`",
        f"- Git commit：`{_markdown_scalar(manifest.get('git_commit', 'unknown'))}`",
        f"- Target index：`{_markdown_scalar(manifest.get('target_index', summary.get('target_index', 'unknown')))}`",
        f"- Target segment：`{_markdown_scalar(target_id)}`",
        f"- 固定 primary endpoint：`{_markdown_scalar(endpoint)}`",
        "- 训练对象：仅 `x_T_fp32`；DreamLite U-Net、condition encoder、VAE 与 Reader 全部冻结。",
        "",
        "## 三层结论",
        "",
        "| 层级 | 结果 | 可解释范围 |",
        "| --- | --- | --- |",
        (
            "| 工程通过 | "
            + ("通过" if technical else "失败")
            + " | 只表示冻结、梯度、四步轨迹、receipt、checkpoint、哈希和快照契约是否成立。 |"
        ),
        f"| 诊断通过 | {_markdown_scalar(diagnostic_text)} | Phase 1A 仅测试单 query 的 Frozen-DreamLite endpoint 可达性。 |",
        "| 科学成功 | `false` | 未训练共享 writer，未测试 state-level、多 seed、held-out、因果控制或 rollout。 |",
        "",
        "不得把 train loss 下降、非零梯度、preflight 通过或单 target reachability 表述为 Picture Memory 训练成功。",
        "",
        "## 固定执行事实",
        "",
    ]
    if mode == "technical-preflight":
        lines.extend(
            [
                f"- backward 次数：`{_markdown_scalar(summary.get('backward_calls'))}`",
                f"- optimizer steps：`{_markdown_scalar(summary.get('optimizer_steps'))}`",
                f"- DreamLite denoiser steps：`{_markdown_scalar(summary.get('dreamlite_denoising_steps'))}`",
                f"- effective sigmas：`{_markdown_scalar(summary.get('effective_sigmas'))}`",
                "- Query-level reachability：未评估。",
            ]
        )
    else:
        statistics = summary.get("target_statistics", {})
        lines.extend(
            [
                f"- optimizer receipts：`{_markdown_scalar(summary.get('optimizer_steps'))}`",
                f"- technical gate：`{str(technical).lower()}`",
                "- query-level reachability gate："
                + (f"`{str(reachability).lower()}`" if type(reachability) is bool else "未评估（技术门失败）"),
                f"- endpoint relative CE change：`{_markdown_scalar(statistics.get('relative_change'))}`",
                f"- improved reverse-cyclic views：`{_markdown_scalar(statistics.get('improved_choice_views'))}/4`",
                f"- accuracy delta：`{_markdown_scalar(statistics.get('accuracy_delta'))}`",
                "- formal success gate：`false`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 原始 artifact 索引",
            "",
            "| 路径 | 字节数 | SHA-256 |",
            "| --- | ---: | --- |",
        ]
    )
    lines.extend(f"| `{_markdown_scalar(row['path'])}` | {row['bytes']} | `{row['sha256']}` |" for row in artifact_rows)
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "canonical R11 的 direct VAE-latent 8/8 不能替代本实验。"
            "本轮即使 query-level gate 通过，也只证明锁定协议下存在可读的完整 Frozen-DreamLite endpoint；"
            "`formal_success_gate` 始终为 `false`。",
            "",
        ]
    )
    destination = output_dir / "REPORT.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)


def _write_failure_report(*, args: argparse.Namespace, error: Exception) -> None:
    manifest_path = args.output_dir / "manifest.json"
    manifest: Mapping[str, Any] = {}
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            manifest = value
    rows = _report_artifact_rows(args.output_dir)
    lines = [
        "# R11_new Phase 1A 失败报告",
        "",
        "## 三层结论",
        "",
        "| 层级 | 结果 |",
        "| --- | --- |",
        "| 工程通过 | 失败 |",
        "| 诊断通过 | 不可判定；技术失败不产生 reachability 结论。 |",
        "| 科学成功 | `false` |",
        "",
        f"- 模式：`{_markdown_scalar(args.mode)}`",
        f"- Target index：`{_markdown_scalar(args.target_index)}`",
        f"- Target segment：`{_markdown_scalar(manifest.get('target_segment_id', 'unknown'))}`",
        f"- Git commit：`{_markdown_scalar(manifest.get('git_commit', 'unknown'))}`",
        f"- 错误：`{_markdown_scalar(error)}`",
        "- 科学结论：无。必须修复技术问题后使用 fresh root 重跑原协议。",
        "",
        "## 已落盘 artifact 索引",
        "",
        "| 路径 | 字节数 | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    lines.extend(f"| `{_markdown_scalar(row['path'])}` | {row['bytes']} | `{row['sha256']}` |" for row in rows)
    lines.append("")
    destination = args.output_dir / "REPORT.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    determinism = r8.configure_strict_cuda_determinism(SEED) if args.strict_determinism else None
    r8.set_all_seeds(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_environment(args.output_dir / "environment.txt")
    _atomic_json(args.output_dir / "runtime.json", _runtime_versions())
    snapshot_bindings = _snapshot_bindings(args)
    _atomic_json(
        args.output_dir / "model_snapshot_verification_start.json",
        {
            "schema": "vision_memory.r11-new-phase1a-model-snapshot-start.v1",
            "bindings": snapshot_bindings,
        },
    )
    processor, _pipe, reader, oracle, source_latents, context = _load_runtime(args)
    manifest = _manifest(
        args=args,
        data=context["data"],
        selected=context["selected"],
        target=context["target"],
        source_rgb=context["source_rgb"],
        source_latents=source_latents,
        initial_x_t=oracle.initial_x_T_fp32,
        condition_record=context["condition_record"],
        snapshot_bindings=snapshot_bindings,
        determinism=determinism,
    )
    _atomic_json(args.output_dir / "manifest.json", manifest)
    train_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=context["reader_device"],
        require_grad=True,
        deterministic_ce=args.strict_determinism,
    )
    optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if args.mode == "technical-preflight":
        summary = _preflight(
            args=args,
            oracle=oracle,
            reader=reader,
            reader_fn=train_reader,
            target=context["target"],
            optimizer=optimizer,
            manifest=manifest,
            snapshot_bindings=snapshot_bindings,
        )
    else:
        eval_reader = r8.choice_reader_callable(
            reader=reader,
            processor=processor,
            reader_device=context["reader_device"],
            require_grad=False,
            deterministic_ce=args.strict_determinism,
        )
        summary = _formal(
            args=args,
            oracle=oracle,
            reader=reader,
            train_reader=train_reader,
            eval_reader=eval_reader,
            target=context["target"],
            source_latents=source_latents,
            optimizer=optimizer,
            manifest=manifest,
            snapshot_bindings=snapshot_bindings,
        )
    summary_path = args.output_dir / "r11_new_phase1a_summary.json"
    _atomic_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    owns_output = False
    try:
        _claim_output_dir(args.output_dir)
        owns_output = True
        _validate_args(args)
        summary = _run(args)
        technical_gate = bool(summary.get("passed", summary.get("gates", {}).get("technical_gate", False)))
        _atomic_json(
            args.output_dir / "terminal.json",
            {
                "schema": "vision_memory.r11-new-phase1a-terminal.v1",
                "status": "succeeded" if technical_gate else "failed_technical",
                "mode": args.mode,
                "technical_gate": technical_gate,
                "diagnostic_evaluated": bool(
                    technical_gate
                    and type(summary.get("gates", {}).get("phase1a_query_level_reachability_gate")) is bool
                ),
                "formal_success_gate": False,
            },
        )
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        _write_report(output_dir=args.output_dir, summary=summary, manifest=manifest)
        _atomic_json(args.output_dir / "artifact_inventory.json", _artifact_inventory(args.output_dir))
    except Exception as exc:
        if owns_output:
            try:
                _atomic_json(
                    args.output_dir / "terminal.json",
                    {
                        "schema": "vision_memory.r11-new-phase1a-terminal.v1",
                        "status": "failed",
                        "mode": args.mode,
                        "target_index": args.target_index,
                        "error": str(exc),
                        "formal_success_gate": False,
                    },
                )
                _write_failure_report(args=args, error=exc)
                _atomic_json(
                    args.output_dir / "artifact_inventory.json",
                    _artifact_inventory(args.output_dir),
                )
            except Exception as reporting_exc:
                try:
                    (args.output_dir / "REPORT.md").write_text(
                        "# R11_new Phase 1A 失败报告\n\n"
                        "- 工程通过：失败\n"
                        "- 诊断通过：不可判定\n"
                        "- 科学成功：`false`\n"
                        f"- 原始错误：`{_markdown_scalar(exc)}`\n"
                        f"- 报告错误：`{_markdown_scalar(reporting_exc)}`\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "milestone": "r11_new_phase1a_completed",
                "mode": args.mode,
                "target_index": args.target_index,
                "technical_gate": technical_gate,
                "formal_success_gate": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if technical_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
