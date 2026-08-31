"""R12 shared event-to-VAE-latent writer with sealed held-out F1 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from scripts.train.latent_r11_vae_oracle import encode_model_latent  # noqa: E402
from vision_memory.data import CYCLIC4  # noqa: E402
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training.checkpoint import save_training_checkpoint  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
)
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_BASIS_COUNT,
    R12_BASIS_OUTPUT_NORM,
    R12_CHECKPOINT_STEPS,
    R12_COEFFICIENT_L2_PENALTY,
    R12_DEV_FINAL_COUNT,
    R12_DEV_SELECT_COUNT,
    R12_EPOCHS,
    R12_GRADIENT_ACCUMULATION,
    R12_LATENT_RMS_PENALTY,
    R12_LATENT_RMS_SOFT_LIMIT,
    R12_MICRO_STEPS,
    R12_OPTIMIZER_STEPS,
    R12_PROTOCOL,
    R12_SELECTION_SEED,
    R12_TRAIN_AUDIT_COUNT,
    R12_TRAIN_SEGMENT_COUNT,
    R12_WEIGHT_DECAY,
    R12_WRITER_LEARNING_RATE,
    build_training_schedule,
    conditioned_target_gate,
    conditioned_target_statistics,
    donor_derangement,
    select_balanced_train_f1,
    select_entity_disjoint_dev_f1,
    selection_audit,
)


IMPLEMENTATION_REVISION = "shared-attention-48basis-r11-seeded-v1"
MANIFEST_SCHEMA = "vision_memory.r12-shared-event-latent-writer-manifest.v1"
MICRO_SCHEMA = "vision_memory.r12-shared-event-latent-writer-micro.v1"
OPTIMIZER_SCHEMA = "vision_memory.r12-shared-event-latent-writer-optimizer.v1"
SUMMARY_SCHEMA = "vision_memory.r12-shared-event-latent-writer-summary.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r12-shared-event-latent-writer-technical-gate.v1"
EMBEDDING_AUDIT_SCHEMA = "vision_memory.r12-event-embedding-cache.v1"
R11_BASIS_AUDIT_SCHEMA = "vision_memory.r12-r11-basis-sources.v1"
SUITE_PREFIX = "r12_f1_shared_writer"
CONFIG_PATH = ROOT / "configs" / "experiments" / "r12_shared_event_latent_writer.json"
INITIAL_LATENT_SHA256 = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"
ARMS = ("conditioned", "constant-control")
REPRESENTATIVE_PER_SPLIT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--r11-root", type=Path, required=True)
    parser.add_argument("--r11-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.seed != 0:
        raise ValueError("R12 fixes seed=0.")
    args.adapter_seed = 0
    args.pairing_seed = 0
    args.split_seed = 20260730
    args.schedule_seed = R12_SELECTION_SEED
    args.selected_step_count = 0
    args.gradient_mode = "full"
    args.resolution = 1024
    args.lora_rank = 4
    args.bootstrap_iterations = 10_000
    return args


def _sha256(path: Path) -> str:
    return r5.sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R12 expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _save_image(path: Path, image: Tensor) -> None:
    value = image.detach().float().cpu()
    if tuple(value.shape) != (1, 3, 1024, 1024):
        raise ValueError(f"R12 image artifact has invalid shape: {tuple(value.shape)}")
    array = (
        value[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _validate_args(args: argparse.Namespace) -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise ValueError(f"R12 preregistration config is missing: {CONFIG_PATH}")
    for name in ("train", "dev", "r11_comparison"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R12 {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader", "r11_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R12 {name} path is not a directory: {getattr(args, name)}")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise ValueError("R12 writer/VAE and Reader must use distinct devices.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("R12 refuses a non-empty output directory.")
    dirty = bool(r8.git_value("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise ValueError("R12 refuses a dirty source tree unless --allow-dirty is explicit.")
    config = _load_json(CONFIG_PATH)
    if (
        config.get("schema") != "vision_memory.r12-shared-event-latent-writer-config.v1"
        or config.get("status")
        != "preregistered_after_complete_r11_before_any_r12_model_outcome"
    ):
        raise ValueError("R12 machine-readable preregistration is invalid.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    expected_data = {
        "train": config["fixed_data"]["train_sha256"],
        "dev": config["fixed_data"]["dev_sha256"],
    }
    if observed_data != expected_data:
        raise ValueError(f"R12 fixed data hash drift: {observed_data}")
    comparison_hash = _sha256(args.r11_comparison)
    comparison = _load_json(args.r11_comparison)
    activation = config["activation"]
    if (
        comparison_hash != activation["required_parent_sha256"]
        or comparison.get("schema") != activation["required_parent_schema"]
        or comparison.get("status") != "completed"
        or comparison.get("formal_success_claim") is not False
        or comparison.get("target_pass_count") != activation["required_parent_target_pass_count"]
        or comparison.get("decision") != activation["required_parent_decision"]
        or comparison.get("source_training_git_commit")
        != activation["source_training_git_commit"]
        or comparison.get("aggregation_git_commit")
        != activation["source_aggregation_git_commit"]
    ):
        raise ValueError("R12 activation requires the exact completed R11 8/8 branch.")
    return {
        "config": config,
        "config_sha256": _sha256(CONFIG_PATH),
        "data_sha256": observed_data,
        "r11_comparison_sha256": comparison_hash,
        "git_commit": r8.git_value("rev-parse", "HEAD"),
        "git_dirty": dirty,
    }


def _tensor_from_checkpoint(payload: Mapping[str, Any]) -> Tensor:
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping) or set(state) != {"latent_fp32"}:
        raise ValueError("R12 R11 step0 checkpoint has an invalid trainable state.")
    latent = state["latent_fp32"]
    if not isinstance(latent, Tensor):
        raise TypeError("R12 R11 step0 latent is not a tensor.")
    return latent.detach().float().cpu()


def _orthogonal_random_bases(first: Sequence[Tensor], *, count: int, seed: int) -> list[Tensor]:
    flattened = [value.detach().float().reshape(-1).cpu() for value in first]
    orthonormal_span: list[Tensor] = []
    for value in flattened:
        candidate = value.clone()
        for basis in orthonormal_span:
            candidate -= torch.dot(candidate, basis) * basis
        norm = candidate.double().norm()
        if not math.isfinite(float(norm)) or float(norm) <= 1e-8:
            raise ValueError("R12 R11 basis deltas are linearly degenerate.")
        orthonormal_span.append(candidate / norm.float())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_bases: list[Tensor] = []
    while len(flattened) + len(random_bases) < count:
        candidate = torch.randn(flattened[0].shape, generator=generator, dtype=torch.float32)
        for _pass in range(2):
            for basis in orthonormal_span:
                candidate -= torch.dot(candidate, basis) * basis
        norm = candidate.double().norm()
        if not math.isfinite(float(norm)) or float(norm) <= 1e-8:
            continue
        candidate /= norm.float()
        random_bases.append(candidate)
        orthonormal_span.append(candidate)
    return random_bases


def _load_r11_basis(
    *, r11_root: Path, config: Mapping[str, Any]
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    source_records = config.get("r11_basis_sources")
    if not isinstance(source_records, list) or len(source_records) != 8:
        raise ValueError("R12 preregistration must bind exactly eight R11 basis sources.")
    initial_latents: list[Tensor] = []
    deltas: list[Tensor] = []
    audit_records = []
    for index, expected in enumerate(source_records):
        if not isinstance(expected, Mapping) or expected.get("target_index") != index:
            raise ValueError("R12 R11 basis source ordering drifted.")
        run = r11_root / f"target-{index:02d}" / "run"
        paths = {
            "endpoint_raw": run / "endpoint_raw.pt",
            "step0": run / "checkpoints" / "step-000.pt",
            "manifest": run / "manifest.json",
        }
        observed_hashes = {name: _sha256(path) for name, path in paths.items()}
        for name, observed in observed_hashes.items():
            if observed != expected[f"{name}_sha256"]:
                raise ValueError(f"R12 R11 basis source hash drift: target={index} {name}")
        manifest = _load_json(paths["manifest"])
        if (
            manifest.get("schema") != "vision_memory.r11-vae-latent-oracle-manifest.v1"
            or manifest.get("target_index") != index
            or manifest.get("target_segment_id") != R10_TARGET_IDS[index]
            or manifest.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        ):
            raise ValueError(f"R12 R11 basis manifest drifted for target {index}.")
        step0_payload = torch.load(paths["step0"], map_location="cpu", weights_only=False)
        endpoint_payload = torch.load(paths["endpoint_raw"], map_location="cpu", weights_only=False)
        initial = _tensor_from_checkpoint(step0_payload)
        endpoint = endpoint_payload.get("latent_fp32")
        if not isinstance(endpoint, Tensor):
            raise TypeError("R12 R11 endpoint latent is missing.")
        endpoint = endpoint.detach().float().cpu()
        if tuple(initial.shape) != (1, 4, 128, 128) or tuple(endpoint.shape) != tuple(initial.shape):
            raise ValueError("R12 R11 basis latent shape drifted.")
        delta = endpoint - initial
        norm = delta.double().norm()
        if not torch.isfinite(delta).all() or not math.isfinite(float(norm)) or float(norm) <= 0.0:
            raise ValueError("R12 R11 basis delta is invalid.")
        initial_latents.append(initial)
        deltas.append(delta)
        audit_records.append(
            {
                "target_index": index,
                "target_segment_id": R10_TARGET_IDS[index],
                "hashes": observed_hashes,
                "delta_norm": float(norm),
                "delta_rms": float(delta.square().mean().sqrt()),
            }
        )
    if any(not torch.equal(initial_latents[0], value) for value in initial_latents[1:]):
        raise ValueError("R12 R11 step0 initial latents differ across targets.")
    initial = initial_latents[0]
    if canonical_tensor_sha256(initial) != INITIAL_LATENT_SHA256:
        raise ValueError("R12 R11 initial latent hash drifted.")
    normalized = [delta / delta.double().norm().float() for delta in deltas]
    random_bases = _orthogonal_random_bases(normalized, count=R12_BASIS_COUNT, seed=0)
    all_bases = normalized + random_bases
    basis = torch.stack([value.reshape(4, 128, 128) for value in all_bases])
    audit = {
        "schema": R11_BASIS_AUDIT_SCHEMA,
        "passed": True,
        "source_records": audit_records,
        "initial_latent_sha256": canonical_tensor_sha256(initial),
        "basis_count": len(all_bases),
        "r11_seeded_basis_count": len(normalized),
        "random_orthogonal_basis_count": len(random_bases),
        "basis_initialization_sha256": canonical_tensor_sha256(basis),
        "basis_shape": list(basis.shape),
    }
    return initial, basis, audit


class SharedEventLatentWriter(nn.Module):
    """One shared attention-pooled coefficient map over a trainable latent dictionary."""

    def __init__(self, *, initial_latent: Tensor, initial_basis: Tensor, hidden_size: int = 2048) -> None:
        super().__init__()
        if tuple(initial_latent.shape) != (1, 4, 128, 128):
            raise ValueError("R12 writer initial latent must have shape [1,4,128,128].")
        if tuple(initial_basis.shape) != (R12_BASIS_COUNT, 4, 128, 128):
            raise ValueError("R12 writer basis shape drifted.")
        self.register_buffer("initial_latent_fp32", initial_latent.detach().float().clone())
        self.basis_raw = nn.Parameter(initial_basis.detach().float().clone())
        self.token_norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.attention_query = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.coefficient_mlp = nn.Sequential(
            nn.Linear(hidden_size, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, R12_BASIS_COUNT, dtype=torch.float32),
        )
        final = self.coefficient_mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self, token_states: Tensor, attention_mask: Tensor, *, conditioned: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if token_states.ndim != 3 or token_states.shape[0] != 1 or token_states.shape[2] != 2048:
            raise ValueError(f"R12 token states have invalid shape: {tuple(token_states.shape)}")
        if attention_mask.shape != token_states.shape[:2]:
            raise ValueError("R12 token attention mask shape mismatch.")
        states = self.token_norm(token_states.float())
        if not conditioned:
            states = torch.zeros_like(states)
        scores = torch.einsum("bld,d->bl", states, self.attention_query) / math.sqrt(states.shape[-1])
        scores = scores.masked_fill(attention_mask == 0, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bl,bld->bd", weights, states)
        raw_coefficients = self.coefficient_mlp(pooled)
        coefficients = 2.0 * torch.tanh(raw_coefficients / 2.0)
        flat_basis = self.basis_raw.flatten(1)
        unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
        delta_flat = R12_BASIS_OUTPUT_NORM * torch.matmul(coefficients, unit_basis)
        delta = delta_flat.reshape(1, 4, 128, 128)
        latent = self.initial_latent_fp32 + delta
        return latent, {
            "coefficients": coefficients,
            "attention_weights": weights,
            "delta": delta,
            "delta_rms": delta.square().mean().sqrt(),
            "basis_norms": flat_basis.double().norm(dim=1).float(),
        }


def _embedding_cache(
    *,
    pipe: Any,
    segments: Sequence[r5.R5Segment],
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    unique = {segment.segment_id: segment for segment in segments}
    ordered = tuple(unique[key] for key in sorted(unique))
    cache: dict[str, Tensor] = {}
    records = []
    batch_size = 8
    with torch.no_grad():
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            prompts = [segment.events[0].event_text for segment in batch]
            embeddings, masks = pipe.encode_prompt(
                mode="generate",
                prompts=prompts,
                device=device,
                dtype=dtype,
            )
            for index, segment in enumerate(batch):
                length = int(masks[index].sum())
                if length <= 0:
                    raise RuntimeError("R12 event encoder produced an empty sequence.")
                value = embeddings[index, :length].detach().float().cpu().contiguous()
                if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
                    raise RuntimeError("R12 event encoder produced invalid token states.")
                cache[segment.segment_id] = value
                records.append(
                    {
                        "segment_id": segment.segment_id,
                        "event_text": segment.events[0].event_text,
                        "event_text_sha256": hashlib.sha256(
                            segment.events[0].event_text.encode()
                        ).hexdigest(),
                        "shape": list(value.shape),
                        "tensor_sha256": canonical_tensor_sha256(value),
                    }
                )
    cache_path = output_dir / "event_embedding_cache.pt"
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save({"schema": EMBEDDING_AUDIT_SCHEMA, "token_states": cache}, temporary)
    os.replace(temporary, cache_path)
    audit = {
        "schema": EMBEDDING_AUDIT_SCHEMA,
        "passed": len(cache) == len(ordered),
        "encoder_mode": "generate",
        "input_fields": ["event_text"],
        "forbidden_fields": ["query", "choices", "target_index", "segment_id_as_model_input"],
        "cache_batch_size": batch_size,
        "cache_dtype": "torch.float32",
        "segment_count": len(cache),
        "records": records,
        "records_sha256": r5.canonical_sha256(records),
        "cache_file": str(cache_path.resolve()),
        "cache_file_sha256": _sha256(cache_path),
    }
    _write_json(output_dir / "event_embedding_audit.json", audit)
    return cache, audit


def _tokens(cache: Mapping[str, Tensor], segment_id: str, device: torch.device) -> tuple[Tensor, Tensor]:
    value = cache[segment_id].to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(value.shape[:2], device=device, dtype=torch.long)
    return value, mask


def _decode(vae: nn.Module, latent: Tensor, compute_dtype: torch.dtype) -> Tensor:
    return decode_model_latents_unit_interval(vae, latent.to(dtype=compute_dtype), clamp=True)


def _item(segment: r5.R5Segment, reset_image: Tensor) -> r5.EvalItem:
    state = reset_image.detach().cpu()
    return r5.EvalItem(
        item_id=segment.segment_id,
        pair_unit=segment.segment_id,
        episode_id=segment.query_source_episode_id,
        query_id=f"{segment.query_source_episode_id}:{segment.query_turn_index}",
        query=segment.query,
        normal_state=state,
        temporal_state=state,
        family=segment.family,
        target_event_kind=segment.target_event_kind,
        query_gap=segment.query_gap,
        updater_count=segment.updater_count,
        cross_slot_interference=segment.cross_slot_interference,
        stale_target_text=segment.stale_target_text,
    )


def _reader_output(
    reader_fn: Any,
    *,
    image: Tensor,
    segment: r5.R5Segment,
    permutation: tuple[int, int, int, int],
) -> Any:
    ordered = tuple(segment.query.choices[index] for index in permutation)
    target_index = permutation.index(segment.query.target_index)
    return reader_fn(
        image,
        r5.format_mcq_query(segment.query.text, ordered),
        ordered,
        target_index,
    )


def _evaluation_rows(
    *,
    writer: SharedEventLatentWriter,
    vae: nn.Module,
    cache: Mapping[str, Tensor],
    segments: Sequence[r5.R5Segment],
    donor_map: Mapping[str, str],
    reader_fn: Any,
    reset_image: Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
    conditioned: bool,
    suite: str,
    endpoint: str,
    image_root: Path,
) -> list[dict[str, Any]]:
    by_id = {segment.segment_id: segment for segment in segments}
    rows: list[dict[str, Any]] = []
    writer.eval()
    with torch.no_grad():
        for segment in segments:
            donor = by_id[donor_map[segment.segment_id]]
            own_tokens, own_mask = _tokens(cache, segment.segment_id, device)
            donor_tokens, donor_mask = _tokens(cache, donor.segment_id, device)
            own_latent, _own_diagnostics = writer(
                own_tokens, own_mask, conditioned=conditioned
            )
            donor_latent, _donor_diagnostics = writer(
                donor_tokens, donor_mask, conditioned=conditioned
            )
            own_image = _decode(vae, own_latent, compute_dtype)
            donor_image = _decode(vae, donor_latent, compute_dtype)
            _save_image(image_root / "normal" / f"{segment.segment_id}.png", own_image)
            _save_image(image_root / "donor" / f"{segment.segment_id}.png", donor_image)
            item = _item(segment, reset_image)
            for view_index, permutation in enumerate(r5.REVERSE_CYCLIC4):
                reset_output = _reader_output(
                    reader_fn,
                    image=reset_image,
                    segment=segment,
                    permutation=permutation,
                )
                for condition in ("normal", "reset", "donor"):
                    rows.append(
                        r5._choice_row(
                            reader_output=reset_output,
                            item=item,
                            checkpoint_label="m0",
                            suite=suite,
                            condition=condition,
                            permutation=permutation,
                            view_index=view_index,
                            donor_item_id=(donor.segment_id if condition == "donor" else None),
                        )
                    )
                normal_output = _reader_output(
                    reader_fn,
                    image=own_image,
                    segment=segment,
                    permutation=permutation,
                )
                donor_output = _reader_output(
                    reader_fn,
                    image=donor_image,
                    segment=segment,
                    permutation=permutation,
                )
                for condition, output, donor_id in (
                    ("normal", normal_output, None),
                    ("reset", reset_output, None),
                    ("donor", donor_output, donor.segment_id),
                ):
                    rows.append(
                        r5._choice_row(
                            reader_output=output,
                            item=item,
                            checkpoint_label=endpoint,
                            suite=suite,
                            condition=condition,
                            permutation=permutation,
                            view_index=view_index,
                            donor_item_id=donor_id,
                        )
                    )
    writer.train()
    return rows


def _checkpoint_diagnostics(
    *,
    writer: SharedEventLatentWriter,
    vae: nn.Module,
    cache: Mapping[str, Tensor],
    representatives: Mapping[str, Sequence[r5.R5Segment]],
    conditioned: bool,
    device: torch.device,
    compute_dtype: torch.dtype,
    image_root: Path,
    step: int,
) -> dict[str, Any]:
    rows = []
    writer.eval()
    with torch.no_grad():
        for split, segments in representatives.items():
            for segment in segments:
                token_states, mask = _tokens(cache, segment.segment_id, device)
                latent, diagnostics = writer(
                    token_states, mask, conditioned=conditioned
                )
                image = _decode(vae, latent, compute_dtype)
                _save_image(
                    image_root / f"step-{step:04d}" / split / f"{segment.segment_id}.png",
                    image,
                )
                coefficients = diagnostics["coefficients"]
                rows.append(
                    {
                        "split": split,
                        "segment_id": segment.segment_id,
                        "coefficient_norm": float(coefficients.double().norm()),
                        "coefficient_max_abs": float(coefficients.abs().max()),
                        "latent_delta_rms": float(diagnostics["delta_rms"]),
                        "image_saturation_fraction": float(
                            ((image <= 0.0) | (image >= 1.0)).double().mean()
                        ),
                    }
                )
    writer.train()
    return {
        "step": step,
        "rows": rows,
        "basis_norm_min": float(writer.basis_raw.detach().flatten(1).double().norm(dim=1).min()),
        "basis_norm_max": float(writer.basis_raw.detach().flatten(1).double().norm(dim=1).max()),
    }


def _save_checkpoint(
    *,
    step: int,
    micro_cursor: int,
    writer: SharedEventLatentWriter,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    output_dir: Path,
    diagnostics: Mapping[str, Any],
) -> None:
    save_training_checkpoint(
        output_dir / "checkpoints" / f"step-{step:04d}.pt",
        trainable_module=writer,
        optimizer=optimizer,
        epoch=step * R12_GRADIENT_ACCUMULATION // R12_TRAIN_SEGMENT_COUNT,
        episode_cursor=micro_cursor,
        optimizer_step=step,
        manifest=manifest,
        trainer_state={
            "schema": "vision_memory.r12-shared-writer-checkpoint-state.v1",
            "next_micro_index": micro_cursor,
            "next_optimizer_step": step,
        },
    )
    _write_json(output_dir / "checkpoint_diagnostics" / f"step-{step:04d}.json", diagnostics)


def _gradient_statistics(writer: SharedEventLatentWriter) -> dict[str, Any]:
    squared = 0.0
    nonzero = 0
    count = 0
    active = []
    for name, parameter in writer.named_parameters():
        gradient = parameter.grad
        count += parameter.numel()
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"R12 non-finite gradient in {name}.")
        active.append(name)
        squared += float(gradient.detach().double().square().sum())
        nonzero += int((gradient != 0).sum())
    norm = math.sqrt(squared)
    if not math.isfinite(norm) or norm <= 0.0 or nonzero <= 0:
        raise RuntimeError("R12 produced a non-finite or zero aggregate gradient.")
    return {
        "gradient_norm": norm,
        "gradient_nonzero_fraction": nonzero / count,
        "active_parameter_names": active,
    }


def _manifest(
    *,
    args: argparse.Namespace,
    validation: Mapping[str, Any],
    selections: Mapping[str, Sequence[r5.R5Segment]],
    schedule: Sequence[Any],
    initial_latent: Tensor,
    basis_audit: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schedule_receipts = [unit.receipt() for unit in schedule]
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": R12_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "arm": args.arm,
        "git_commit": validation["git_commit"],
        "git_dirty": validation["git_dirty"],
        "config_sha256": validation["config_sha256"],
        "data_sha256": validation["data_sha256"],
        "r11_comparison_sha256": validation["r11_comparison_sha256"],
        "selections": {name: selection_audit(values) for name, values in selections.items()},
        "schedule": {
            "micro_steps": len(schedule),
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "gradient_accumulation": R12_GRADIENT_ACCUMULATION,
            "receipt_sha256": r5.canonical_sha256(schedule_receipts),
        },
        "initial_latent_sha256": canonical_tensor_sha256(initial_latent),
        "r11_basis_audit": dict(basis_audit),
        "event_embedding_audit_sha256": r5.canonical_sha256(embedding_audit),
        "model_snapshot_payloads_start": dict(snapshot_bindings),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "information_boundary": {
            "writer_input": ["event_text token states"],
            "forbidden_writer_inputs": [
                "query",
                "choices",
                "target_index",
                "answer text extracted from QuerySpec",
                "segment ID",
                "entity embedding",
                "per-item latent",
            ],
            "dev_gradient_steps": 0,
        },
        "fixed_contract": {
            "event_encoder": "frozen DreamLite internal Qwen3-VL-2B generate mode",
            "writer": "shared attention pooling plus 2048-512-48 coefficients",
            "basis_count": R12_BASIS_COUNT,
            "basis_output_norm": R12_BASIS_OUTPUT_NORM,
            "dreamlite_unet_executed": False,
            "semantic_edit_prompt_used": False,
            "optimizer": "AdamW",
            "learning_rate": R12_WRITER_LEARNING_RATE,
            "network_weight_decay": R12_WEIGHT_DECAY,
            "basis_weight_decay": 0.0,
            "epochs": R12_EPOCHS,
            "micro_steps": R12_MICRO_STEPS,
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "checkpoint_steps": list(R12_CHECKPOINT_STEPS),
            "gradient_clipping": None,
            "primary_endpoint": f"shared_step{R12_OPTIMIZER_STEPS}",
            "best_checkpoint_selection_forbidden": True,
        },
        "diagnostic_only_not_formal_success": True,
    }


def _technical_gate(
    *,
    args: argparse.Namespace,
    writer: SharedEventLatentWriter,
    pipe: Any,
    reader: nn.Module,
    micro_rows: Sequence[Mapping[str, Any]],
    optimizer_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Any],
    selections: Mapping[str, Sequence[r5.R5Segment]],
    embedding_audit: Mapping[str, Any],
    basis_audit: Mapping[str, Any],
    checkpoint_steps: set[int],
    diagnostic_steps: set[int],
    evaluation_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    snapshots_unchanged: bool,
) -> dict[str, Any]:
    expected_views = R12_EPOCHS // 4
    view_counts = Counter(
        (row["segment_id"], int(row["forward_cyclic_training_view"]))
        for row in micro_rows
    )
    exact_views = all(
        view_counts[(segment.segment_id, view)] == expected_views
        for segment in selections["train"]
        for view in range(4)
    )
    expected_train_ids = {segment.segment_id for segment in selections["train"]}
    dev_ids = {
        segment.segment_id
        for name in ("dev_select", "dev_final")
        for segment in selections[name]
    }
    trainable_names = [name for name, parameter in writer.named_parameters() if parameter.requires_grad]
    evaluation_complete = True
    expected_counts = {
        "train_audit": R12_TRAIN_AUDIT_COUNT,
        "dev_select": R12_DEV_SELECT_COUNT,
        "dev_final": R12_DEV_FINAL_COUNT,
    }
    for split, count in expected_counts.items():
        rows = evaluation_rows.get(split, ())
        if len(rows) != count * 2 * 3 * 4:
            evaluation_complete = False
            continue
        cells = Counter(
            (row["pair_unit"], row["checkpoint"], row["condition"], int(row["view_index"]))
            for row in rows
        )
        if set(cells.values()) != {1}:
            evaluation_complete = False
    frozen_modules = {
        "vae": pipe.vae,
        "unet": pipe.unet,
        "event_encoder": pipe.text_encoder,
        "reader": reader,
    }
    frozen_ok = all(
        not parameter.requires_grad and parameter.grad is None
        for module in frozen_modules.values()
        for parameter in module.parameters()
    )
    passed = bool(
        len(micro_rows) == R12_MICRO_STEPS
        and [int(row["global_micro_index"]) for row in micro_rows]
        == list(range(R12_MICRO_STEPS))
        and {row["segment_id"] for row in micro_rows} == expected_train_ids
        and not ({row["segment_id"] for row in micro_rows} & dev_ids)
        and exact_views
        and len(optimizer_rows) == R12_OPTIMIZER_STEPS
        and [int(row["optimizer_step"]) for row in optimizer_rows]
        == list(range(1, R12_OPTIMIZER_STEPS + 1))
        and all(math.isfinite(float(row["gradient_norm"])) and float(row["gradient_norm"]) > 0.0 for row in optimizer_rows)
        and all(math.isfinite(float(row["gradient_nonzero_fraction"])) and float(row["gradient_nonzero_fraction"]) > 0.0 for row in optimizer_rows)
        and checkpoint_steps == set(R12_CHECKPOINT_STEPS)
        and diagnostic_steps == set(R12_CHECKPOINT_STEPS)
        and embedding_audit.get("passed") is True
        and embedding_audit.get("input_fields") == ["event_text"]
        and basis_audit.get("passed") is True
        and basis_audit.get("basis_count") == R12_BASIS_COUNT
        and all(parameter.dtype == torch.float32 for parameter in writer.parameters())
        and frozen_ok
        and snapshots_unchanged
        and evaluation_complete
        and args.arm in ARMS
        and len(schedule) == R12_MICRO_STEPS
    )
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": passed,
        "arm": args.arm,
        "micro_records": len(micro_rows),
        "optimizer_records": len(optimizer_rows),
        "exact_per_segment_training_views": exact_views,
        "expected_exposures_per_view": expected_views,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "checkpoint_diagnostic_steps_observed": sorted(diagnostic_steps),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in writer.parameters()),
        "frozen_modules_clean": frozen_ok,
        "snapshots_unchanged": snapshots_unchanged,
        "evaluation_complete": evaluation_complete,
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in optimizer_rows),
        "minimum_gradient_nonzero_fraction": min(
            float(row["gradient_nonzero_fraction"]) for row in optimizer_rows
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R12 requires CUDA.")
    validation = _validate_args(args)
    determinism = (
        r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    )
    r8.set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    config = validation["config"]

    data = r5._load_data(args, optimizer_steps=0)
    train_selected, train_audit = select_balanced_train_f1(data.train_pools["F1"])
    dev_records = tuple(r5.read_episode_jsonl(args.dev))
    dev_pools = r5.build_r5_family_pools(dev_records, pairing_seed=args.split_seed)
    dev_select, dev_final = select_entity_disjoint_dev_f1(dev_pools["F1"])
    selections = {
        "train": train_selected,
        "train_audit": train_audit,
        "dev_select": dev_select,
        "dev_final": dev_final,
    }
    schedule = build_training_schedule(train_selected)
    selection_payload = {name: selection_audit(values) for name, values in selections.items()}
    _write_json(args.output_dir / "selection_audit.json", selection_payload)
    _write_json(
        args.output_dir / "schedule_audit.json",
        {
            "micro_steps": len(schedule),
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "receipts_sha256": r5.canonical_sha256([unit.receipt() for unit in schedule]),
            "receipts": [unit.receipt() for unit in schedule],
        },
    )

    initial_r11, initial_basis, basis_audit = _load_r11_basis(
        r11_root=args.r11_root, config=config
    )
    _write_json(args.output_dir / "r11_basis_audit.json", basis_audit)

    writer_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    writer_dtype = r5.compute_dtype(writer_device)
    reader_dtype = r5.compute_dtype(reader_device)
    pipe = r5._load_pipeline(args, writer_device, writer_dtype)
    pipe.unet.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    processor, reader = r5._load_reader(args, reader_device, reader_dtype)
    snapshot_bindings = {
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
    initial_rgb = r5._initial_rgb_tensor(
        resolution=1024, device=writer_device, dtype=writer_dtype
    )
    with torch.no_grad():
        recomputed_initial = encode_model_latent(pipe.vae, initial_rgb).detach().float().cpu()
    if canonical_tensor_sha256(recomputed_initial) != INITIAL_LATENT_SHA256 or not torch.equal(
        recomputed_initial, initial_r11
    ):
        raise RuntimeError("R12 recomputed blank latent differs from the exact R11 initialization.")

    embedding_segments = tuple(
        {segment.segment_id: segment for values in selections.values() for segment in values}.values()
    )
    cache, embedding_audit = _embedding_cache(
        pipe=pipe,
        segments=embedding_segments,
        device=writer_device,
        dtype=writer_dtype,
        output_dir=args.output_dir,
    )
    writer = SharedEventLatentWriter(
        initial_latent=recomputed_initial.to(writer_device),
        initial_basis=initial_basis.to(writer_device),
    ).to(writer_device)
    conditioned = args.arm == "conditioned"
    network_parameters = [
        parameter
        for name, parameter in writer.named_parameters()
        if name != "basis_raw"
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": network_parameters,
                "lr": R12_WRITER_LEARNING_RATE,
                "weight_decay": R12_WEIGHT_DECAY,
            },
            {
                "params": [writer.basis_raw],
                "lr": R12_WRITER_LEARNING_RATE,
                "weight_decay": 0.0,
            },
        ],
        lr=R12_WRITER_LEARNING_RATE,
    )
    train_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
    )
    eval_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    reset_image = _decode(pipe.vae, writer.initial_latent_fp32, writer_dtype).detach()
    _save_image(args.output_dir / "images" / "reset.png", reset_image)

    manifest = _manifest(
        args=args,
        validation=validation,
        selections=selections,
        schedule=schedule,
        initial_latent=recomputed_initial,
        basis_audit=basis_audit,
        embedding_audit=embedding_audit,
        snapshot_bindings=snapshot_bindings,
        determinism=determinism,
    )
    _write_json(args.output_dir / "manifest.json", manifest)
    representatives = {
        "train_audit": train_audit[:REPRESENTATIVE_PER_SPLIT],
        "dev_select": dev_select[:REPRESENTATIVE_PER_SPLIT],
    }
    checkpoint_diagnostics = _checkpoint_diagnostics(
        writer=writer,
        vae=pipe.vae,
        cache=cache,
        representatives=representatives,
        conditioned=conditioned,
        device=writer_device,
        compute_dtype=writer_dtype,
        image_root=args.output_dir / "images" / "trajectory",
        step=0,
    )
    _save_checkpoint(
        step=0,
        micro_cursor=0,
        writer=writer,
        optimizer=optimizer,
        manifest=manifest,
        output_dir=args.output_dir,
        diagnostics=checkpoint_diagnostics,
    )

    micro_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    micro_path = args.output_dir / "micro_metrics.jsonl"
    optimizer_path = args.output_dir / "optimizer_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    rolling_ce: list[float] = []
    for unit in schedule:
        segment = unit.segment
        token_states, mask = _tokens(cache, segment.segment_id, writer_device)
        latent, diagnostics = writer(token_states, mask, conditioned=conditioned)
        image = _decode(pipe.vae, latent, writer_dtype)
        permutation = CYCLIC4[unit.forward_cyclic_training_view]
        output = _reader_output(
            train_reader,
            image=image,
            segment=segment,
            permutation=permutation,
        )
        ce = output.loss
        if not isinstance(ce, Tensor) or ce.numel() != 1 or not torch.isfinite(ce):
            raise RuntimeError("R12 Reader returned an invalid scalar CE.")
        delta_rms = diagnostics["delta_rms"]
        coefficients = diagnostics["coefficients"]
        latent_penalty = R12_LATENT_RMS_PENALTY * torch.relu(
            delta_rms - R12_LATENT_RMS_SOFT_LIMIT
        ).square()
        coefficient_penalty = R12_COEFFICIENT_L2_PENALTY * coefficients.square().mean()
        # Reader CE lives on the Reader GPU while writer regularizers live on
        # the writer GPU. Cross-device ``to`` remains differentiable and keeps
        # the preregistered scalar objective unchanged.
        objective = (
            ce
            + latent_penalty.to(device=ce.device)
            + coefficient_penalty.to(device=ce.device)
        )
        if not torch.isfinite(objective):
            raise RuntimeError("R12 objective became non-finite.")
        (objective / R12_GRADIENT_ACCUMULATION).backward()
        row = {
            "schema": MICRO_SCHEMA,
            "global_micro_index": unit.global_micro_index,
            "epoch_zero": unit.epoch_zero,
            "micro_in_epoch": unit.micro_in_epoch,
            "optimizer_step_zero": unit.optimizer_step_zero,
            "segment_id": segment.segment_id,
            "target_value": segment.query.target,
            "forward_cyclic_training_view": unit.forward_cyclic_training_view,
            "permutation": list(permutation),
            "ce": float(ce.detach()),
            "objective": float(objective.detach()),
            "latent_delta_rms": float(delta_rms.detach()),
            "latent_penalty": float(latent_penalty.detach()),
            "coefficient_penalty": float(coefficient_penalty.detach()),
            "coefficient_norm": float(coefficients.detach().double().norm()),
            "coefficient_max_abs": float(coefficients.detach().abs().max()),
            "image_saturation_fraction": float(
                ((image.detach() <= 0.0) | (image.detach() >= 1.0)).double().mean()
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        micro_rows.append(row)
        rolling_ce.append(row["ce"])
        _append_jsonl(micro_path, (row,))

        if (unit.global_micro_index + 1) % R12_GRADIENT_ACCUMULATION:
            continue
        gradient = _gradient_statistics(writer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = unit.optimizer_step_zero + 1
        optimizer_row = {
            "schema": OPTIMIZER_SCHEMA,
            "optimizer_step": step,
            "last_global_micro_index": unit.global_micro_index,
            "epoch_zero": unit.epoch_zero,
            "learning_rate": R12_WRITER_LEARNING_RATE,
            "gradient_clipped": False,
            **gradient,
            "basis_parameter_norm": float(writer.basis_raw.detach().double().norm()),
            "attention_query_norm": float(writer.attention_query.detach().double().norm()),
            "elapsed_seconds": time.monotonic() - started,
        }
        optimizer_rows.append(optimizer_row)
        _append_jsonl(optimizer_path, (optimizer_row,))
        if step in R12_CHECKPOINT_STEPS:
            diagnostics_at_checkpoint = _checkpoint_diagnostics(
                writer=writer,
                vae=pipe.vae,
                cache=cache,
                representatives=representatives,
                conditioned=conditioned,
                device=writer_device,
                compute_dtype=writer_dtype,
                image_root=args.output_dir / "images" / "trajectory",
                step=step,
            )
            _save_checkpoint(
                step=step,
                micro_cursor=unit.global_micro_index + 1,
                writer=writer,
                optimizer=optimizer,
                manifest=manifest,
                output_dir=args.output_dir,
                diagnostics=diagnostics_at_checkpoint,
            )
        if step == 1 or step % 16 == 0 or step in R12_CHECKPOINT_STEPS:
            window = rolling_ce[-64:]
            print(
                json.dumps(
                    {
                        "milestone": "r12_optimizer_step",
                        "arm": args.arm,
                        "optimizer_step": step,
                        "mean_last64_micro_ce": sum(window) / len(window),
                        "gradient_norm": gradient["gradient_norm"],
                        "elapsed_seconds": optimizer_row["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    endpoint = f"shared_step{R12_OPTIMIZER_STEPS}"
    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    split_statistics: dict[str, list[dict[str, Any]]] = {}
    for split in ("train_audit", "dev_select", "dev_final"):
        segments = selections[split]
        suite = f"{SUITE_PREFIX}_{split}"
        donor_map = donor_derangement(segments)
        rows = _evaluation_rows(
            writer=writer,
            vae=pipe.vae,
            cache=cache,
            segments=segments,
            donor_map=donor_map,
            reader_fn=eval_reader,
            reset_image=reset_image,
            device=writer_device,
            compute_dtype=writer_dtype,
            conditioned=conditioned,
            suite=suite,
            endpoint=endpoint,
            image_root=args.output_dir / "images" / "endpoint" / split,
        )
        evaluation_rows[split] = rows
        _append_jsonl(args.output_dir / f"{split}_evaluation_rows.jsonl", rows)

    end_bindings = {
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
    _write_json(args.output_dir / "model_snapshot_verification_end.json", end_bindings)
    snapshots_unchanged = {
        key: value["snapshot_payload_sha256"] for key, value in snapshot_bindings.items()
    } == {
        key: value["snapshot_payload_sha256"] for key, value in end_bindings.items()
    }
    checkpoint_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "checkpoints").glob("step-*.pt")
    }
    diagnostic_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "checkpoint_diagnostics").glob("step-*.json")
    }
    technical = _technical_gate(
        args=args,
        writer=writer,
        pipe=pipe,
        reader=reader,
        micro_rows=micro_rows,
        optimizer_rows=optimizer_rows,
        schedule=schedule,
        selections=selections,
        embedding_audit=embedding_audit,
        basis_audit=basis_audit,
        checkpoint_steps=checkpoint_steps,
        diagnostic_steps=diagnostic_steps,
        evaluation_rows=evaluation_rows,
        snapshots_unchanged=snapshots_unchanged,
    )
    _write_json(args.output_dir / "technical_gate.json", technical)
    for split in ("train_audit", "dev_select", "dev_final"):
        suite = f"{SUITE_PREFIX}_{split}"
        statistics = []
        for segment in selections[split]:
            value = conditioned_target_statistics(
                evaluation_rows[split],
                suite=suite,
                target_segment_id=segment.segment_id,
                endpoint=endpoint,
            )
            value["target_gate"] = conditioned_target_gate(
                value, technical_gate=bool(technical["passed"])
            )
            value["target_value"] = segment.query.target
            statistics.append(value)
        split_statistics[split] = statistics
        _write_json(args.output_dir / f"{split}_statistics.json", statistics)

    pass_counts = {
        split: sum(bool(value["target_gate"]) for value in statistics)
        for split, statistics in split_statistics.items()
    }
    arm_gate = bool(
        technical["passed"]
        and pass_counts
        == {
            "train_audit": R12_TRAIN_AUDIT_COUNT,
            "dev_select": R12_DEV_SELECT_COUNT,
            "dev_final": R12_DEV_FINAL_COUNT,
        }
    )
    if arm_gate:
        decision = (
            "conditioned_shared_f1_candidate_pass_requires_constant_control"
            if conditioned
            else "constant_control_false_positive_detected"
        )
    elif pass_counts["train_audit"] == R12_TRAIN_AUDIT_COUNT:
        decision = "train_fit_without_complete_heldout_causal_generalization"
    else:
        decision = "shared_writer_did_not_fit_fixed_f1_boundary"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": R12_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "arm": args.arm,
        "git_commit": validation["git_commit"],
        "config_sha256": validation["config_sha256"],
        "optimizer_steps": len(optimizer_rows),
        "micro_steps": len(micro_rows),
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "endpoint": endpoint,
        "selection_audits": selection_payload,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "split_target_pass_counts": pass_counts,
            "arm_gate": arm_gate,
            "formal_success_gate": False,
        },
        "split_statistics": split_statistics,
        "decision": decision,
        "full_success_claim_allowed": False,
        "diagnostic_only_not_formal_success": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(args.output_dir / "r12_shared_writer_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = _run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
