"""R14 symmetric wrong-donor ranking with the fixed R13 causal evaluation.

R14 preserves the R13 mean-centered writer, exact R12 decomposition
initialization, data, schedule, endpoint policy, and causal evaluation.  Its
single scientific intervention is training-time credit assignment: every
training event is paired bidirectionally with one different-value event, and
the own image must beat the paired wrong-donor image by a bounded four-choice
cross-entropy margin.
"""

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
import torch.nn.functional as functional
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
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_BASIS_COUNT,
    R12_BASIS_OUTPUT_NORM,
    R12_CHECKPOINT_STEPS,
    R12_COEFFICIENT_L2_PENALTY,
    R12_DEV_FINAL_COUNT,
    R12_DEV_SELECT_COUNT,
    R12_GRADIENT_ACCUMULATION,
    R12_LATENT_RMS_PENALTY,
    R12_LATENT_RMS_SOFT_LIMIT,
    R12_MICRO_STEPS,
    R12_OPTIMIZER_STEPS,
    R12_SELECTION_SEED,
    R12_TRAIN_AUDIT_COUNT,
    R12_TRAIN_SEGMENT_COUNT,
    R12_WEIGHT_DECAY,
    R12_WRITER_LEARNING_RATE,
    build_training_schedule,
    donor_derangement,
    select_balanced_train_f1,
    select_entity_disjoint_dev_f1,
    selection_audit,
)
from vision_memory.training.r14_symmetric_donor import (  # noqa: E402
    R14_FRESH_DEV_FINAL_COUNT,
    R14_FRESH_DEV_FINAL_SHA256,
    R14_PAIR_SEED,
    R14_PROTOCOL,
    R14_RANKING_MARGIN,
    R14_RANKING_WEIGHT,
    centered_target_gate,
    centered_target_statistics,
    choice_target_margin,
    select_fresh_dev_final_f1,
    symmetric_donor_mapping,
    symmetric_pairing_audit,
    symmetric_ranking_loss,
)


IMPLEMENTATION_REVISION = "r13-centered-residual-symmetric-donor-ranking-v1"
MANIFEST_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-manifest.v1"
MICRO_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-micro.v1"
OPTIMIZER_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-optimizer.v1"
SUMMARY_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-summary.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-technical-gate.v1"
EMBEDDING_AUDIT_SCHEMA = "vision_memory.r14-event-embedding-cache.v1"
SOURCE_AUDIT_SCHEMA = "vision_memory.r14-r12-source-decomposition.v1"
SUITE_PREFIX = "r14_f1_symmetric_donor"
CONFIG_PATH = ROOT / "configs" / "experiments" / "r14_symmetric_donor_writer.json"
R14_PREREG_PATH = ROOT / "reports" / "r14-symmetric-donor-ranking-preregistration-20260904.md"
SOURCE_PREREG_PATH = ROOT / "reports" / "r13-source-decomposition-preregistration-20260904.json"
R13_ANALYSIS_PATH = ROOT / "reports" / "r13-centered-residual-writer-results-20260904" / "ANALYSIS.json"
R13_DELIVERY_MANIFEST_PATH = (
    ROOT / "reports" / "r13-centered-residual-writer-results-20260904" / "DELIVERY_MANIFEST.json"
)
INITIAL_LATENT_SHA256 = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"
EVALUATION_SPLITS = ("train_audit", "dev_select", "dev_replay", "dev_final")
REPRESENTATIVE_PER_SPLIT = 4
PRIMARY_ENDPOINT = f"symmetric_step{R12_OPTIMIZER_STEPS}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--r12-conditioned-root", type=Path, required=True)
    parser.add_argument("--r12-control-root", type=Path, required=True)
    parser.add_argument("--r12-comparison", type=Path, required=True)
    parser.add_argument("--r12-collapse-audit", type=Path, required=True)
    parser.add_argument("--r13-root", type=Path, required=True)
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
        raise ValueError("R14 fixes seed=0.")
    args.adapter_seed = 0
    args.pairing_seed = R14_PAIR_SEED
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


def _text_sha256_lf(path: Path) -> str:
    """Hash repository text identically on Windows and Linux checkouts."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R14 expected a JSON object: {path}")
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
    if value.ndim == 4:
        value = value[0]
    value = value.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((value * 255.0).round().astype("uint8")).save(path)


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "terminal": root / "terminal.json",
        "inventory": root / "artifact_inventory.json",
        "summary": root / "run" / "r12_shared_writer_summary.json",
        "checkpoint": root / "run" / "checkpoints" / "step-1152.pt",
        "embedding_cache": root / "run" / "event_embedding_cache.pt",
        "embedding_audit": root / "run" / "event_embedding_audit.json",
        "technical_gate": root / "run" / "technical_gate.json",
    }


def _validate_source_arm(*, root: Path, arm: str, expected: Mapping[str, str]) -> dict[str, Path]:
    paths = _source_paths(root)
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"R14 missing R12 {arm} source {name}: {path}")
    observed = {name: _sha256(paths[name]) for name in expected}
    if observed != dict(expected):
        raise ValueError(f"R14 R12 {arm} source hash drift: {observed}")
    terminal = _load_json(paths["terminal"])
    summary = _load_json(paths["summary"])
    technical = _load_json(paths["technical_gate"])
    if (
        terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("scientific_arm_gate") is not False
        or summary.get("schema") != "vision_memory.r12-shared-event-latent-writer-summary.v1"
        or summary.get("status") != "completed"
        or summary.get("arm") != arm
        or summary.get("gates", {}).get("technical_gate") is not True
        or summary.get("gates", {}).get("arm_gate") is not False
        or technical.get("passed") is not True
    ):
        raise ValueError(f"R14 requires the completed failed-science/valid-technical R12 {arm} arm.")
    return paths


def _r13_source_paths(root: Path) -> dict[str, Path]:
    return {
        "terminal": root / "terminal.json",
        "artifact_inventory": root / "artifact_inventory.json",
        "manifest": root / "run" / "manifest.json",
        "summary": root / "run" / "r13_centered_residual_summary.json",
        "technical_gate": root / "run" / "technical_gate.json",
        "checkpoint": root / "run" / "checkpoints" / "step-1152.pt",
    }


def _validate_r13_parent(*, root: Path, activation: Mapping[str, Any]) -> dict[str, Any]:
    paths = _r13_source_paths(root)
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"R14 missing R13 parent {name}: {path}")
    expected_hashes = activation["r13_source_sha256"]
    observed_hashes = {name: _sha256(paths[name]) for name in expected_hashes}
    if observed_hashes != dict(expected_hashes):
        raise ValueError(f"R14 R13 parent source hash drift: {observed_hashes}")

    terminal = _load_json(paths["terminal"])
    summary = _load_json(paths["summary"])
    technical = _load_json(paths["technical_gate"])
    expected_counts = activation["r13_split_target_pass_counts"]
    summary_gates = summary.get("gates", {})
    if (
        terminal.get("schema") != "vision_memory.r13-centered-residual-terminal.v1"
        or terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("scientific_arm_gate") is not False
        or terminal.get("formal_success_claim") is not False
        or summary.get("schema") != "vision_memory.r13-centered-residual-writer-summary.v1"
        or summary.get("status") != "completed"
        or summary.get("git_commit") != activation["r13_training_git_commit"]
        or summary_gates.get("technical_gate") is not True
        or summary_gates.get("arm_gate") is not False
        or summary_gates.get("formal_success_gate") is not False
        or summary_gates.get("split_target_pass_counts") != expected_counts
        or technical.get("schema") != "vision_memory.r13-centered-residual-writer-technical-gate.v1"
        or technical.get("passed") is not True
    ):
        raise ValueError("R14 requires the exact completed technical-pass/scientific-fail R13 parent.")

    report_paths = {
        "analysis": R13_ANALYSIS_PATH,
        "delivery_manifest": R13_DELIVERY_MANIFEST_PATH,
    }
    for name, path in report_paths.items():
        if not path.is_file():
            raise ValueError(f"R14 missing R13 delivered report {name}: {path}")
    expected_report_hashes = activation["r13_report_sha256"]
    observed_report_hashes = {name: _text_sha256_lf(report_paths[name]) for name in expected_report_hashes}
    if observed_report_hashes != dict(expected_report_hashes):
        raise ValueError(f"R14 R13 delivered report hash drift: {observed_report_hashes}")
    analysis = _load_json(R13_ANALYSIS_PATH)
    if (
        analysis.get("status") != "completed"
        or analysis.get("formal_success_claim") is not False
        or analysis.get("scientific_arm_gate") is not False
        or analysis.get("technical_gate") is not True
        or analysis.get("split_target_pass_counts") != expected_counts
        or analysis.get("git_commit") != activation["r13_training_git_commit"]
    ):
        raise ValueError("R14 R13 delivered analysis does not bind the preregistered failure outcome.")
    return {
        "paths": paths,
        "source_sha256": observed_hashes,
        "report_sha256": observed_report_hashes,
    }


def _validate_args(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R14 requires CUDA.")
    if args.dreamlite_device == args.reader_device:
        raise ValueError("R14 writer/VAE and Reader must use distinct devices.")
    for name in ("train", "dev", "r12_comparison", "r12_collapse_audit"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R14 missing file argument {name}: {getattr(args, name)}")
    for name in ("dreamlite", "reader", "r12_conditioned_root", "r12_control_root", "r13_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R14 missing directory argument {name}: {getattr(args, name)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("R14 refuses a non-empty output directory.")
    config = _load_json(CONFIG_PATH)
    if (
        config.get("schema") != "vision_memory.r14-symmetric-donor-ranking-config.v1"
        or config.get("status") != "preregistered_before_any_r14_model_outcome"
    ):
        raise ValueError("R14 configuration is not in the locked preregistered state.")
    if (
        not R14_PREREG_PATH.is_file()
        or config.get("preregistration", {}).get("path") != R14_PREREG_PATH.relative_to(ROOT).as_posix()
        or _text_sha256_lf(R14_PREREG_PATH) != config.get("preregistration", {}).get("sha256")
    ):
        raise ValueError("R14 human-readable preregistration artifact drifted.")
    head = r5.git_value("rev-parse", "HEAD")
    dirty = bool(r5.git_value("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise ValueError("R14 requires a clean source snapshot.")
    results_parent = config["activation"]["r13_results_git_commit"]
    if r5.git_value("merge-base", results_parent, head) != results_parent:
        raise ValueError("R14 source snapshot does not descend from the delivered R13 results commit.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != {
        "train": config["fixed_data"]["train_sha256"],
        "dev": config["fixed_data"]["dev_sha256"],
    }:
        raise ValueError(f"R14 formal data hash drift: {observed_data}")
    parent = config["activation"]
    if (
        not SOURCE_PREREG_PATH.is_file()
        or _text_sha256_lf(SOURCE_PREREG_PATH) != parent["source_decomposition_preregistration_sha256"]
    ):
        raise ValueError("R14 inherited R13 source-decomposition preregistration artifact drifted.")
    source_prereg = _load_json(SOURCE_PREREG_PATH)
    if (
        source_prereg.get("source_only_no_model_outcome") is not True
        or source_prereg.get("fixed_base_latent_sha256") != config["writer"]["fixed_base_latent_sha256"]
    ):
        raise ValueError("R14 inherited R13 source-decomposition preregistration content drifted.")
    if _sha256(args.r12_comparison) != parent["r12_comparison_sha256"]:
        raise ValueError("R14 comparison parent hash drifted.")
    if _sha256(args.r12_collapse_audit) != parent["r12_collapse_audit_sha256"]:
        raise ValueError("R14 collapse-audit parent hash drifted.")
    conditioned = _validate_source_arm(
        root=args.r12_conditioned_root,
        arm="conditioned",
        expected=parent["conditioned_source_sha256"],
    )
    control = _validate_source_arm(
        root=args.r12_control_root,
        arm="constant-control",
        expected=parent["control_source_sha256"],
    )
    r13_parent = _validate_r13_parent(root=args.r13_root, activation=parent)
    return {
        "config": config,
        "config_sha256": _sha256(CONFIG_PATH),
        "r14_preregistration_sha256": _text_sha256_lf(R14_PREREG_PATH),
        "git_commit": head,
        "git_dirty": dirty,
        "data_sha256": observed_data,
        "conditioned_paths": conditioned,
        "control_paths": control,
        "r13_parent_paths": r13_parent["paths"],
        "r13_source_sha256": r13_parent["source_sha256"],
        "r13_report_sha256": r13_parent["report_sha256"],
        "r12_comparison_sha256": _sha256(args.r12_comparison),
        "r12_collapse_audit_sha256": _sha256(args.r12_collapse_audit),
        "source_decomposition_preregistration_sha256": _text_sha256_lf(SOURCE_PREREG_PATH),
    }


def _load_checkpoint_state(path: Path, *, expected_arm: str) -> Mapping[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    expected_names = {
        "basis_raw",
        "attention_query",
        "token_norm.weight",
        "token_norm.bias",
        "coefficient_mlp.0.weight",
        "coefficient_mlp.0.bias",
        "coefficient_mlp.2.weight",
        "coefficient_mlp.2.bias",
    }
    manifest = payload.get("manifest", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("optimizer_step") != R12_OPTIMIZER_STEPS
        or not isinstance(state, Mapping)
        or set(state) != expected_names
        or manifest.get("arm") != expected_arm
        or manifest.get("protocol") != "R12-Shared-Event-Latent-Writer"
    ):
        raise ValueError(f"R14 invalid R12 {expected_arm} step-1152 checkpoint.")
    return state


def _embedding_cache(
    *,
    pipe: Any,
    segments: Sequence[r5.R5Segment],
    source_cache_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    source_payload = torch.load(source_cache_path, map_location="cpu", weights_only=False)
    source_cache = source_payload.get("token_states")
    if (
        source_payload.get("schema") != "vision_memory.r12-event-embedding-cache.v1"
        or not isinstance(source_cache, Mapping)
        or not source_cache
    ):
        raise ValueError("R14 invalid R12 source embedding cache.")
    unique = {segment.segment_id: segment for segment in segments}
    ordered = tuple(unique[key] for key in sorted(unique))
    cache: dict[str, Tensor] = {}
    records = []
    source_matches = 0
    batch_size = 8
    with torch.no_grad():
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            embeddings, masks = pipe.encode_prompt(
                mode="generate",
                prompts=[segment.events[0].event_text for segment in batch],
                device=device,
                dtype=dtype,
            )
            for index, segment in enumerate(batch):
                length = int(masks[index].sum())
                if length <= 0:
                    raise RuntimeError("R14 event encoder produced an empty sequence.")
                value = embeddings[index, :length].detach().float().cpu().contiguous()
                if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
                    raise RuntimeError("R14 event encoder produced invalid token states.")
                source_value = source_cache.get(segment.segment_id)
                matches = source_value is None or torch.equal(value, source_value)
                if not matches:
                    raise RuntimeError(f"R14 event embedding drift for {segment.segment_id}.")
                source_matches += int(source_value is not None)
                cache[segment.segment_id] = value
                records.append(
                    {
                        "segment_id": segment.segment_id,
                        "event_text_sha256": hashlib.sha256(segment.events[0].event_text.encode()).hexdigest(),
                        "shape": list(value.shape),
                        "tensor_sha256": canonical_tensor_sha256(value),
                        "r12_source_match": source_value is not None,
                    }
                )
    cache_path = output_dir / "event_embedding_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        "r12_source_match_count": source_matches,
        "fresh_segment_count": len(cache) - source_matches,
        "records": records,
        "records_sha256": r5.canonical_sha256(records),
        "cache_file_sha256": _sha256(cache_path),
    }
    _write_json(output_dir / "event_embedding_audit.json", audit)
    return cache, audit


def _pooled_features(
    cache: Mapping[str, Tensor], source_state: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    weight = source_state["token_norm.weight"].float()
    bias = source_state["token_norm.bias"].float()
    query = source_state["attention_query"].float()
    features: dict[str, Tensor] = {}
    records = []
    for segment_id in sorted(cache):
        tokens = cache[segment_id].float()
        states = functional.layer_norm(tokens, (2048,), weight, bias, eps=1e-5)
        scores = states @ query / math.sqrt(states.shape[-1])
        attention = torch.softmax(scores, dim=0)
        pooled = (attention @ states).detach().float().contiguous()
        if pooled.shape != (2048,) or not torch.isfinite(pooled).all():
            raise RuntimeError(f"R14 invalid pooled feature for {segment_id}.")
        features[segment_id] = pooled
        entropy = -(attention * attention.clamp_min(1e-30).log()).sum()
        records.append(
            {
                "segment_id": segment_id,
                "feature_sha256": canonical_tensor_sha256(pooled),
                "normalized_attention_entropy": float(entropy / math.log(max(2, attention.numel()))),
                "attention_max": float(attention.max()),
            }
        )
    audit = {
        "source": "frozen R12 conditioned step-1152 token_norm and attention_query",
        "trainable": False,
        "feature_count": len(features),
        "records_sha256": r5.canonical_sha256(records),
        "records": records,
    }
    return features, audit


def _source_coefficients(features: Tensor, state: Mapping[str, Tensor]) -> Tensor:
    hidden = functional.gelu(
        functional.linear(
            features.float(),
            state["coefficient_mlp.0.weight"].float(),
            state["coefficient_mlp.0.bias"].float(),
        )
    )
    raw = functional.linear(
        hidden,
        state["coefficient_mlp.2.weight"].float(),
        state["coefficient_mlp.2.bias"].float(),
    )
    return 2.0 * torch.tanh(raw / 2.0)


class MeanCenteredResidualWriter(nn.Module):
    """Frozen common base plus an exactly train-mean-zero conditional residual."""

    def __init__(
        self,
        *,
        initial_latent: Tensor,
        train_features: Tensor,
        source_state: Mapping[str, Tensor],
    ) -> None:
        super().__init__()
        if tuple(initial_latent.shape) != (1, 4, 128, 128):
            raise ValueError("R14 initial latent shape drifted.")
        if train_features.shape != (R12_TRAIN_SEGMENT_COUNT, 2048):
            raise ValueError(f"R14 train feature shape drifted: {tuple(train_features.shape)}")
        source_basis = source_state["basis_raw"].detach().float().clone()
        flat_basis = source_basis.flatten(1)
        unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
        source_coefficients = _source_coefficients(train_features, source_state)
        source_anchor = source_coefficients.mean(dim=0, keepdim=True)
        common_delta = R12_BASIS_OUTPUT_NORM * source_anchor @ unit_basis
        fixed_base = initial_latent.detach().float() + common_delta.reshape(1, 4, 128, 128)
        self.register_buffer("initial_latent_fp32", initial_latent.detach().float().clone())
        self.register_buffer("fixed_base_latent_fp32", fixed_base.detach().float().clone())
        self.register_buffer("train_features_fp32", train_features.detach().float().clone())
        self.register_buffer("source_anchor_coefficients_fp32", source_anchor.detach().float().clone())
        self.basis_raw = nn.Parameter(source_basis)
        self.coefficient_mlp = nn.Sequential(
            nn.Linear(2048, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, R12_BASIS_COUNT, dtype=torch.float32),
        )
        with torch.no_grad():
            self.coefficient_mlp[0].weight.copy_(source_state["coefficient_mlp.0.weight"])
            self.coefficient_mlp[0].bias.copy_(source_state["coefficient_mlp.0.bias"])
            self.coefficient_mlp[2].weight.copy_(source_state["coefficient_mlp.2.weight"])
            self.coefficient_mlp[2].bias.copy_(source_state["coefficient_mlp.2.bias"])

    def _coefficients(self, features: Tensor) -> Tensor:
        raw = self.coefficient_mlp(features.float())
        return 2.0 * torch.tanh(raw / 2.0)

    def center_invariants(self) -> dict[str, Tensor]:
        all_coefficients = self._coefficients(self.train_features_fp32)
        anchor = all_coefficients.mean(dim=0, keepdim=True)
        centered = all_coefficients - anchor
        flat_basis = self.basis_raw.flatten(1)
        unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
        mean_delta = R12_BASIS_OUTPUT_NORM * centered.mean(dim=0, keepdim=True) @ unit_basis
        return {
            "anchor_coefficients": anchor,
            "mean_residual_coefficients": centered.mean(dim=0, keepdim=True),
            "mean_residual_delta_flat": mean_delta,
        }

    def forward(self, feature: Tensor, *, conditioned: bool = True) -> tuple[Tensor, dict[str, Tensor]]:
        if feature.ndim != 2 or feature.shape != (1, 2048):
            raise ValueError(f"R14 feature shape drifted: {tuple(feature.shape)}")
        invariants = self.center_invariants()
        current = self._coefficients(feature)
        residual_coefficients = current - invariants["anchor_coefficients"]
        if not conditioned:
            residual_coefficients = torch.zeros_like(residual_coefficients)
        flat_basis = self.basis_raw.flatten(1)
        unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
        residual_flat = R12_BASIS_OUTPUT_NORM * residual_coefficients @ unit_basis
        residual = residual_flat.reshape(1, 4, 128, 128)
        latent = self.fixed_base_latent_fp32 + residual
        return latent, {
            "coefficients": current,
            "anchor_coefficients": invariants["anchor_coefficients"],
            "residual_coefficients": residual_coefficients,
            "residual": residual,
            "residual_rms": residual.square().mean().sqrt(),
            "mean_residual_coefficient_max_abs": invariants["mean_residual_coefficients"].abs().max(),
            "mean_residual_delta_max_abs": invariants["mean_residual_delta_flat"].abs().max(),
            "basis_norms": flat_basis.double().norm(dim=1).float(),
        }


def _build_source_decomposition(
    *,
    initial_latent: Tensor,
    train_segments: Sequence[r5.R5Segment],
    features: Mapping[str, Tensor],
    source_state: Mapping[str, Tensor],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[MeanCenteredResidualWriter, dict[str, Any]]:
    # Floating-point reduction is order-sensitive.  Canonical segment-ID order
    # is part of the preregistered decomposition so the fixed-base tensor hash
    # is identical across schedule/data container orderings.
    canonical_train = sorted(train_segments, key=lambda segment: segment.segment_id)
    train_features = torch.stack([features[segment.segment_id] for segment in canonical_train])
    # Derive and hash the frozen base on CPU. A CUDA GEMM can be numerically
    # equivalent yet bitwise different, which would make a tensor SHA depend on
    # hardware. Only after validation is the complete writer moved to the GPU.
    writer = MeanCenteredResidualWriter(
        initial_latent=initial_latent.cpu(),
        train_features=train_features.cpu(),
        source_state={name: value.cpu() for name, value in source_state.items()},
    )
    source_basis = source_state["basis_raw"].float().flatten(1)
    source_unit_basis = source_basis / source_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
    max_error = 0.0
    with torch.no_grad():
        for feature in features.values():
            batch = feature.unsqueeze(0).cpu()
            source_coeff = _source_coefficients(batch, source_state)
            source_latent = initial_latent.cpu() + (R12_BASIS_OUTPUT_NORM * source_coeff @ source_unit_basis).reshape(
                1, 4, 128, 128
            )
            reconstructed, _ = writer(batch)
            max_error = max(max_error, float((source_latent - reconstructed).abs().max()))
    base_hash = canonical_tensor_sha256(writer.fixed_base_latent_fp32.detach().cpu())
    expected_base_hash = config["writer"]["fixed_base_latent_sha256"]
    if base_hash != expected_base_hash:
        raise RuntimeError(f"R14 fixed-base hash drift: expected {expected_base_hash}, observed {base_hash}.")
    if max_error > float(config["technical_gate"]["source_reconstruction_max_abs_lte"]):
        raise RuntimeError(f"R14 source decomposition reconstruction drift: {max_error}")
    audit = {
        "schema": SOURCE_AUDIT_SCHEMA,
        "passed": True,
        "source_event_count": len(features),
        "train_event_count": len(train_segments),
        "fixed_base_latent_sha256": base_hash,
        "source_reconstruction_max_abs": max_error,
        "source_anchor_coefficient_norm": float(writer.source_anchor_coefficients_fp32.detach().double().norm()),
        "initial_latent_sha256": canonical_tensor_sha256(initial_latent),
    }
    return writer.to(device), audit


def _decode(vae: nn.Module, latent: Tensor, compute_dtype: torch.dtype) -> Tensor:
    return decode_model_latents_unit_interval(vae, latent.to(dtype=compute_dtype), clamp=True)


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


def _evaluation_rows(
    *,
    writer: MeanCenteredResidualWriter,
    vae: nn.Module,
    features: Mapping[str, Tensor],
    segments: Sequence[r5.R5Segment],
    donor_map: Mapping[str, str],
    reader_fn: Any,
    reset_image: Tensor,
    base_image: Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
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
            own_latent, _ = writer(features[segment.segment_id].unsqueeze(0).to(device))
            donor_latent, _ = writer(features[donor.segment_id].unsqueeze(0).to(device))
            own_image = _decode(vae, own_latent, compute_dtype)
            donor_image = _decode(vae, donor_latent, compute_dtype)
            _save_image(image_root / "normal" / f"{segment.segment_id}.png", own_image)
            _save_image(image_root / "donor" / f"{segment.segment_id}.png", donor_image)
            item = _item(segment, reset_image)
            for view_index, permutation in enumerate(r5.REVERSE_CYCLIC4):
                reset_output = _reader_output(reader_fn, image=reset_image, segment=segment, permutation=permutation)
                for condition in ("normal", "reset", "donor", "base"):
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
                outputs = {
                    "normal": _reader_output(reader_fn, image=own_image, segment=segment, permutation=permutation),
                    "reset": reset_output,
                    "donor": _reader_output(reader_fn, image=donor_image, segment=segment, permutation=permutation),
                    "base": _reader_output(reader_fn, image=base_image, segment=segment, permutation=permutation),
                }
                for condition, output in outputs.items():
                    rows.append(
                        r5._choice_row(
                            reader_output=output,
                            item=item,
                            checkpoint_label=endpoint,
                            suite=suite,
                            condition=condition,
                            permutation=permutation,
                            view_index=view_index,
                            donor_item_id=(donor.segment_id if condition == "donor" else None),
                        )
                    )
    writer.train()
    return rows


def _latent_rms_penalty(delta: Tensor) -> Tensor:
    mean_square = delta.square().mean()
    penalty_rms = mean_square.clamp_min(R12_LATENT_RMS_SOFT_LIMIT**2).sqrt()
    return R12_LATENT_RMS_PENALTY * (penalty_rms - R12_LATENT_RMS_SOFT_LIMIT).square()


def _gradient_statistics(writer: MeanCenteredResidualWriter) -> dict[str, Any]:
    squared = 0.0
    nonzero = 0
    count = 0
    active = []
    for name, parameter in writer.named_parameters():
        count += parameter.numel()
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"R14 non-finite gradient in {name}.")
        active.append(name)
        squared += float(gradient.detach().double().square().sum())
        nonzero += int((gradient != 0).sum())
    norm = math.sqrt(squared)
    if not math.isfinite(norm) or norm <= 0.0 or nonzero <= 0:
        raise RuntimeError("R14 produced a non-finite or zero aggregate gradient.")
    return {
        "gradient_norm": norm,
        "gradient_nonzero_fraction": nonzero / count,
        "active_parameter_names": active,
    }


def _checkpoint_diagnostics(
    *,
    writer: MeanCenteredResidualWriter,
    vae: nn.Module,
    features: Mapping[str, Tensor],
    representatives: Mapping[str, Sequence[r5.R5Segment]],
    device: torch.device,
    compute_dtype: torch.dtype,
    image_root: Path,
    step: int,
) -> dict[str, Any]:
    rows = []
    writer.eval()
    with torch.no_grad():
        invariants = writer.center_invariants()
        for split, segments in representatives.items():
            for segment in segments:
                latent, diagnostics = writer(features[segment.segment_id].unsqueeze(0).to(device))
                image = _decode(vae, latent, compute_dtype)
                _save_image(
                    image_root / f"step-{step:04d}" / split / f"{segment.segment_id}.png",
                    image,
                )
                rows.append(
                    {
                        "split": split,
                        "segment_id": segment.segment_id,
                        "residual_coefficient_norm": float(diagnostics["residual_coefficients"].double().norm()),
                        "residual_rms": float(diagnostics["residual_rms"]),
                        "image_saturation_fraction": float(((image <= 0.0) | (image >= 1.0)).double().mean()),
                    }
                )
    writer.train()
    return {
        "step": step,
        "rows": rows,
        "fixed_base_latent_sha256": canonical_tensor_sha256(writer.fixed_base_latent_fp32.detach().cpu()),
        "train_mean_residual_coefficient_max_abs": float(invariants["mean_residual_coefficients"].abs().max()),
        "train_mean_residual_delta_max_abs": float(invariants["mean_residual_delta_flat"].abs().max()),
        "basis_norm_min": float(writer.basis_raw.detach().flatten(1).double().norm(dim=1).min()),
        "basis_norm_max": float(writer.basis_raw.detach().flatten(1).double().norm(dim=1).max()),
    }


def _save_checkpoint(
    *,
    step: int,
    micro_cursor: int,
    writer: MeanCenteredResidualWriter,
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
            "schema": "vision_memory.r14-symmetric-donor-ranking-checkpoint-state.v1",
            "next_micro_index": micro_cursor,
            "next_optimizer_step": step,
        },
    )
    _write_json(output_dir / "checkpoint_diagnostics" / f"step-{step:04d}.json", diagnostics)


def _manifest(
    *,
    args: argparse.Namespace,
    validation: Mapping[str, Any],
    selections: Mapping[str, Sequence[r5.R5Segment]],
    schedule: Sequence[Any],
    source_audit: Mapping[str, Any],
    pairing_audit: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = validation["config"]
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": R14_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": validation["git_commit"],
        "git_dirty": validation["git_dirty"],
        "config_sha256": validation["config_sha256"],
        "data_sha256": validation["data_sha256"],
        "activation": {
            "r14_preregistration_sha256": validation["r14_preregistration_sha256"],
            "r13_source_sha256": validation["r13_source_sha256"],
            "r13_report_sha256": validation["r13_report_sha256"],
            "r13_training_git_commit": config["activation"]["r13_training_git_commit"],
            "r13_results_git_commit": config["activation"]["r13_results_git_commit"],
            "r12_comparison_sha256": validation["r12_comparison_sha256"],
            "r12_collapse_audit_sha256": validation["r12_collapse_audit_sha256"],
            "source_decomposition_preregistration_sha256": validation["source_decomposition_preregistration_sha256"],
            "conditioned_sources": config["activation"]["conditioned_source_sha256"],
            "control_sources": config["activation"]["control_source_sha256"],
        },
        "selection_audits": {name: selection_audit(values) for name, values in selections.items()},
        "information_boundary": config["information_boundary"],
        "writer": config["writer"],
        "optimization": config["optimization"],
        "evaluation": config["evaluation"],
        "schedule": {
            "micro_steps": len(schedule),
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "receipts_sha256": r5.canonical_sha256([unit.receipt() for unit in schedule]),
        },
        "training_pairing": dict(pairing_audit),
        "source_decomposition": dict(source_audit),
        "embedding_audit": dict(embedding_audit),
        "feature_audit_sha256": r5.canonical_sha256(feature_audit),
        "snapshot_bindings": dict(snapshot_bindings),
        "determinism": determinism,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "best_checkpoint_selection_forbidden": True,
        "formal_success_claim_allowed": False,
    }


def _selection_relationship_audit(
    selections: Mapping[str, Sequence[r5.R5Segment]],
) -> dict[str, Any]:
    disjoint_splits = ("train", "dev_select", "dev_replay", "dev_final")
    entities = {split: {segment.query_entity_id for segment in selections[split]} for split in disjoint_splits}
    overlaps = {
        f"{left}__{right}": len(entities[left] & entities[right])
        for left_index, left in enumerate(disjoint_splits)
        for right in disjoint_splits[left_index + 1 :]
    }
    train_ids = {segment.segment_id for segment in selections["train"]}
    audit_ids = {segment.segment_id for segment in selections["train_audit"]}
    train_values = {segment.query.target for segment in selections["train"]}
    evaluation_values = {
        split: {segment.query.target for segment in selections[split]}
        for split in ("dev_select", "dev_replay", "dev_final")
    }
    passed = bool(
        all(value == 0 for value in overlaps.values())
        and audit_ids.issubset(train_ids)
        and all(values.issubset(train_values) for values in evaluation_values.values())
        and len(train_values) == 36
        and all(len(values) == 24 for values in evaluation_values.values())
    )
    return {
        "passed": passed,
        "entity_overlap_counts": overlaps,
        "train_audit_is_train_subset": audit_ids.issubset(train_ids),
        "target_value_counts": {
            "train": len(train_values),
            **{split: len(values) for split, values in evaluation_values.items()},
        },
        "evaluation_target_values_subset_of_train": {
            split: values.issubset(train_values) for split, values in evaluation_values.items()
        },
    }


def _technical_gate(
    *,
    args: argparse.Namespace,
    writer: MeanCenteredResidualWriter,
    pipe: Any,
    reader: nn.Module,
    micro_rows: Sequence[Mapping[str, Any]],
    optimizer_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Any],
    selections: Mapping[str, Sequence[r5.R5Segment]],
    pairing_map: Mapping[str, str],
    pairing_audit: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    checkpoint_steps: set[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    evaluation_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    snapshots_unchanged: bool,
) -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    train_ids = {segment.segment_id for segment in selections["train"]}
    counts = Counter(row["segment_id"] for row in micro_rows)
    views = Counter((row["segment_id"], int(row["forward_cyclic_training_view"])) for row in micro_rows)
    exact_views = all(
        counts[segment_id] == 32 and all(views[(segment_id, view)] == 8 for view in range(4))
        for segment_id in train_ids
    )
    by_id = {segment.segment_id: segment for segment in selections["train"]}
    pairing_ok = bool(
        pairing_audit.get("schema") == "vision_memory.r14-symmetric-donor-pairing-audit.v1"
        and pairing_audit.get("seed") == args.pairing_seed == R14_PAIR_SEED
        and pairing_audit.get("segment_count") == R12_TRAIN_SEGMENT_COUNT
        and pairing_audit.get("pair_count") == R12_TRAIN_SEGMENT_COUNT // 2
        and pairing_audit.get("different_target_value") is True
        and pairing_audit.get("involution") is True
        and pairing_audit.get("pairs_sha256") == config["training_pairing"]["pairs_sha256"]
        and set(pairing_map) == train_ids
        and all(
            pairing_map.get(pairing_map[source]) == source and pairing_map[source] != source for source in train_ids
        )
        and all(by_id[source].query.target != by_id[pairing_map[source]].query.target for source in train_ids)
    )
    donor_assignment_ok = all(
        row.get("donor_segment_id") == pairing_map.get(row["segment_id"])
        and row.get("donor_target_value") != row.get("target_value")
        and row.get("reader_calls") == 2
        and row.get("same_query_and_permutation_for_own_and_donor") is True
        for row in micro_rows
    )
    metric_keys = (
        "ce",
        "own_ce",
        "donor_ce",
        "ranking_gap",
        "ranking_loss",
        "own_target_margin",
        "donor_target_margin",
        "objective",
        "residual_rms",
        "donor_residual_rms",
        "residual_penalty",
        "coefficient_penalty",
    )
    finite_metrics = all(math.isfinite(float(row[key])) for row in micro_rows for key in metric_keys)
    ranking_formula_ok = all(
        abs(float(row["ranking_loss"]) - max(0.0, R14_RANKING_MARGIN + float(row["own_ce"]) - float(row["donor_ce"])))
        <= 1e-5
        and bool(row["ranking_satisfied"]) == (float(row["ranking_loss"]) <= 1e-7)
        and float(row["ranking_margin"]) == R14_RANKING_MARGIN
        and float(row["ranking_weight"]) == R14_RANKING_WEIGHT
        for row in micro_rows
    )
    objective_formula_ok = all(
        abs(
            float(row["objective"])
            - (
                float(row["own_ce"])
                + R14_RANKING_WEIGHT * float(row["ranking_loss"])
                + float(row["residual_penalty"])
                + float(row["coefficient_penalty"])
            )
        )
        <= 1e-4
        for row in micro_rows
    )
    frozen_modules = (pipe.unet, pipe.text_encoder, pipe.vae, reader)
    frozen_ok = all(
        not parameter.requires_grad and parameter.grad is None
        for module in frozen_modules
        for parameter in module.parameters()
    )
    expected_rows = {split: len(selections[split]) * 4 * 4 * 2 for split in EVALUATION_SPLITS}
    evaluation_complete = all(
        len(evaluation_rows[split]) == expected_rows[split]
        and {
            (row["pair_unit"], row["checkpoint"], row["condition"], int(row["view_index"]))
            for row in evaluation_rows[split]
        }
        == {
            (segment.segment_id, checkpoint, condition, view)
            for segment in selections[split]
            for checkpoint in ("m0", PRIMARY_ENDPOINT)
            for condition in ("normal", "reset", "donor", "base")
            for view in range(4)
        }
        for split in EVALUATION_SPLITS
    )
    center_limit = float(config["technical_gate"]["train_mean_residual_max_abs_lte"])
    center_ok = all(
        float(value["train_mean_residual_coefficient_max_abs"]) <= center_limit
        and float(value["train_mean_residual_delta_max_abs"])
        <= float(config["technical_gate"]["train_mean_residual_delta_max_abs_lte"])
        and value["fixed_base_latent_sha256"] == config["writer"]["fixed_base_latent_sha256"]
        for value in diagnostics.values()
    )
    checks = {
        "micro_count": len(micro_rows) == R12_MICRO_STEPS,
        "optimizer_count": len(optimizer_rows) == R12_OPTIMIZER_STEPS,
        "schedule_count": len(schedule) == R12_MICRO_STEPS,
        "all_train_ids_seen": {row["segment_id"] for row in micro_rows} == train_ids,
        "exact_training_views": exact_views,
        "finite_metrics": finite_metrics,
        "symmetric_pairing": pairing_ok,
        "paired_donor_assignment": donor_assignment_ok,
        "ranking_formula": ranking_formula_ok,
        "objective_formula": objective_formula_ok,
        "finite_nonzero_gradients": all(
            math.isfinite(float(row["gradient_norm"]))
            and float(row["gradient_norm"]) > 0.0
            and float(row["gradient_nonzero_fraction"]) > 0.0
            for row in optimizer_rows
        ),
        "checkpoint_steps": checkpoint_steps == set(R12_CHECKPOINT_STEPS),
        "checkpoint_diagnostics": set(diagnostics) == set(R12_CHECKPOINT_STEPS),
        "mean_zero_constraint": center_ok,
        "source_decomposition": source_audit.get("passed") is True
        and source_audit.get("source_event_count") == 216
        and source_audit.get("train_event_count") == R12_TRAIN_SEGMENT_COUNT,
        "embedding_cache": embedding_audit.get("passed") is True
        and embedding_audit.get("segment_count") == 216
        and embedding_audit.get("r12_source_match_count") == 192
        and embedding_audit.get("fresh_segment_count") == R14_FRESH_DEV_FINAL_COUNT,
        "frozen_modules": frozen_ok,
        "snapshots_unchanged": snapshots_unchanged,
        "evaluation_complete": evaluation_complete,
        "writer_fp32": all(parameter.dtype == torch.float32 for parameter in writer.parameters()),
        "gradient_clipping_disabled": all(row.get("gradient_clipped") is False for row in optimizer_rows),
    }
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "micro_records": len(micro_rows),
        "optimizer_records": len(optimizer_rows),
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "trainable_parameter_names": [name for name, parameter in writer.named_parameters() if parameter.requires_grad],
        "trainable_parameter_count": sum(parameter.numel() for parameter in writer.parameters()),
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in optimizer_rows),
        "minimum_gradient_nonzero_fraction": min(float(row["gradient_nonzero_fraction"]) for row in optimizer_rows),
        "evaluation_row_counts": {key: len(value) for key, value in evaluation_rows.items()},
        "pairing_sha256": pairing_audit.get("pairs_sha256"),
        "ranking_satisfied_micro_count": sum(bool(row["ranking_satisfied"]) for row in micro_rows),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    validation = _validate_args(args)
    determinism = r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    r8.set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    config = validation["config"]

    data = r5._load_data(args, optimizer_steps=0)
    train_selected, train_audit = select_balanced_train_f1(data.train_pools["F1"])
    dev_records = tuple(r5.read_episode_jsonl(args.dev))
    dev_pool = r5.build_r5_family_pools(dev_records, pairing_seed=args.split_seed)["F1"]
    dev_select, dev_replay = select_entity_disjoint_dev_f1(dev_pool)
    excluded_entities = {segment.query_entity_id for segment in (*train_selected, *dev_select, *dev_replay)}
    dev_final = select_fresh_dev_final_f1(dev_pool, excluded_entities=excluded_entities)
    selections = {
        "train": train_selected,
        "train_audit": train_audit,
        "dev_select": dev_select,
        "dev_replay": dev_replay,
        "dev_final": dev_final,
    }
    selection_payload = {name: selection_audit(values) for name, values in selections.items()}
    if selection_payload["dev_final"]["payload_sha256"] != R14_FRESH_DEV_FINAL_SHA256:
        raise RuntimeError("R14 fresh-final payload hash drifted after selection.")
    relationship_audit = _selection_relationship_audit(selections)
    if not relationship_audit["passed"]:
        raise RuntimeError(f"R14 split relationship audit failed: {relationship_audit}")
    _write_json(
        args.output_dir / "selection_audit.json",
        {"splits": selection_payload, "relationships": relationship_audit},
    )
    pairing_map = symmetric_donor_mapping(train_selected, seed=args.pairing_seed)
    pairing_audit = symmetric_pairing_audit(train_selected, pairing_map)
    train_by_id = {segment.segment_id: segment for segment in train_selected}
    if (
        pairing_audit["pairs_sha256"] != config["training_pairing"]["pairs_sha256"]
        or pairing_audit["pair_count"] != R12_TRAIN_SEGMENT_COUNT // 2
        or pairing_audit["different_target_value"] is not True
        or pairing_audit["involution"] is not True
    ):
        raise RuntimeError(f"R14 symmetric training-pair audit drifted: {pairing_audit}")
    _write_json(args.output_dir / "pairing_audit.json", pairing_audit)
    schedule = build_training_schedule(train_selected)
    _write_json(
        args.output_dir / "schedule_audit.json",
        {
            "micro_steps": len(schedule),
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "receipts_sha256": r5.canonical_sha256([unit.receipt() for unit in schedule]),
            "receipts": [unit.receipt() for unit in schedule],
        },
    )

    device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    compute_dtype = r5.compute_dtype(device)
    reader_dtype = r5.compute_dtype(reader_device)
    pipe = r5._load_pipeline(args, device, compute_dtype)
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
    initial_rgb = r5._initial_rgb_tensor(resolution=1024, device=device, dtype=compute_dtype)
    with torch.no_grad():
        initial_latent = encode_model_latent(pipe.vae, initial_rgb).detach().float().cpu()
    if canonical_tensor_sha256(initial_latent) != INITIAL_LATENT_SHA256:
        raise RuntimeError("R14 blank latent hash drifted.")

    conditioned_state = _load_checkpoint_state(
        validation["conditioned_paths"]["checkpoint"], expected_arm="conditioned"
    )
    _load_checkpoint_state(validation["control_paths"]["checkpoint"], expected_arm="constant-control")
    embedding_segments = tuple(
        {segment.segment_id: segment for values in selections.values() for segment in values}.values()
    )
    cache, embedding_audit = _embedding_cache(
        pipe=pipe,
        segments=embedding_segments,
        source_cache_path=validation["conditioned_paths"]["embedding_cache"],
        device=device,
        dtype=compute_dtype,
        output_dir=args.output_dir,
    )
    features, feature_audit = _pooled_features(cache, conditioned_state)
    _write_json(args.output_dir / "feature_audit.json", feature_audit)
    writer, source_audit = _build_source_decomposition(
        initial_latent=initial_latent,
        train_segments=train_selected,
        features=features,
        source_state=conditioned_state,
        config=config,
        device=device,
    )
    _write_json(args.output_dir / "source_decomposition_audit.json", source_audit)

    network_parameters = [parameter for name, parameter in writer.named_parameters() if name != "basis_raw"]
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
    reset_image = _decode(pipe.vae, writer.initial_latent_fp32, compute_dtype).detach()
    base_image = _decode(pipe.vae, writer.fixed_base_latent_fp32, compute_dtype).detach()
    _save_image(args.output_dir / "images" / "reset.png", reset_image)
    _save_image(args.output_dir / "images" / "fixed_base.png", base_image)

    manifest = _manifest(
        args=args,
        validation=validation,
        selections=selections,
        schedule=schedule,
        source_audit=source_audit,
        pairing_audit=pairing_audit,
        embedding_audit=embedding_audit,
        feature_audit=feature_audit,
        snapshot_bindings=snapshot_bindings,
        determinism=determinism,
    )
    _write_json(args.output_dir / "manifest.json", manifest)
    representatives = {
        split: selections[split][:REPRESENTATIVE_PER_SPLIT] for split in ("train_audit", "dev_select", "dev_final")
    }
    checkpoint_diagnostics: dict[int, dict[str, Any]] = {}
    checkpoint_diagnostics[0] = _checkpoint_diagnostics(
        writer=writer,
        vae=pipe.vae,
        features=features,
        representatives=representatives,
        device=device,
        compute_dtype=compute_dtype,
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
        diagnostics=checkpoint_diagnostics[0],
    )

    micro_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    rolling_own_ce: list[float] = []
    rolling_donor_ce: list[float] = []
    rolling_ranking_loss: list[float] = []
    rolling_ranking_satisfied: list[bool] = []
    for unit in schedule:
        segment = unit.segment
        donor_segment = train_by_id[pairing_map[segment.segment_id]]
        feature = features[segment.segment_id].unsqueeze(0).to(device)
        donor_feature = features[donor_segment.segment_id].unsqueeze(0).to(device)
        latent, diagnostics = writer(feature)
        donor_latent, donor_diagnostics = writer(donor_feature)
        image = _decode(pipe.vae, latent, compute_dtype)
        donor_image = _decode(pipe.vae, donor_latent, compute_dtype)
        permutation = CYCLIC4[unit.forward_cyclic_training_view]
        own_output = _reader_output(train_reader, image=image, segment=segment, permutation=permutation)
        donor_output = _reader_output(train_reader, image=donor_image, segment=segment, permutation=permutation)
        own_ce = own_output.loss
        donor_ce = donor_output.loss
        if (
            not isinstance(own_ce, Tensor)
            or own_ce.numel() != 1
            or not torch.isfinite(own_ce)
            or not isinstance(donor_ce, Tensor)
            or donor_ce.numel() != 1
            or not torch.isfinite(donor_ce)
        ):
            raise RuntimeError("R14 Reader returned an invalid own/donor scalar CE pair.")
        ranking_loss = symmetric_ranking_loss(own_ce, donor_ce)
        own_residual_penalty = _latent_rms_penalty(diagnostics["residual"])
        donor_residual_penalty = _latent_rms_penalty(donor_diagnostics["residual"])
        residual_penalty = 0.5 * (own_residual_penalty + donor_residual_penalty)
        own_coefficient_penalty = R12_COEFFICIENT_L2_PENALTY * diagnostics["residual_coefficients"].square().mean()
        donor_coefficient_penalty = (
            R12_COEFFICIENT_L2_PENALTY * donor_diagnostics["residual_coefficients"].square().mean()
        )
        coefficient_penalty = 0.5 * (own_coefficient_penalty + donor_coefficient_penalty)
        objective = (
            own_ce
            + R14_RANKING_WEIGHT * ranking_loss
            + residual_penalty.to(device=own_ce.device)
            + coefficient_penalty.to(device=own_ce.device)
        )
        if not torch.isfinite(objective):
            raise RuntimeError("R14 objective became non-finite.")
        (objective / R12_GRADIENT_ACCUMULATION).backward()
        ordered_target_index = permutation.index(segment.query.target_index)
        own_target_margin = choice_target_margin(own_output.choice_logits, ordered_target_index)
        donor_target_margin = choice_target_margin(donor_output.choice_logits, ordered_target_index)
        row = {
            "schema": MICRO_SCHEMA,
            "global_micro_index": unit.global_micro_index,
            "epoch_zero": unit.epoch_zero,
            "micro_in_epoch": unit.micro_in_epoch,
            "optimizer_step_zero": unit.optimizer_step_zero,
            "segment_id": segment.segment_id,
            "target_value": segment.query.target,
            "donor_segment_id": donor_segment.segment_id,
            "donor_target_value": donor_segment.query.target,
            "forward_cyclic_training_view": unit.forward_cyclic_training_view,
            "permutation": list(permutation),
            "same_query_and_permutation_for_own_and_donor": True,
            "reader_calls": 2,
            "ce": float(own_ce.detach()),
            "own_ce": float(own_ce.detach()),
            "donor_ce": float(donor_ce.detach()),
            "ranking_gap": float((donor_ce - own_ce).detach()),
            "ranking_margin": R14_RANKING_MARGIN,
            "ranking_weight": R14_RANKING_WEIGHT,
            "ranking_loss": float(ranking_loss.detach()),
            "ranking_satisfied": bool(float(ranking_loss.detach()) <= 1e-7),
            "own_target_margin": float(own_target_margin.detach()),
            "donor_target_margin": float(donor_target_margin.detach()),
            "objective": float(objective.detach()),
            "residual_rms": float(diagnostics["residual_rms"].detach()),
            "donor_residual_rms": float(donor_diagnostics["residual_rms"].detach()),
            "residual_penalty": float(residual_penalty.detach()),
            "own_residual_penalty": float(own_residual_penalty.detach()),
            "donor_residual_penalty": float(donor_residual_penalty.detach()),
            "coefficient_penalty": float(coefficient_penalty.detach()),
            "own_coefficient_penalty": float(own_coefficient_penalty.detach()),
            "donor_coefficient_penalty": float(donor_coefficient_penalty.detach()),
            "residual_coefficient_norm": float(diagnostics["residual_coefficients"].detach().double().norm()),
            "donor_residual_coefficient_norm": float(
                donor_diagnostics["residual_coefficients"].detach().double().norm()
            ),
            "mean_residual_coefficient_max_abs": float(diagnostics["mean_residual_coefficient_max_abs"].detach()),
            "mean_residual_delta_max_abs": float(diagnostics["mean_residual_delta_max_abs"].detach()),
            "image_saturation_fraction": float(((image.detach() <= 0.0) | (image.detach() >= 1.0)).double().mean()),
            "donor_image_saturation_fraction": float(
                ((donor_image.detach() <= 0.0) | (donor_image.detach() >= 1.0)).double().mean()
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        micro_rows.append(row)
        rolling_own_ce.append(row["own_ce"])
        rolling_donor_ce.append(row["donor_ce"])
        rolling_ranking_loss.append(row["ranking_loss"])
        rolling_ranking_satisfied.append(row["ranking_satisfied"])
        _append_jsonl(args.output_dir / "micro_metrics.jsonl", (row,))
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
            "elapsed_seconds": time.monotonic() - started,
        }
        optimizer_rows.append(optimizer_row)
        _append_jsonl(args.output_dir / "optimizer_metrics.jsonl", (optimizer_row,))
        if step in R12_CHECKPOINT_STEPS:
            checkpoint_diagnostics[step] = _checkpoint_diagnostics(
                writer=writer,
                vae=pipe.vae,
                features=features,
                representatives=representatives,
                device=device,
                compute_dtype=compute_dtype,
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
                diagnostics=checkpoint_diagnostics[step],
            )
        if step == 1 or step % 16 == 0 or step in R12_CHECKPOINT_STEPS:
            own_window = rolling_own_ce[-64:]
            donor_window = rolling_donor_ce[-64:]
            ranking_window = rolling_ranking_loss[-64:]
            satisfied_window = rolling_ranking_satisfied[-64:]
            print(
                json.dumps(
                    {
                        "milestone": "r14_optimizer_step",
                        "optimizer_step": step,
                        "mean_last64_micro_ce": sum(own_window) / len(own_window),
                        "mean_last64_own_ce": sum(own_window) / len(own_window),
                        "mean_last64_donor_ce": sum(donor_window) / len(donor_window),
                        "mean_last64_ranking_loss": sum(ranking_window) / len(ranking_window),
                        "last64_ranking_satisfied_fraction": sum(satisfied_window) / len(satisfied_window),
                        "gradient_norm": gradient["gradient_norm"],
                        "elapsed_seconds": optimizer_row["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    endpoint = PRIMARY_ENDPOINT
    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    statistics_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in EVALUATION_SPLITS:
        segments = selections[split]
        suite = f"{SUITE_PREFIX}_{split}"
        rows = _evaluation_rows(
            writer=writer,
            vae=pipe.vae,
            features=features,
            segments=segments,
            donor_map=donor_derangement(segments),
            reader_fn=eval_reader,
            reset_image=reset_image,
            base_image=base_image,
            device=device,
            compute_dtype=compute_dtype,
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
    snapshots_unchanged = {key: value["snapshot_payload_sha256"] for key, value in snapshot_bindings.items()} == {
        key: value["snapshot_payload_sha256"] for key, value in end_bindings.items()
    }
    checkpoint_steps = {
        int(path.stem.removeprefix("step-")) for path in (args.output_dir / "checkpoints").glob("step-*.pt")
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
        pairing_map=pairing_map,
        pairing_audit=pairing_audit,
        embedding_audit=embedding_audit,
        source_audit=source_audit,
        checkpoint_steps=checkpoint_steps,
        diagnostics=checkpoint_diagnostics,
        evaluation_rows=evaluation_rows,
        snapshots_unchanged=snapshots_unchanged,
    )
    _write_json(args.output_dir / "technical_gate.json", technical)
    for split in EVALUATION_SPLITS:
        suite = f"{SUITE_PREFIX}_{split}"
        statistics = []
        for segment in selections[split]:
            value = centered_target_statistics(
                evaluation_rows[split],
                suite=suite,
                target_segment_id=segment.segment_id,
                endpoint=endpoint,
            )
            value["target_gate"] = centered_target_gate(value, technical_gate=bool(technical["passed"]))
            value["target_value"] = segment.query.target
            statistics.append(value)
        statistics_by_split[split] = statistics
        _write_json(args.output_dir / f"{split}_statistics.json", statistics)
    pass_counts = {
        split: sum(bool(value["target_gate"]) for value in statistics)
        for split, statistics in statistics_by_split.items()
    }
    required = {
        "train_audit": R12_TRAIN_AUDIT_COUNT,
        "dev_select": R12_DEV_SELECT_COUNT,
        "dev_replay": R12_DEV_FINAL_COUNT,
        "dev_final": R14_FRESH_DEV_FINAL_COUNT,
    }
    arm_gate = bool(technical["passed"] and pass_counts == required)
    if arm_gate:
        decision = "symmetric_donor_ranked_shared_f1_candidate_pass_advance_to_recurrent_state_algebra"
    elif pass_counts["train_audit"] == R12_TRAIN_AUDIT_COUNT:
        decision = "symmetric_donor_ranking_fit_without_complete_heldout_causal_generalization"
    else:
        decision = "symmetric_donor_ranking_did_not_fit_fixed_f1_boundary"
    first_window = micro_rows[:64]
    last_window = micro_rows[-64:]
    training_ranking = {
        "margin": R14_RANKING_MARGIN,
        "weight": R14_RANKING_WEIGHT,
        "first_64_mean_own_ce": sum(float(row["own_ce"]) for row in first_window) / len(first_window),
        "first_64_mean_donor_ce": sum(float(row["donor_ce"]) for row in first_window) / len(first_window),
        "first_64_mean_ranking_loss": sum(float(row["ranking_loss"]) for row in first_window) / len(first_window),
        "first_64_satisfied_fraction": sum(bool(row["ranking_satisfied"]) for row in first_window) / len(first_window),
        "last_64_mean_own_ce": sum(float(row["own_ce"]) for row in last_window) / len(last_window),
        "last_64_mean_donor_ce": sum(float(row["donor_ce"]) for row in last_window) / len(last_window),
        "last_64_mean_ranking_loss": sum(float(row["ranking_loss"]) for row in last_window) / len(last_window),
        "last_64_satisfied_fraction": sum(bool(row["ranking_satisfied"]) for row in last_window) / len(last_window),
        "all_micro_satisfied_fraction": sum(bool(row["ranking_satisfied"]) for row in micro_rows) / len(micro_rows),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": R14_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": validation["git_commit"],
        "config_sha256": validation["config_sha256"],
        "optimizer_steps": len(optimizer_rows),
        "micro_steps": len(micro_rows),
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "endpoint": endpoint,
        "selection_audits": selection_payload,
        "training_pairing": pairing_audit,
        "training_ranking": training_ranking,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "split_target_pass_counts": pass_counts,
            "required_target_pass_counts": required,
            "arm_gate": arm_gate,
            "formal_success_gate": False,
        },
        "split_statistics": statistics_by_split,
        "decision": decision,
        "full_success_claim_allowed": False,
        "diagnostic_only_not_formal_success": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(args.output_dir / "r14_symmetric_donor_summary.json", summary)
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
