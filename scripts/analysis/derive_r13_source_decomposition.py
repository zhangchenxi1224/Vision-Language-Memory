"""Derive the deterministic R13 common-base hashes from exact R11/R12 parents.

This is a source-only preregistration audit: it performs no Reader inference,
optimization, checkpoint selection, or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_trainable_tensor(path: Path) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    if payload.get("schema_version") != 1 or not isinstance(state, Mapping) or len(state) != 1:
        raise ValueError("R13 source audit requires an R11 step-000 single-latent checkpoint.")
    value = next(iter(state.values())).detach().float().cpu()
    if value.shape != (1, 4, 128, 128):
        raise ValueError(f"R13 R11 initial latent shape drifted: {tuple(value.shape)}")
    return value


def _r12_state(path: Path) -> Mapping[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    expected = {
        "basis_raw",
        "attention_query",
        "token_norm.weight",
        "token_norm.bias",
        "coefficient_mlp.0.weight",
        "coefficient_mlp.0.bias",
        "coefficient_mlp.2.weight",
        "coefficient_mlp.2.bias",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("optimizer_step") != 1152
        or not isinstance(state, Mapping)
        or set(state) != expected
        or payload.get("manifest", {}).get("arm") != "conditioned"
    ):
        raise ValueError("R13 source audit requires the exact R12 conditioned step-1152 checkpoint.")
    return state


def _train_ids(path: Path) -> list[str]:
    ids = {
        str(json.loads(line)["segment_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if len(ids) != 144:
        raise ValueError(f"R13 source audit expected 144 R12 train IDs, got {len(ids)}.")
    return sorted(ids)


def derive(
    *,
    r11_step0: Path,
    r12_checkpoint: Path,
    r12_embedding_cache: Path,
    r12_micro_metrics: Path,
) -> dict[str, Any]:
    initial = _single_trainable_tensor(r11_step0)
    state = _r12_state(r12_checkpoint)
    cache_payload = torch.load(r12_embedding_cache, map_location="cpu", weights_only=False)
    cache = cache_payload.get("token_states")
    if cache_payload.get("schema") != "vision_memory.r12-event-embedding-cache.v1" or not isinstance(cache, Mapping):
        raise ValueError("R13 source audit received an invalid R12 embedding cache.")
    train_ids = _train_ids(r12_micro_metrics)
    if not set(train_ids).issubset(cache):
        raise ValueError("R13 source audit train IDs are missing from the R12 embedding cache.")

    pooled = []
    weight = state["token_norm.weight"].float()
    bias = state["token_norm.bias"].float()
    query = state["attention_query"].float()
    for segment_id in train_ids:
        tokens = cache[segment_id].float()
        states = functional.layer_norm(tokens, (2048,), weight, bias, eps=1e-5)
        attention = torch.softmax(states @ query / math.sqrt(2048), dim=0)
        pooled.append(attention @ states)
    features = torch.stack(pooled)
    hidden = functional.gelu(
        functional.linear(
            features,
            state["coefficient_mlp.0.weight"].float(),
            state["coefficient_mlp.0.bias"].float(),
        )
    )
    raw = functional.linear(
        hidden,
        state["coefficient_mlp.2.weight"].float(),
        state["coefficient_mlp.2.bias"].float(),
    )
    coefficients = 2.0 * torch.tanh(raw / 2.0)
    anchor = coefficients.mean(dim=0, keepdim=True)
    flat_basis = state["basis_raw"].float().flatten(1)
    unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
    common_delta = (80.0 * anchor @ unit_basis).reshape(1, 4, 128, 128)
    fixed_base = initial + common_delta
    centered = coefficients - anchor
    centered_delta = 80.0 * centered.mean(dim=0, keepdim=True) @ unit_basis
    return {
        "schema": "vision_memory.r13-source-decomposition-preregistration.v1",
        "source_only_no_model_outcome": True,
        "source_sha256": {
            "r11_step0": _sha256(r11_step0),
            "r12_conditioned_step1152": _sha256(r12_checkpoint),
            "r12_embedding_cache": _sha256(r12_embedding_cache),
            "r12_micro_metrics": _sha256(r12_micro_metrics),
        },
        "train_event_count": len(train_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "initial_latent_sha256": canonical_tensor_sha256(initial),
        "train_features_sha256": canonical_tensor_sha256(features),
        "source_anchor_coefficients_sha256": canonical_tensor_sha256(anchor),
        "source_anchor_coefficient_norm": float(anchor.double().norm()),
        "common_delta_sha256": canonical_tensor_sha256(common_delta),
        "fixed_base_latent_sha256": canonical_tensor_sha256(fixed_base),
        "train_mean_centered_coefficient_max_abs": float(centered.mean(dim=0).abs().max()),
        "train_mean_centered_delta_max_abs": float(centered_delta.abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r11-step0", type=Path, required=True)
    parser.add_argument("--r12-checkpoint", type=Path, required=True)
    parser.add_argument("--r12-embedding-cache", type=Path, required=True)
    parser.add_argument("--r12-micro-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = derive(
        r11_step0=args.r11_step0,
        r12_checkpoint=args.r12_checkpoint,
        r12_embedding_cache=args.r12_embedding_cache,
        r12_micro_metrics=args.r12_micro_metrics,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
