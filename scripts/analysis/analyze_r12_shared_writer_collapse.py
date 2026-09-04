"""Post-hoc representation audit for a completed R12 shared-writer arm.

This audit never changes the preregistered R12 outcome.  It asks where event
identity is lost: frozen token states, learned attention pooling, coefficient
mapping, or the latent dictionary projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor


EXPECTED_COMMIT = "c401954e5624e99347306a60c3f86202c941ab34"
EXPECTED_CONFIG_SHA256 = "85712f6e53fee8c83366cd3ca41f132e9152288b6226d7f330bcf5f49e7705e1"
SPLITS = ("train", "train_audit", "dev_select", "dev_final")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R12 collapse audit expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"R12 collapse audit expected JSON objects: {path}")
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_sources(
    summary_path: Path,
    checkpoint_path: Path,
    embedding_cache_path: Path,
    micro_metrics_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Tensor], Mapping[str, Tensor]]:
    summary = _load_json(summary_path)
    if (
        summary.get("schema") != "vision_memory.r12-shared-event-latent-writer-summary.v1"
        or summary.get("status") != "completed"
        or summary.get("git_commit") != EXPECTED_COMMIT
        or summary.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or summary.get("optimizer_steps") != 1152
        or summary.get("micro_steps") != 4608
    ):
        raise ValueError("R12 collapse audit source summary is incomplete or drifted.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("optimizer_step") != 1152:
        raise ValueError("R12 collapse audit requires the exact step-1152 checkpoint.")
    state = checkpoint.get("trainable_state")
    required_state = {
        "basis_raw",
        "attention_query",
        "token_norm.weight",
        "token_norm.bias",
        "coefficient_mlp.0.weight",
        "coefficient_mlp.0.bias",
        "coefficient_mlp.2.weight",
        "coefficient_mlp.2.bias",
    }
    if not isinstance(state, Mapping) or set(state) != required_state:
        raise ValueError("R12 collapse audit checkpoint trainable state drifted.")
    cache_payload = torch.load(embedding_cache_path, map_location="cpu", weights_only=False)
    cache = cache_payload.get("token_states")
    if (
        cache_payload.get("schema") != "vision_memory.r12-event-embedding-cache.v1"
        or not isinstance(cache, Mapping)
        or not cache
    ):
        raise ValueError("R12 collapse audit embedding cache is invalid.")
    micro = _load_jsonl(micro_metrics_path)
    if len(micro) != 4608:
        raise ValueError(f"R12 collapse audit micro-metric count drifted: {len(micro)}")
    return summary, state, cache


def _labels(
    summary: Mapping[str, Any], micro_metrics_path: Path, cache: Mapping[str, Tensor]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    labels: dict[str, str] = {}
    split_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    for row in _load_jsonl(micro_metrics_path):
        segment_id = str(row["segment_id"])
        target = str(row["target_value"])
        previous = labels.setdefault(segment_id, target)
        if previous != target:
            raise ValueError(f"R12 target label drift for {segment_id}.")
    split_ids["train"] = sorted(labels)
    for split in SPLITS[1:]:
        statistics = summary["split_statistics"][split]
        for row in statistics:
            segment_id = str(row["target_segment_id"])
            target = str(row["target_value"])
            if segment_id in labels and split != "train_audit":
                raise ValueError(f"R12 split overlap in collapse audit: {segment_id}")
            previous = labels.setdefault(segment_id, target)
            if previous != target:
                raise ValueError(f"R12 target label drift for {segment_id}.")
            split_ids[split].append(segment_id)
        split_ids[split].sort()
    if set(labels) != set(cache):
        raise ValueError(
            "R12 collapse audit label/cache mismatch: "
            f"missing={sorted(set(cache) - set(labels))}, extra={sorted(set(labels) - set(cache))}"
        )
    return labels, split_ids


def _writer_representations(
    state: Mapping[str, Tensor], cache: Mapping[str, Tensor], segment_ids: Sequence[str]
) -> tuple[dict[str, Tensor], dict[str, float], dict[str, float]]:
    weight = state["token_norm.weight"].float()
    bias = state["token_norm.bias"].float()
    query = state["attention_query"].float()
    first_weight = state["coefficient_mlp.0.weight"].float()
    first_bias = state["coefficient_mlp.0.bias"].float()
    final_weight = state["coefficient_mlp.2.weight"].float()
    final_bias = state["coefficient_mlp.2.bias"].float()
    basis = state["basis_raw"].float().flatten(1)
    unit_basis = basis / basis.double().norm(dim=1).float().clamp_min(1e-12).unsqueeze(1)
    token_means = []
    attention_pools = []
    coefficients = []
    attention_entropies = []
    attention_maxima = []
    for segment_id in segment_ids:
        tokens = cache[segment_id].float()
        if tokens.ndim != 2 or tokens.shape[1] != 2048 or not torch.isfinite(tokens).all():
            raise ValueError(f"R12 invalid cached token states: {segment_id}")
        states = functional.layer_norm(tokens, (2048,), weight, bias, eps=1e-5)
        token_means.append(states.mean(dim=0))
        scores = states @ query / math.sqrt(states.shape[-1])
        attention = torch.softmax(scores, dim=0)
        pooled = attention @ states
        attention_pools.append(pooled)
        hidden = functional.gelu(functional.linear(pooled, first_weight, first_bias))
        raw = functional.linear(hidden, final_weight, final_bias)
        coefficients.append(2.0 * torch.tanh(raw / 2.0))
        entropy = -(attention * attention.clamp_min(1e-30).log()).sum()
        normalized_entropy = entropy / math.log(max(2, attention.numel()))
        attention_entropies.append(float(normalized_entropy))
        attention_maxima.append(float(attention.max()))
    attention_pool_tensor = torch.stack(attention_pools)
    coefficient_tensor = torch.stack(coefficients)
    latent_codes = 80.0 * coefficient_tensor @ unit_basis
    zero_hidden = functional.gelu(first_bias)
    zero_raw = functional.linear(zero_hidden, final_weight, final_bias)
    zero_coefficients = 2.0 * torch.tanh(zero_raw / 2.0)
    residual_from_zero = coefficient_tensor - zero_coefficients.unsqueeze(0)
    coefficient_norms = coefficient_tensor.double().norm(dim=1)
    residual_norms = residual_from_zero.double().norm(dim=1)
    zero_cosines = functional.cosine_similarity(
        coefficient_tensor.double(), zero_coefficients.double().unsqueeze(0), dim=1
    )
    common_pool = attention_pool_tensor.mean(dim=0)
    common_hidden = functional.gelu(functional.linear(common_pool, first_weight, first_bias))
    common_raw = functional.linear(common_hidden, final_weight, final_bias)
    common_coefficients = 2.0 * torch.tanh(common_raw / 2.0)
    residual_from_common = coefficient_tensor - common_coefficients.unsqueeze(0)
    common_residual_norms = residual_from_common.double().norm(dim=1)
    common_cosines = functional.cosine_similarity(
        coefficient_tensor.double(), common_coefficients.double().unsqueeze(0), dim=1
    )
    return (
        {
            "uniform_token_mean": torch.stack(token_means),
            "learned_attention_pool": attention_pool_tensor,
            "coefficients": coefficient_tensor,
            "latent_delta_flat": latent_codes,
        },
        {
            "normalized_entropy_mean": sum(attention_entropies) / len(attention_entropies),
            "normalized_entropy_min": min(attention_entropies),
            "max_weight_mean": sum(attention_maxima) / len(attention_maxima),
            "max_weight_max": max(attention_maxima),
            "attention_query_norm": float(query.double().norm()),
        },
        {
            "zero_input_coefficient_norm": float(zero_coefficients.double().norm()),
            "event_coefficient_norm_mean": float(coefficient_norms.mean()),
            "event_residual_from_zero_norm_mean": float(residual_norms.mean()),
            "event_residual_from_zero_norm_max": float(residual_norms.max()),
            "event_residual_to_total_norm_ratio_mean": float(
                (residual_norms / coefficient_norms.clamp_min(1e-12)).mean()
            ),
            "event_to_zero_coefficient_cosine_mean": float(zero_cosines.mean()),
            "event_to_zero_coefficient_cosine_min": float(zero_cosines.min()),
            "common_event_coefficient_norm": float(common_coefficients.double().norm()),
            "event_residual_from_common_norm_mean": float(common_residual_norms.mean()),
            "event_residual_from_common_norm_max": float(common_residual_norms.max()),
            "event_residual_from_common_to_total_norm_ratio_mean": float(
                (common_residual_norms / coefficient_norms.clamp_min(1e-12)).mean()
            ),
            "event_to_common_coefficient_cosine_mean": float(common_cosines.mean()),
            "event_to_common_coefficient_cosine_min": float(common_cosines.min()),
        },
    )


def _pairwise_summary(values: Tensor, label_values: Sequence[str]) -> dict[str, float | int | None]:
    values = values.float()
    if values.ndim != 2 or len(values) != len(label_values):
        raise ValueError("R12 collapse audit representation shape drifted.")
    distances = torch.pdist(values.double())
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    norms = values.double().norm(dim=1)
    within = []
    between = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            destination = within if label_values[left] == label_values[right] else between
            destination.append(float((values[left].double() - values[right].double()).norm()))
    return {
        "items": len(values),
        "dimensions": values.shape[1],
        "mean_vector_norm": float(norms.mean()),
        "centered_rms_norm": float(centered.square().sum(dim=1).mean().sqrt()),
        "mean_pairwise_l2": float(distances.mean()) if len(distances) else 0.0,
        "median_pairwise_l2": float(distances.median()) if len(distances) else 0.0,
        "maximum_pairwise_l2": float(distances.max()) if len(distances) else 0.0,
        "same_target_mean_l2": sum(within) / len(within) if within else None,
        "different_target_mean_l2": sum(between) / len(between) if between else None,
        "same_to_different_distance_ratio": (
            (sum(within) / len(within)) / (sum(between) / len(between))
            if within and between and sum(between) > 0.0
            else None
        ),
    }


def _ridge_probe(
    values: Tensor,
    segment_ids: Sequence[str],
    labels: Mapping[str, str],
    split_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    index = {segment_id: position for position, segment_id in enumerate(segment_ids)}
    classes = sorted({labels[segment_id] for segment_id in split_ids["train"]})
    class_index = {value: position for position, value in enumerate(classes)}
    train_indices = torch.tensor([index[value] for value in split_ids["train"]])
    train = values[train_indices].double()
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train = (train - mean) / scale
    targets = torch.zeros(len(train), len(classes), dtype=torch.float64)
    for row, segment_id in enumerate(split_ids["train"]):
        targets[row, class_index[labels[segment_id]]] = 1.0
    gram = train @ train.T
    ridge = max(1e-9, 1e-3 * float(torch.diag(gram).mean()))
    alpha = torch.linalg.solve(gram + ridge * torch.eye(len(train), dtype=torch.float64), targets)
    result = {"classes": len(classes), "ridge": ridge, "accuracy": {}}
    for split in SPLITS:
        rows = list(split_ids[split])
        selected = values[torch.tensor([index[value] for value in rows])].double()
        selected = (selected - mean) / scale
        scores = selected @ train.T @ alpha
        prediction = scores.argmax(dim=1).tolist()
        correct = sum(pred == class_index[labels[segment_id]] for pred, segment_id in zip(prediction, rows))
        result["accuracy"][split] = correct / len(rows)
    return result


def _nearest_centroid_probe(
    values: Tensor,
    segment_ids: Sequence[str],
    labels: Mapping[str, str],
    split_ids: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    index = {segment_id: position for position, segment_id in enumerate(segment_ids)}
    classes = sorted({labels[segment_id] for segment_id in split_ids["train"]})
    normalized = functional.normalize(values.double(), dim=1)
    centroids = []
    for target in classes:
        rows = [index[value] for value in split_ids["train"] if labels[value] == target]
        centroids.append(functional.normalize(normalized[rows].mean(dim=0), dim=0))
    centroid_tensor = torch.stack(centroids)
    result = {}
    for split in SPLITS:
        rows = list(split_ids[split])
        selected = normalized[torch.tensor([index[value] for value in rows])]
        prediction = (selected @ centroid_tensor.T).argmax(dim=1).tolist()
        correct = sum(classes[pred] == labels[value] for pred, value in zip(prediction, rows))
        result[split] = correct / len(rows)
    return result


def analyze(
    *,
    summary_path: Path,
    checkpoint_path: Path,
    embedding_cache_path: Path,
    micro_metrics_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    summary, state, cache = _validate_sources(
        summary_path, checkpoint_path, embedding_cache_path, micro_metrics_path
    )
    labels, split_ids = _labels(summary, micro_metrics_path, cache)
    segment_ids = sorted(cache)
    representations, attention, coefficient_anchor = _writer_representations(
        state, cache, segment_ids
    )
    representation_audit = {}
    for name, values in representations.items():
        split_diversity = {}
        index = {segment_id: position for position, segment_id in enumerate(segment_ids)}
        for split in SPLITS:
            ids = split_ids[split]
            selected = values[torch.tensor([index[value] for value in ids])]
            split_diversity[split] = _pairwise_summary(selected, [labels[value] for value in ids])
        representation_audit[name] = {
            "diversity": split_diversity,
            "ridge_target_value_probe": _ridge_probe(
                values, segment_ids, labels, split_ids
            ),
            "nearest_centroid_target_value_accuracy": _nearest_centroid_probe(
                values, segment_ids, labels, split_ids
            ),
        }
    result = {
        "schema": "vision_memory.r12-shared-writer-collapse-audit.v1",
        "status": "completed",
        "posthoc_diagnostic_only": True,
        "formal_success_claim": False,
        "source_arm": summary["arm"],
        "source_outcome": {
            "decision": summary["decision"],
            "gates": summary["gates"],
        },
        "source_sha256": {
            "summary": _sha256(summary_path),
            "checkpoint": _sha256(checkpoint_path),
            "embedding_cache": _sha256(embedding_cache_path),
            "micro_metrics": _sha256(micro_metrics_path),
        },
        "split_counts": {split: len(split_ids[split]) for split in SPLITS},
        "attention": attention,
        "coefficient_anchor": coefficient_anchor,
        "representations": representation_audit,
    }
    _write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--micro-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        summary_path=args.summary,
        checkpoint_path=args.checkpoint,
        embedding_cache_path=args.embedding_cache,
        micro_metrics_path=args.micro_metrics,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
