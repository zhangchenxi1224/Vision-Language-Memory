from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl as read_synthetic_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.training import (  # noqa: E402
    StaticLearnedInitialImage,
    format_mcq_query,
    read_prefeval_adapted_jsonl,
    read_prefeval_supervised_jsonl,
    run_episode,
    select_curriculum_episodes,
)


FORMAL_OVERFIT_EPISODES = 64
FORMAL_OVERFIT_MAX_OPTIMIZER_STEPS = 2_000
FORMAL_OVERFIT_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal frozen-Qwen lightweight visual-memory training")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("synthetic", "prefeval-export", "prefeval-supervised"),
        default="synthetic",
    )
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("recurrent", "static-initial-image"), default="recurrent")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--state-channels", type=int, default=64)
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--learn-initial-state", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument(
        "--overfit-gate",
        action="store_true",
        help="Fail unless recurrent updater reaches --overfit-threshold on exactly --overfit-episodes training episodes.",
    )
    parser.add_argument("--overfit-episodes", type=int, default=64)
    parser.add_argument("--overfit-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noop-policy", choices=("update", "skip"), default="update")
    parser.add_argument("--curriculum", choices=("full", "set-only"), default="full")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.strip()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file() or not marker.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Reader has no non-empty revision lock: {marker}")
    return marker.read_text(encoding="utf-8").strip()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def load_records(path: Path, dataset_format: str, *, role: str) -> list[Any]:
    if dataset_format == "synthetic":
        return list(read_synthetic_jsonl(path))
    expected_split = "adapt_train" if role == "train" else "adapt_dev"
    if dataset_format == "prefeval-export":
        return read_prefeval_adapted_jsonl(path, allowed_splits={expected_split})
    return read_prefeval_supervised_jsonl(path, allowed_splits={expected_split})


def turn_kind(turn: Mapping[str, Any] | Any) -> str:
    if isinstance(turn, Mapping):
        value = turn.get("kind", turn.get("type"))
    else:
        value = getattr(turn, "type", None)
    return str(getattr(value, "value", value))


def event_payload(turn: Mapping[str, Any] | Any) -> tuple[str, str | None]:
    if isinstance(turn, Mapping):
        text = turn.get("event_text")
        event_kind = turn.get("event_kind", turn.get("transition"))
    else:
        text = getattr(turn, "event_text", None)
        event_kind = getattr(turn, "event_kind", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Updater turn has no event_text.")
    event_kind = getattr(event_kind, "value", event_kind)
    return text, None if event_kind is None else str(event_kind)


def query_payload(
    turn: Mapping[str, Any] | Any,
) -> tuple[str, tuple[str, ...], int, str | None]:
    nested = turn.get("query") if isinstance(turn, Mapping) else getattr(turn, "query", None)
    source = nested if nested is not None else turn
    if isinstance(source, Mapping):
        text = source.get("text", source.get("query_text"))
        choices = source.get("choices")
        target_index = source.get("target_index")
        comparison_id = source.get("comparison_id")
    else:
        text = getattr(source, "text", None)
        choices = getattr(source, "choices", None)
        target_index = getattr(source, "target_index", None)
        comparison_id = getattr(source, "comparison_id", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Reader turn has no query text.")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or len(choices) != 4:
        raise ValueError("Reader turn must contain four choices.")
    if not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("Reader turn has no valid target_index.")
    if comparison_id is not None and (not isinstance(comparison_id, str) or not comparison_id):
        raise ValueError("query comparison_id must be a non-empty string when provided.")
    return text, tuple(str(choice) for choice in choices), target_index, comparison_id


def episode_value(episode: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return episode.get(name, default) if isinstance(episode, Mapping) else getattr(episode, name, default)


def training_subset_audit(episodes: Sequence[Any]) -> dict[str, Any]:
    """Fingerprint the exact ordered episodes consumed by the optimizer."""

    episode_ids: list[str] = []
    for episode in episodes:
        episode_id = episode_value(episode, "episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Every selected training episode requires a non-empty episode_id.")
        episode_ids.append(episode_id)
    serialized = "\n".join(episode_ids).encode("utf-8")
    return {
        "count": len(episode_ids),
        "ordered_episode_ids_sha256": hashlib.sha256(serialized).hexdigest(),
        "hash_serialization": "UTF-8 episode IDs joined by LF with no trailing LF",
        "ordered_episode_ids": episode_ids,
    }


def episode_pattern(episode: Mapping[str, Any] | Any) -> str:
    template_id = episode_value(episode, "template_id")
    if not isinstance(template_id, str):
        return "unknown"
    match = re.search(r"(?:^|-)pattern-(\d+)(?:-|$)", template_id)
    return f"pattern_{match.group(1)}" if match else "unknown"


def tensor_spatial_diagnostics(
    tensor: torch.Tensor,
    *,
    value_bounds: tuple[float, float],
) -> dict[str, float]:
    """Summarize spatial collapse and boundary saturation without retaining tensors."""

    if tensor.ndim < 2 or tensor.numel() == 0:
        raise ValueError("Diagnostic tensors must have non-empty spatial dimensions.")
    values = tensor.detach().to(dtype=torch.float32)
    finite = torch.isfinite(values)
    finite_fraction = float(finite.float().mean().item())
    if finite_fraction != 1.0:
        raise RuntimeError(f"Non-finite evaluation tensor; finite_fraction={finite_fraction:.6f}")

    height, width = values.shape[-2:]
    center_height = max(1, height // 2)
    center_width = max(1, width // 2)
    top = (height - center_height) // 2
    left = (width - center_width) // 2
    center = values[..., top : top + center_height, left : left + center_width]
    spatial = values.reshape(-1, height * width)
    center_spatial = center.reshape(-1, center_height * center_width)

    lower, upper = value_bounds
    if not lower < upper:
        raise ValueError("value_bounds must be strictly increasing.")
    margin = 0.01 * (upper - lower)
    saturation = (values <= lower + margin) | (values >= upper - margin)
    minimum = float(values.min().item())
    maximum = float(values.max().item())
    return {
        "value_mean": float(values.mean().item()),
        "value_std": float(values.std(correction=0).item()),
        "value_min": minimum,
        "value_max": maximum,
        "dynamic_range": maximum - minimum,
        "spatial_std_mean": float(spatial.std(dim=1, correction=0).mean().item()),
        "center_spatial_std_mean": float(center_spatial.std(dim=1, correction=0).mean().item()),
        "saturation_fraction": float(saturation.float().mean().item()),
        "finite_fraction": finite_fraction,
    }


class _PairwiseMoments:
    def __init__(self) -> None:
        self.count = 0
        self.shape: tuple[int, ...] | None = None
        self.sum_vector: torch.Tensor | None = None
        self.sum_squared_norm = 0.0

    def add(self, tensor: torch.Tensor) -> None:
        detached = tensor.detach().to(device="cpu", dtype=torch.float64)
        shape = tuple(detached.shape)
        if self.shape is None:
            self.shape = shape
        elif shape != self.shape:
            raise ValueError(f"Pairwise diagnostic shape changed from {self.shape} to {shape}.")
        vector = detached.reshape(-1)
        if self.sum_vector is None:
            self.sum_vector = torch.zeros_like(vector)
        self.sum_vector.add_(vector)
        self.sum_squared_norm += float(torch.dot(vector, vector).item())
        self.count += 1

    @property
    def pair_count(self) -> int:
        return self.count * (self.count - 1) // 2

    def squared_difference_sum(self) -> float:
        if self.count < 2 or self.sum_vector is None:
            return 0.0
        value = self.count * self.sum_squared_norm - float(torch.dot(self.sum_vector, self.sum_vector).item())
        return max(0.0, value)


def _distance_summary(
    *,
    squared_difference_sum: float,
    pair_count: int,
    element_count: int,
) -> dict[str, float | int | None]:
    if pair_count == 0 or element_count == 0:
        return {
            "pair_count": pair_count,
            "mean_squared_element_distance": None,
            "rms_element_distance": None,
        }
    mean_squared = squared_difference_sum / (pair_count * element_count)
    return {
        "pair_count": pair_count,
        "mean_squared_element_distance": mean_squared,
        "rms_element_distance": math.sqrt(max(0.0, mean_squared)),
    }


class PairwiseTensorDiagnostics:
    """Exact all-pair RMS distances from sufficient statistics, grouped by target."""

    def __init__(self) -> None:
        self.all = _PairwiseMoments()
        self.by_target: dict[int, _PairwiseMoments] = {}

    def add(self, tensor: torch.Tensor, *, target_index: int) -> None:
        self.all.add(tensor)
        self.by_target.setdefault(target_index, _PairwiseMoments()).add(tensor)

    def summary(self) -> dict[str, Any]:
        element_count = 0 if self.all.sum_vector is None else self.all.sum_vector.numel()
        all_squared = self.all.squared_difference_sum()
        same_target_squared = sum(moments.squared_difference_sum() for moments in self.by_target.values())
        same_target_pairs = sum(moments.pair_count for moments in self.by_target.values())
        different_target_squared = max(0.0, all_squared - same_target_squared)
        different_target_pairs = self.all.pair_count - same_target_pairs
        return {
            "query_count": self.all.count,
            "tensor_shape": list(self.all.shape) if self.all.shape is not None else None,
            "all_pairs": _distance_summary(
                squared_difference_sum=all_squared,
                pair_count=self.all.pair_count,
                element_count=element_count,
            ),
            "same_target_pairs": _distance_summary(
                squared_difference_sum=same_target_squared,
                pair_count=same_target_pairs,
                element_count=element_count,
            ),
            "different_target_pairs": _distance_summary(
                squared_difference_sum=different_target_squared,
                pair_count=different_target_pairs,
                element_count=element_count,
            ),
        }


class _ScalarMoments:
    """Streaming scalar min/mean/max used by lightweight dynamics diagnostics."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite lightweight dynamics statistic: {value}")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": self.count,
            "min": self.minimum,
            "mean": self.total / self.count,
            "max": self.maximum,
        }


class LightweightDynamicsDiagnostics:
    """Aggregate ConvGRU input, gate, and output health without retaining tensors."""

    def __init__(self) -> None:
        self.updater_calls = 0
        self.conditioned_input_rms = _ScalarMoments()
        self.conditioned_input_absolute_max = _ScalarMoments()
        self.cell_output_absolute_max = _ScalarMoments()
        self.reset_elements = 0
        self.reset_below = 0
        self.reset_above = 0
        self.update_elements = 0
        self.update_below = 0
        self.update_above = 0
        self.cell_output_elements = 0
        self.cell_output_outside_nominal = 0

    @staticmethod
    def _gate_saturation_summary(*, elements: int, below: int, above: int) -> dict[str, float | int | None]:
        if elements == 0:
            return {
                "element_count": 0,
                "below_0_01_fraction": None,
                "above_0_99_fraction": None,
                "saturated_fraction": None,
            }
        return {
            "element_count": elements,
            "below_0_01_fraction": below / elements,
            "above_0_99_fraction": above / elements,
            "saturated_fraction": (below + above) / elements,
        }

    def _capture_cell(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if len(inputs) != 2 or not all(isinstance(value, torch.Tensor) for value in inputs):
            raise RuntimeError("ConvGRU dynamics hook expected conditioned input and hidden tensors.")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("ConvGRU dynamics hook expected a tensor output.")
        conditioned_input, hidden = inputs
        gates = getattr(module, "gates", None)
        hidden_channels = getattr(module, "hidden_channels", None)
        if not isinstance(gates, torch.nn.Module) or not isinstance(hidden_channels, int):
            raise RuntimeError("ConvGRU dynamics hook requires gates and hidden_channels attributes.")

        with torch.no_grad():
            conditioned = conditioned_input.detach().to(dtype=torch.float32)
            cell_output = output.detach().to(dtype=torch.float32)
            gate_logits = gates(torch.cat([conditioned_input.detach(), hidden.detach()], dim=1))
            reset, update = gate_logits.split(hidden_channels, dim=1)
            reset = torch.sigmoid(reset).to(dtype=torch.float32)
            update = torch.sigmoid(update).to(dtype=torch.float32)

            self.conditioned_input_rms.add(float(conditioned.square().mean().sqrt().item()))
            self.conditioned_input_absolute_max.add(float(conditioned.abs().max().item()))
            self.cell_output_absolute_max.add(float(cell_output.abs().max().item()))
            self.reset_elements += reset.numel()
            self.reset_below += int((reset < 0.01).sum().item())
            self.reset_above += int((reset > 0.99).sum().item())
            self.update_elements += update.numel()
            self.update_below += int((update < 0.01).sum().item())
            self.update_above += int((update > 0.99).sum().item())
            self.cell_output_elements += cell_output.numel()
            self.cell_output_outside_nominal += int(((cell_output < -0.98) | (cell_output > 0.98)).sum().item())
            self.updater_calls += 1

    @contextmanager
    def capture(self, cell: torch.nn.Module) -> Iterator[None]:
        """Install one temporary hook and guarantee removal, including on evaluation errors."""

        handle = cell.register_forward_hook(self._capture_cell)
        try:
            yield
        finally:
            handle.remove()

    def summary(self) -> dict[str, Any]:
        outside_fraction = (
            self.cell_output_outside_nominal / self.cell_output_elements if self.cell_output_elements else None
        )
        return {
            "schema_version": "vision_memory.lightweight.dynamics.v1",
            "definitions": {
                "conditioned_cell_input": "the FiLM-conditioned event map passed as the first ConvGRU cell input",
                "rms": "per-updater-call root mean square over every conditioned-input element",
                "absolute_max": "per-updater-call maximum absolute tensor value",
                "gate_saturation": "element-weighted fractions strictly below 0.01 or strictly above 0.99",
                "cell_output_outside_nominal": "element-weighted fraction outside the closed interval [-0.98, 0.98]",
            },
            "updater_calls": self.updater_calls,
            "conditioned_cell_input": {
                "rms": self.conditioned_input_rms.summary(),
                "absolute_max": self.conditioned_input_absolute_max.summary(),
            },
            "reset_gate_saturation": self._gate_saturation_summary(
                elements=self.reset_elements,
                below=self.reset_below,
                above=self.reset_above,
            ),
            "update_gate_saturation": self._gate_saturation_summary(
                elements=self.update_elements,
                below=self.update_below,
                above=self.update_above,
            ),
            "cell_output": {
                "absolute_max": self.cell_output_absolute_max.summary(),
                "element_count": self.cell_output_elements,
                "outside_nominal_fraction": outside_fraction,
            },
        }


def _parameter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    component_norms: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        component_norms.append(torch.linalg.vector_norm(gradient.to(dtype=torch.float32)))
    if not component_norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(component_norms)).item())


def gradient_norms_before_clip(model: torch.nn.Module) -> dict[str, float]:
    """Return disjoint pre-clip L2 gradient norms for auditable updater modules."""

    if not isinstance(model, LightweightVisualUpdater):
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        return {"model": _parameter_gradient_norm(parameters)}

    grouped_modules = (
        ("event_encoder", model.event_encoder),
        ("event_projection", model.event_projection),
        ("event_spatial_projection", model.event_spatial_projection),
        ("film", model.film),
        ("cell", model.cell),
        ("rgb_head", model.rgb_head),
    )
    norms: dict[str, float] = {}
    assigned_parameter_ids: set[int] = set()
    for name, module in grouped_modules:
        parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
        parameter_ids = {id(parameter) for parameter in parameters}
        overlap = assigned_parameter_ids.intersection(parameter_ids)
        if overlap:
            raise RuntimeError(f"Gradient diagnostic module groups overlap at {name!r}.")
        assigned_parameter_ids.update(parameter_ids)
        norms[name] = _parameter_gradient_norm(parameters)

    residual = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned_parameter_ids
    )
    if residual:
        norms["other_trainable_parameters"] = _parameter_gradient_norm(residual)
    return norms


def clip_gradients_with_diagnostics(
    *,
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    max_norm: float,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Clip gradients once and return the exact JSON-ready pre-clip audit fields."""

    if max_norm <= 0 or epsilon <= 0:
        raise ValueError("max_norm and epsilon must be positive.")
    module_norms = gradient_norms_before_clip(model)
    global_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    global_norm_value = float(global_norm.item())
    if not math.isfinite(global_norm_value) or global_norm_value <= 0:
        raise RuntimeError(f"Invalid lightweight gradient norm: {global_norm_value}")
    return {
        "gradient_norm_before_clip": global_norm_value,
        "module_gradient_norms_before_clip": module_norms,
        "gradient_clipping_factor": min(1.0, max_norm / (global_norm_value + epsilon)),
        "gradient_clipping_epsilon": epsilon,
    }


def _grouped_accuracy(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for record in records:
        label = str(record.get(key) or "unknown")
        counts = groups.setdefault(label, [0, 0])
        counts[0] += int(bool(record["correct"]))
        counts[1] += 1
    return {
        label: {"correct": counts[0], "count": counts[1], "accuracy": counts[0] / counts[1]}
        for label, counts in sorted(groups.items())
    }


def _aggregate_tensor_diagnostics(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    metric_names = tuple(records[0][key])
    result: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(record[key][metric_name]) for record in records]
        result[metric_name] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
    return result


def evaluation_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    state_pairwise: PairwiseTensorDiagnostics,
    image_pairwise: PairwiseTensorDiagnostics,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an evaluation without query records.")
    return {
        "schema_version": "vision_memory.lightweight.evaluation_diagnostics.v1",
        "definitions": {
            "spatial_std_mean": "mean population std over HxW, computed independently per leading slice",
            "center_spatial_std_mean": "same statistic over the central floor(H/2) x floor(W/2) crop",
            "state_saturation": "fraction outside the inner 98% of nominal [-1, 1]",
            "image_saturation": "fraction outside the inner 98% of nominal [0, 1]",
            "pairwise_distance": "exact RMS per tensor element over unordered query pairs",
            "target_grouping": "post-hoc diagnostic grouping only; targets never enter model inputs",
        },
        "query_count": len(records),
        "accuracy": sum(int(bool(record["correct"])) for record in records) / len(records),
        "accuracy_by_pattern": _grouped_accuracy(records, "pattern"),
        "accuracy_by_subtype": _grouped_accuracy(records, "subtype"),
        "accuracy_by_turn_type": _grouped_accuracy(records, "turn_type"),
        "accuracy_by_distractor_variant": _grouped_accuracy(records, "distractor_variant"),
        "state_query_statistics": _aggregate_tensor_diagnostics(records, "state_diagnostics"),
        "image_query_statistics": _aggregate_tensor_diagnostics(records, "image_diagnostics"),
        "pairwise_state_distances": state_pairwise.summary(),
        "pairwise_image_distances": image_pairwise.summary(),
    }


def overfit_evaluation_paths(output_dir: Path, optimizer_step: int) -> tuple[Path, Path]:
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative.")
    stem = output_dir / "overfit_evaluations" / f"step_{optimizer_step:07d}"
    return stem.with_name(stem.name + "_predictions.jsonl"), stem.with_name(stem.name + "_diagnostics.json")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_vram_gib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0


def _normalized_value(value: Any) -> str | None:
    value = getattr(value, "value", value)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def validate_overfit_gate_configuration(args: argparse.Namespace) -> None:
    """Prevent a formal gate run from silently weakening preregistered constants."""

    if not args.overfit_gate:
        return
    if args.method != "recurrent":
        raise SystemExit("--overfit-gate is defined only for the recurrent lightweight updater.")
    if args.dataset_format != "synthetic":
        raise SystemExit("--overfit-gate requires the preregistered synthetic episode format.")
    if args.curriculum != "full":
        raise SystemExit("--overfit-gate requires the full transition curriculum.")
    if args.noop_policy != "update":
        raise SystemExit("--overfit-gate requires distractor/no-op events to enter the updater.")
    if args.learn_initial_state:
        raise SystemExit("--overfit-gate requires the fixed zero lightweight hidden initial state.")
    if args.overfit_episodes != FORMAL_OVERFIT_EPISODES:
        raise SystemExit(
            f"--overfit-gate requires exactly {FORMAL_OVERFIT_EPISODES} episodes; got {args.overfit_episodes}."
        )
    if args.max_optimizer_steps != FORMAL_OVERFIT_MAX_OPTIMIZER_STEPS:
        raise SystemExit(
            "--overfit-gate requires an optimizer-step budget of exactly "
            f"{FORMAL_OVERFIT_MAX_OPTIMIZER_STEPS}; got {args.max_optimizer_steps}."
        )
    if args.overfit_threshold != FORMAL_OVERFIT_THRESHOLD:
        raise SystemExit(
            f"--overfit-gate requires accuracy threshold {FORMAL_OVERFIT_THRESHOLD:.2f}; got {args.overfit_threshold}."
        )


def training_budget_open(
    args: argparse.Namespace,
    *,
    epoch: int,
    optimizer_step: int,
    gate_passed: bool,
) -> bool:
    """Return whether another epoch may start under ordinary or step-driven gate training."""

    if gate_passed:
        return False
    if args.max_optimizer_steps is not None and optimizer_step >= args.max_optimizer_steps:
        return False
    if args.overfit_gate:
        # The formal gate is optimizer-step bounded. ``--epochs`` is deliberately not
        # a second, much smaller hidden budget (64 / accumulation * two epochs was only
        # 16 optimizer steps with the defaults).
        return True
    return epoch < args.epochs


def _query_comparison_ids(episode: Mapping[str, Any] | Any) -> tuple[str, ...]:
    result: list[str] = []
    turns = episode_value(episode, "turns")
    for turn in turns:
        if turn_kind(turn) not in {"query", "mixed"}:
            continue
        _text, _choices, _target_index, comparison_id = query_payload(turn)
        if comparison_id is None:
            raise SystemExit(
                f"Overfit gate episode {episode_value(episode, 'episode_id')!r} has a query "
                "without schema-v2 comparison_id metadata."
            )
        result.append(comparison_id)
    return tuple(result)


def validate_overfit_gate_episodes(episodes: Sequence[Any]) -> None:
    """Require 64 unique, reciprocal clean/distractor episodes from schema v2."""

    if len(episodes) != FORMAL_OVERFIT_EPISODES:
        raise SystemExit(
            f"Overfit gate requires exactly {FORMAL_OVERFIT_EPISODES} selected training episodes; got {len(episodes)}."
        )
    by_id = {str(episode_value(episode, "episode_id")): episode for episode in episodes}
    if len(by_id) != len(episodes) or "" in by_id:
        raise SystemExit("Overfit gate requires 64 unique, non-empty episode_id values.")

    variants = {"clean": 0, "distractor": 0}
    for episode_id, episode in by_id.items():
        variant = _normalized_value(episode_value(episode, "distractor_variant"))
        if variant not in variants:
            raise SystemExit(
                f"Overfit gate episode {episode_id!r} must be a paired clean/distractor member; "
                f"got distractor_variant={variant!r}."
            )
        variants[variant] += 1
        pair_id = _normalized_value(episode_value(episode, "distractor_pair_id"))
        counterpart_id = _normalized_value(episode_value(episode, "distractor_episode_id"))
        if pair_id is None or counterpart_id is None or counterpart_id not in by_id:
            raise SystemExit(f"Overfit gate episode {episode_id!r} has an incomplete in-subset distractor pair.")
        counterpart = by_id[counterpart_id]
        counterpart_variant = _normalized_value(episode_value(counterpart, "distractor_variant"))
        expected_variant = "distractor" if variant == "clean" else "clean"
        if counterpart_variant != expected_variant:
            raise SystemExit(f"Overfit gate pair {pair_id!r} does not contain one clean and one distractor member.")
        if _normalized_value(episode_value(counterpart, "distractor_pair_id")) != pair_id:
            raise SystemExit(f"Overfit gate pair {pair_id!r} has inconsistent pair IDs.")
        if _normalized_value(episode_value(counterpart, "distractor_episode_id")) != episode_id:
            raise SystemExit(f"Overfit gate pair {pair_id!r} is not reciprocally linked.")
        if _query_comparison_ids(counterpart) != _query_comparison_ids(episode):
            raise SystemExit(f"Overfit gate pair {pair_id!r} has mismatched query comparison IDs.")

    expected_per_variant = FORMAL_OVERFIT_EPISODES // 2
    if variants != {"clean": expected_per_variant, "distractor": expected_per_variant}:
        raise SystemExit(f"Overfit gate requires 32 clean and 32 distractor episodes; got {variants}.")


def evaluate_accuracy(
    *,
    episodes: Sequence[Any],
    model: torch.nn.Module,
    reader: Any,
    processor: Any,
    device: torch.device,
    noop_policy: str,
    predictions_path: Path | None = None,
    diagnostics_path: Path | None = None,
    method: str,
    seed: int,
) -> float:
    model.eval()
    records: list[dict[str, Any]] = []
    collect_diagnostics = diagnostics_path is not None
    state_pairwise = PairwiseTensorDiagnostics()
    image_pairwise = PairwiseTensorDiagnostics()
    dynamics = (
        LightweightDynamicsDiagnostics()
        if collect_diagnostics and isinstance(model, LightweightVisualUpdater)
        else None
    )
    dynamics_context = dynamics.capture(model.cell) if dynamics is not None else nullcontext()
    correct = 0
    total = 0
    with dynamics_context, torch.no_grad():
        for episode in episodes:
            state = model.initial_state(batch_size=1, device=device, dtype=torch.float32)
            turns = episode_value(episode, "turns")
            query_index = 0
            last_transition = "none"
            updater_calls_since_query = 0
            noop_events_since_query = 0
            noop_events_applied_since_query = 0
            event_latency_since_query = 0.0
            for turn_id, turn in enumerate(turns):
                kind = turn_kind(turn)
                if kind in {"event", "mixed"}:
                    text, event_kind = event_payload(turn)
                    if noop_policy == "skip" and event_kind is None:
                        raise ValueError("skip-noop evaluation requires every event_kind label.")
                    if not (noop_policy == "skip" and event_kind == "noop"):
                        synchronize(device)
                        event_started = time.monotonic()
                        state = model.update(state, text)
                        synchronize(device)
                        event_latency_since_query += time.monotonic() - event_started
                        updater_calls_since_query += 1
                        noop_events_applied_since_query += int(event_kind == "noop")
                    noop_events_since_query += int(event_kind == "noop")
                    last_transition = event_kind or "unknown"
                if kind in {"query", "mixed"}:
                    query, choices, target_index, comparison_id = query_payload(turn)
                    image = model.render(state)[0]
                    synchronize(device)
                    started = time.monotonic()
                    score = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(query, choices),
                        choices=choices,
                        device=device,
                    )
                    synchronize(device)
                    query_latency = time.monotonic() - started
                    is_correct = score.predicted_index == target_index
                    correct += int(is_correct)
                    total += 1
                    choice_mean_nll = [float(value) for value in score.mean_nll]
                    target_mean_nll = choice_mean_nll[target_index]
                    best_other_mean_nll = min(
                        value for index, value in enumerate(choice_mean_nll) if index != target_index
                    )
                    record = {
                        "episode_id": str(episode_value(episode, "episode_id")),
                        "query_id": f"{episode_value(episode, 'episode_id')}:q{query_index}",
                        "query_ordinal": query_index,
                        "turn_id": turn_id,
                        "turn_type": kind,
                        "method": method,
                        "seed": seed,
                        "condition": "standard",
                        "prediction_index": score.predicted_index,
                        "target_index": target_index,
                        "correct": is_correct,
                        "choice_mean_nll": choice_mean_nll,
                        "target_mean_nll": target_mean_nll,
                        "predicted_mean_nll": choice_mean_nll[score.predicted_index],
                        "target_nll_margin": best_other_mean_nll - target_mean_nll,
                        "split": episode_value(episode, "split"),
                        "topic": episode_value(episode, "topic"),
                        "template_id": episode_value(episode, "template_id"),
                        "pattern": episode_pattern(episode),
                        "form": episode_value(episode, "form", last_transition),
                        "protocol": episode_value(episode, "protocol"),
                        "forced_write_k": episode_value(episode, "forced_write_k"),
                        "ood_group": episode_value(episode, "ood_group"),
                        "subtype": last_transition,
                        "base_pair_id": episode_value(episode, "base_pair_id"),
                        "pair_id": episode_value(episode, "pair_id"),
                        "counterfactual_pair_id": episode_value(
                            episode, "pair_id", episode_value(episode, "base_pair_id")
                        ),
                        "semantic_counterfactual_pair_id": episode_value(episode, "pair_id"),
                        "counterfactual_episode_id": episode_value(episode, "counterfactual_episode_id"),
                        "distractor_pair_id": episode_value(episode, "distractor_pair_id"),
                        "distractor_episode_id": episode_value(episode, "distractor_episode_id"),
                        "distractor_variant": getattr(
                            episode_value(episode, "distractor_variant"),
                            "value",
                            episode_value(episode, "distractor_variant"),
                        ),
                        "query_comparison_id": comparison_id,
                        "noop_policy": "skip" if noop_policy == "skip" else "keep",
                        "updater_calls_since_query": updater_calls_since_query,
                        "noop_events_since_query": noop_events_since_query,
                        "noop_events_applied_since_query": noop_events_applied_since_query,
                        "event_latency_seconds": event_latency_since_query,
                        "query_latency_seconds": query_latency,
                        "latency_seconds": event_latency_since_query + query_latency,
                        "state_bytes": int(state.numel() * state.element_size()),
                        "peak_vram_gib": peak_vram_gib(device),
                        "peak_reader_vram_gib": peak_vram_gib(device),
                    }
                    if collect_diagnostics:
                        record["state_diagnostics"] = tensor_spatial_diagnostics(state, value_bounds=(-1.0, 1.0))
                        record["image_diagnostics"] = tensor_spatial_diagnostics(image, value_bounds=(0.0, 1.0))
                        state_pairwise.add(state, target_index=target_index)
                        image_pairwise.add(image, target_index=target_index)
                    records.append(record)
                    query_index += 1
                    updater_calls_since_query = 0
                    noop_events_since_query = 0
                    noop_events_applied_since_query = 0
                    event_latency_since_query = 0.0
    if total == 0:
        raise RuntimeError("Evaluation encountered no queries.")
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = evaluation_diagnostics(
            records,
            state_pairwise=state_pairwise,
            image_pairwise=image_pairwise,
        )
        if dynamics is not None:
            diagnostics["lightweight_dynamics"] = dynamics.summary()
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    model.train()
    return correct / total


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.method == "static-initial-image":
        if args.learn_initial_state:
            raise ValueError("static-initial-image is already learned; do not pass --learn-initial-state.")
        return StaticLearnedInitialImage(output_size=args.output_size)
    return LightweightVisualUpdater(
        state_channels=args.state_channels,
        state_size=args.state_size,
        output_size=args.output_size,
        learned_initial_state=args.learn_initial_state,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
    optimizer_step: int,
    best_dev_accuracy: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "manifest": dict(manifest),
            "model_config": dict(model_config),
            "optimizer_step": optimizer_step,
            "best_dev_accuracy": best_dev_accuracy,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    positive = {
        "output_size": args.output_size,
        "state_channels": args.state_channels,
        "state_size": args.state_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip,
        "eval_every": args.eval_every,
        "eval_limit": args.eval_limit,
        "overfit_episodes": args.overfit_episodes,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise SystemExit(f"Arguments must be positive: {invalid}")
    if args.max_train_episodes is not None and args.max_train_episodes <= 0:
        raise SystemExit("--max-train-episodes must be positive.")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise SystemExit("--max-optimizer-steps must be positive.")
    if not 0.0 < args.overfit_threshold <= 1.0:
        raise SystemExit("--overfit-threshold must be in (0, 1].")
    validate_overfit_gate_configuration(args)
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative.")
    if not torch.cuda.is_available():
        raise SystemExit("Formal lightweight training requires CUDA and the real frozen Qwen Reader.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Formal training refuses a non-empty --output-dir.")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("Formal training refuses a dirty worktree; --allow-dirty is debugging-only.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA.")
    set_seeds(args.seed)
    train_raw = load_records(args.train, args.dataset_format, role="train")
    dev = load_records(args.dev, args.dataset_format, role="dev")[: args.eval_limit]
    train, selection = select_curriculum_episodes(train_raw, curriculum=args.curriculum)
    if args.max_train_episodes is not None:
        train = train[: args.max_train_episodes]
    if args.overfit_gate:
        train = train[: args.overfit_episodes]
        validate_overfit_gate_episodes(train)
    if not train or not dev:
        raise SystemExit("Train/dev data must remain non-empty after filtering.")

    args.output_dir.mkdir(parents=True)
    reader_revision = locked_revision(args.reader)
    train_subset = training_subset_audit(train)
    manifest = {
        "schema_version": "vision_memory.lightweight.training.v1",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "reader_revision": reader_revision,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "arguments": serializable_args(args),
        "training_budget": {
            "mode": "optimizer_steps" if args.overfit_gate else "epochs",
            "max_optimizer_steps": args.max_optimizer_steps,
            "configured_epochs": args.epochs,
        },
        "overfit_gate_definition": (
            {
                "episodes": FORMAL_OVERFIT_EPISODES,
                "max_optimizer_steps": FORMAL_OVERFIT_MAX_OPTIMIZER_STEPS,
                "minimum_train_mcq_accuracy": FORMAL_OVERFIT_THRESHOLD,
                "distractor_policy": "update",
            }
            if args.overfit_gate
            else None
        ),
        "curriculum": {
            **selection.to_dict(),
            "selected_after_limit": len(train),
            "turns_rewritten": False,
        },
        "train_subset": train_subset,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=args.output_size * args.output_size,
        max_pixels=args.output_size * args.output_size,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("A tensor-native fast Qwen image processor is required.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    model = build_model(args).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Lightweight method has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    def update_fn(state, event_text, _episode_id, _turn_id):
        return model.update(state, event_text)

    def reader_loss(image, query, target):
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image[0],
            query=query,
            target=target,
            device=device,
            require_image_grad=True,
        )

    model_config = {
        "method": args.method,
        "output_size": args.output_size,
        "state_channels": args.state_channels,
        "state_size": args.state_size,
        "learn_initial_state": args.learn_initial_state,
    }
    optimizer_step = 0
    episodes_processed = 0
    completed_epochs = 0
    best_dev = -1.0
    best_train = -1.0
    overfit_gate_passed = False
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    stop_training = False
    epoch = 0
    while training_budget_open(
        args,
        epoch=epoch,
        optimizer_step=optimizer_step,
        gate_passed=overfit_gate_passed,
    ):
        order = list(range(len(train)))
        random.Random((args.seed << 16) ^ epoch).shuffle(order)
        for group_start in range(0, len(order), args.gradient_accumulation):
            group = order[group_start : group_start + args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            group_losses: list[float] = []
            for index in group:
                result = run_episode(
                    episode=train[index],
                    initial_state=model.initial_state(batch_size=1, device=device, dtype=torch.float32),
                    update_fn=update_fn,
                    decode_fn=model.render,
                    reader_loss_fn=reader_loss,
                    noop_policy=args.noop_policy,
                    collect_states=False,
                )
                (result.loss / len(group)).backward()
                assert_no_frozen_parameter_grads(reader, "Qwen Reader")
                group_losses.append(float(result.loss.item()))
                episodes_processed += 1
            gradient_diagnostics = clip_gradients_with_diagnostics(
                model=model,
                parameters=trainable,
                max_norm=args.gradient_clip,
            )
            optimizer.step()
            optimizer_step += 1
            with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "train",
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "loss": sum(group_losses) / len(group_losses),
                            **gradient_diagnostics,
                            "accumulated_episodes": len(group),
                            "episodes_processed": episodes_processed,
                        }
                    )
                    + "\n"
                )
            reached_step_limit = args.max_optimizer_steps is not None and optimizer_step >= args.max_optimizer_steps
            if optimizer_step % args.eval_every == 0 or reached_step_limit:
                if args.overfit_gate:
                    predictions_path, diagnostics_path = overfit_evaluation_paths(args.output_dir, optimizer_step)
                    train_accuracy = evaluate_accuracy(
                        episodes=train,
                        model=model,
                        reader=reader,
                        processor=processor,
                        device=device,
                        noop_policy=args.noop_policy,
                        predictions_path=predictions_path,
                        diagnostics_path=diagnostics_path,
                        method=args.method,
                        seed=args.seed,
                    )
                    best_train = max(best_train, train_accuracy)
                    overfit_gate_passed = train_accuracy >= args.overfit_threshold
                    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "kind": "overfit_train",
                                    "optimizer_step": optimizer_step,
                                    "accuracy": train_accuracy,
                                    "threshold": args.overfit_threshold,
                                    "passed": overfit_gate_passed,
                                    "predictions_path": str(predictions_path.resolve()),
                                    "diagnostics_path": str(diagnostics_path.resolve()),
                                }
                            )
                            + "\n"
                        )
                else:
                    accuracy = evaluate_accuracy(
                        episodes=dev,
                        model=model,
                        reader=reader,
                        processor=processor,
                        device=device,
                        noop_policy=args.noop_policy,
                        method=args.method,
                        seed=args.seed,
                    )
                    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "kind": "dev",
                                    "optimizer_step": optimizer_step,
                                    "accuracy": accuracy,
                                }
                            )
                            + "\n"
                        )
                    if accuracy > best_dev:
                        best_dev = accuracy
                        save_checkpoint(
                            args.output_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            manifest=manifest,
                            model_config=model_config,
                            optimizer_step=optimizer_step,
                            best_dev_accuracy=best_dev,
                        )
            if overfit_gate_passed or reached_step_limit:
                stop_training = True
                break
        completed_epochs += 1
        epoch += 1
        if stop_training:
            break

    final_accuracy = evaluate_accuracy(
        episodes=dev,
        model=model,
        reader=reader,
        processor=processor,
        device=device,
        noop_policy=args.noop_policy,
        predictions_path=args.output_dir / "dev_predictions.jsonl",
        diagnostics_path=args.output_dir / "dev_diagnostics.json",
        method=args.method,
        seed=args.seed,
    )
    if final_accuracy > best_dev:
        best_dev = final_accuracy
        save_checkpoint(
            args.output_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            model_config=model_config,
            optimizer_step=optimizer_step,
            best_dev_accuracy=best_dev,
        )
    final_train_accuracy: float | None = None
    if args.overfit_gate:
        final_train_accuracy = evaluate_accuracy(
            episodes=train,
            model=model,
            reader=reader,
            processor=processor,
            device=device,
            noop_policy=args.noop_policy,
            predictions_path=args.output_dir / "train_predictions.jsonl",
            diagnostics_path=args.output_dir / "train_diagnostics.json",
            method=args.method,
            seed=args.seed,
        )
        best_train = max(best_train, final_train_accuracy)
        if final_train_accuracy >= args.overfit_threshold:
            overfit_gate_passed = True
    if args.overfit_gate and overfit_gate_passed:
        save_checkpoint(
            args.output_dir / "gate.pt",
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            model_config=model_config,
            optimizer_step=optimizer_step,
            best_dev_accuracy=best_dev,
        )
    save_checkpoint(
        args.output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        model_config=model_config,
        optimizer_step=optimizer_step,
        best_dev_accuracy=best_dev,
    )
    summary = {
        "method": args.method,
        "optimizer_steps": optimizer_step,
        "optimizer_step_budget": args.max_optimizer_steps,
        "episodes_processed": episodes_processed,
        "completed_epochs": completed_epochs,
        "best_dev_accuracy": best_dev,
        "final_dev_accuracy": final_accuracy,
        "final_train_accuracy": final_train_accuracy,
        "best_train_accuracy": best_train if args.overfit_gate else None,
        "overfit_gate_enabled": args.overfit_gate,
        "overfit_gate_threshold": args.overfit_threshold if args.overfit_gate else None,
        "overfit_gate_passed": overfit_gate_passed if args.overfit_gate else None,
        "train_subset_count": train_subset["count"],
        "train_subset_ordered_episode_ids_sha256": train_subset["ordered_episode_ids_sha256"],
        "final_prediction_artifacts": {
            "dev_predictions": str((args.output_dir / "dev_predictions.jsonl").resolve()),
            "dev_diagnostics": str((args.output_dir / "dev_diagnostics.json").resolve()),
            "train_predictions": (
                str((args.output_dir / "train_predictions.jsonl").resolve()) if args.overfit_gate else None
            ),
            "train_diagnostics": (
                str((args.output_dir / "train_diagnostics.json").resolve()) if args.overfit_gate else None
            ),
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.overfit_gate and not overfit_gate_passed:
        raise RuntimeError(
            f"Lightweight overfit gate failed: best train accuracy {best_train:.4f} "
            f"< {args.overfit_threshold:.4f} within {optimizer_step} optimizer steps."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
