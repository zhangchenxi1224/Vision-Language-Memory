from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import Episode, read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import ChoiceScoreOutput, qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402

from scripts.data.qwen_sanity import (  # noqa: E402
    UniqueQuery,
    collect_unique_queries,
    locked_revision,
    sha256_file,
)
from scripts.train.lightweight_episode import training_subset_audit  # noqa: E402


CONTROL_CODE_COUNT = 4
TARGET_SUPERVISED_LABEL_LEAK = True
DIAGNOSTIC_SCOPE = "renderer_manifold_only"
DISCLAIMER = (
    "TARGET-SUPERVISED LABEL-LEAK DIAGNOSTIC ONLY: target_index selects one of four trainable "
    "hidden-state codes before the existing LightweightVisualUpdater rgb_head/render path. "
    "This only probes the renderer manifold; it is not a method, baseline, ablation, memory "
    "updater, generalization result, or publishable result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-selected hidden-state diagnostic for the lightweight renderer manifold"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Number of episodes selected from the ordered dataset prefix (default: 64)",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-channels", type=int, default=64)
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select_episode_prefix(dataset: Path, *, limit: int) -> tuple[list[Episode], dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    episodes = list(read_jsonl(dataset))[:limit]
    if not episodes:
        raise ValueError("The selected dataset prefix contains no episodes")
    return episodes, training_subset_audit(episodes)


def deterministic_query_order(query_count: int, *, steps: int, seed: int) -> list[int]:
    if query_count < 1 or steps < 1:
        raise ValueError("query_count and steps must be positive")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(query_count))
        random.Random((seed << 16) ^ epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def query_pattern(item: UniqueQuery) -> str:
    marker = "-pattern-"
    if marker not in item.template_id:
        return "unknown"
    suffix = item.template_id.split(marker, maxsplit=1)[1]
    number = suffix.split("-", maxsplit=1)[0]
    return f"pattern_{number}"


def grouped_accuracy(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        counts = groups[str(record[key])]
        counts[0] += int(bool(record["correct"]))
        counts[1] += 1
    return {
        name: {"correct": counts[0], "count": counts[1], "accuracy": counts[0] / counts[1]}
        for name, counts in sorted(groups.items())
    }


def renderer_control_exit_code(passed_threshold: bool) -> int:
    return 0 if passed_threshold else 3


class TargetSelectedRendererControl(nn.Module):
    """Four leaked-label state codes rendered only through the production RGB head."""

    def __init__(
        self,
        renderer: LightweightVisualUpdater,
        *,
        code_count: int = CONTROL_CODE_COUNT,
        initialization_std: float = 0.1,
        code_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if code_count != CONTROL_CODE_COUNT:
            raise ValueError(f"This four-choice diagnostic requires exactly {CONTROL_CODE_COUNT} codes")
        if initialization_std <= 0 or code_bound <= 0:
            raise ValueError("initialization_std and code_bound must be positive")
        self.renderer = renderer
        self.code_count = code_count
        self.code_bound = float(code_bound)

        self.renderer.requires_grad_(False)
        self.renderer.eval()
        self.renderer.rgb_head.requires_grad_(True)
        self.state_codes = nn.Parameter(
            torch.empty(code_count, renderer.state_channels, renderer.state_size, renderer.state_size)
        )
        nn.init.normal_(self.state_codes, mean=0.0, std=initialization_std)
        with torch.no_grad():
            self.state_codes.clamp_(-self.code_bound, self.code_bound)

        self._forbidden_call_counts = {"event_encoder": 0, "convgru": 0}
        self._forbidden_call_handles = (
            self.renderer.event_encoder.register_forward_pre_hook(self._forbid_event_encoder),
            self.renderer.cell.register_forward_pre_hook(self._forbid_convgru),
        )
        self.assert_parameter_contract()

    def _forbid_event_encoder(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._forbidden_call_counts["event_encoder"] += 1
        raise RuntimeError("Renderer diagnostic must not call event_encoder")

    def _forbid_convgru(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self._forbidden_call_counts["convgru"] += 1
        raise RuntimeError("Renderer diagnostic must not call ConvGRU")

    def allowed_trainable_parameter_names(self) -> set[str]:
        return {"state_codes"} | {
            f"renderer.rgb_head.{name}" for name, _parameter in self.renderer.rgb_head.named_parameters()
        }

    def trainable_parameter_names(self) -> set[str]:
        return {name for name, parameter in self.named_parameters() if parameter.requires_grad}

    def assert_parameter_contract(self) -> None:
        expected = self.allowed_trainable_parameter_names()
        actual = self.trainable_parameter_names()
        if actual != expected:
            raise RuntimeError(
                "Renderer-control trainable parameter contract failed: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
        expected_shape = (
            CONTROL_CODE_COUNT,
            self.renderer.state_channels,
            self.renderer.state_size,
            self.renderer.state_size,
        )
        if tuple(self.state_codes.shape) != expected_shape:
            raise RuntimeError(f"Expected state code shape {expected_shape}, got {tuple(self.state_codes.shape)}")

    def select_codes(self, target_indices: int | Tensor) -> Tensor:
        if isinstance(target_indices, int):
            indices = torch.tensor([target_indices], device=self.state_codes.device, dtype=torch.long)
        elif isinstance(target_indices, Tensor):
            indices = target_indices.to(device=self.state_codes.device)
            if indices.ndim == 0:
                indices = indices.unsqueeze(0)
            if indices.ndim != 1 or indices.dtype != torch.long:
                raise ValueError("target_indices must be a scalar or one-dimensional torch.long tensor")
        else:
            raise TypeError("target_indices must be an int or Tensor")
        if indices.numel() == 0 or (indices < 0).any() or (indices >= self.code_count).any():
            raise ValueError(f"target_indices must be non-empty and lie in [0, {self.code_count})")
        return self.state_codes.index_select(0, indices)

    def forward(self, target_indices: int | Tensor) -> Tensor:
        return self.renderer.render(self.select_codes(target_indices))

    @torch.no_grad()
    def clamp_codes_(self) -> None:
        self.state_codes.clamp_(-self.code_bound, self.code_bound)

    def assert_gradient_contract(self) -> tuple[float, float]:
        self.assert_parameter_contract()
        forbidden = [
            name
            for name, parameter in self.named_parameters()
            if name not in self.allowed_trainable_parameter_names() and parameter.grad is not None
        ]
        if forbidden:
            raise RuntimeError(f"Frozen non-renderer parameters accumulated gradients: {forbidden[:8]}")

        code_gradient = self.state_codes.grad
        if code_gradient is None or not torch.isfinite(code_gradient).all():
            raise RuntimeError("State-code gradient is missing or non-finite")
        renderer_gradients = [
            parameter.grad
            for parameter in self.renderer.rgb_head.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not renderer_gradients or any(not torch.isfinite(gradient).all() for gradient in renderer_gradients):
            raise RuntimeError("RGB-head gradients are missing or non-finite")
        code_norm = float(code_gradient.float().norm().item())
        renderer_norm = float(
            torch.sqrt(sum(gradient.float().square().sum() for gradient in renderer_gradients)).item()
        )
        if code_norm <= 0 or renderer_norm <= 0:
            raise RuntimeError(
                f"Renderer-control gradients must be positive; codes={code_norm}, rgb_head={renderer_norm}"
            )
        return code_norm, renderer_norm

    def forbidden_call_counts(self) -> dict[str, int]:
        return dict(self._forbidden_call_counts)


@dataclass(frozen=True)
class OptimizationStep:
    loss: float
    state_code_gradient_norm: float
    rgb_head_gradient_norm: float


def optimize_target_selected_image(
    diagnostic: TargetSelectedRendererControl,
    *,
    target_index: int,
    optimizer: torch.optim.Optimizer,
    loss_from_image: Callable[[Tensor], Tensor],
) -> OptimizationStep:
    """Run one target-leaking diagnostic step; loss_from_image receives a CHW image."""

    optimizer.zero_grad(set_to_none=True)
    image = diagnostic(target_index)[0]
    loss = loss_from_image(image)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("The diagnostic loss must be a finite scalar")
    loss.backward()
    code_norm, renderer_norm = diagnostic.assert_gradient_contract()
    optimizer.step()
    diagnostic.clamp_codes_()
    return OptimizationStep(
        loss=float(loss.item()),
        state_code_gradient_norm=code_norm,
        rgb_head_gradient_norm=renderer_norm,
    )


ChoiceScorer = Callable[[UniqueQuery, Tensor], ChoiceScoreOutput]


def evaluate_renderer_control(
    diagnostic: TargetSelectedRendererControl,
    queries: Sequence[UniqueQuery],
    *,
    choice_scorer: ChoiceScorer,
) -> tuple[float, list[dict[str, Any]]]:
    if not queries:
        raise ValueError("queries must be non-empty")
    records: list[dict[str, Any]] = []
    for item in queries:
        query = item.query
        image = diagnostic(query.target_index)[0].detach()
        score = choice_scorer(item, image)
        if len(score.mean_nll) != CONTROL_CODE_COUNT:
            raise RuntimeError("The renderer diagnostic requires four-choice mean-NLL scores")
        record = {
            "comparison_id": item.comparison_id,
            "member_episode_ids": sorted(member.episode_id for member in item.members),
            "topic": item.topic,
            "template_id": item.template_id,
            "pattern": query_pattern(item),
            "target_index": query.target_index,
            "target_text": query.target,
            "predicted_index": score.predicted_index,
            "choice_mean_nll": list(score.mean_nll),
            "target_mean_nll": score.mean_nll[query.target_index],
            "correct": score.predicted_index == query.target_index,
            "target_supervised_label_leak": TARGET_SUPERVISED_LABEL_LEAK,
            "diagnostic_scope": DIAGNOSTIC_SCOPE,
            "diagnostic_disclaimer": DISCLAIMER,
        }
        records.append(record)
    accuracy = sum(int(record["correct"]) for record in records) / len(records)
    return accuracy, records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.learning_rate <= 0:
        raise SystemExit("--steps, --eval-every, and --lr must be positive")
    if args.limit < 1 or args.state_channels < 1 or args.state_size < 1 or args.output_size < 1:
        raise SystemExit("--limit and all state/output dimensions must be positive")
    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("--threshold must be in (0, 1]")
    if not torch.cuda.is_available():
        raise SystemExit("The renderer-control diagnostic requires CUDA and the real frozen Qwen Reader")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("The renderer-control diagnostic refuses a non-empty --output-dir")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("The renderer-control diagnostic refuses a dirty worktree")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA")
    set_seeds(args.seed)
    episodes, subset_audit = select_episode_prefix(args.dataset, limit=args.limit)
    raw_query_count, queries = collect_unique_queries(episodes)
    if any(len(item.query.choices) != CONTROL_CODE_COUNT for item in queries):
        raise RuntimeError("Every selected query must have exactly four choices")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    renderer = LightweightVisualUpdater(
        state_channels=args.state_channels,
        state_size=args.state_size,
        output_size=args.output_size,
    ).to(device=device, dtype=torch.float32)
    diagnostic = TargetSelectedRendererControl(renderer).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in diagnostic.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    manifest_path = args.output_dir / "manifest.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    evaluations_path = args.output_dir / "evaluations.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    checkpoint_path = args.output_dir / "renderer_control_checkpoint.pt"
    manifest = {
        "schema_version": "vision_memory.target_selected_renderer_manifest.v1",
        "target_supervised_label_leak": TARGET_SUPERVISED_LABEL_LEAK,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_disclaimer": DISCLAIMER,
        "selector": "target_index -> one of four trainable hidden-state codes",
        "renderer_path": "LightweightVisualUpdater.rgb_head followed by render interpolation",
        "event_encoder_called": False,
        "convgru_called": False,
        "trainable_parameter_names": sorted(diagnostic.trainable_parameter_names()),
        "state_code_shape": list(diagnostic.state_codes.shape),
        "state_code_bound": diagnostic.code_bound,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "limit_unit": "ordered_episode_prefix",
        "train_subset": subset_audit,
        "raw_query_count": raw_query_count,
        "comparison_query_count": len(queries),
        "reader": str(args.reader.resolve()),
        "reader_revision": locked_revision(args.reader),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    _write_json(manifest_path, manifest)

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
        raise RuntimeError("A tensor-native fast Qwen image processor is required")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False

    def choice_scorer(item: UniqueQuery, image: Tensor) -> ChoiceScoreOutput:
        query = item.query
        return qwen3vl_choice_nll(
            model=reader,
            processor=processor,
            image=image,
            query=format_mcq_query(query.text, query.choices),
            choices=query.choices,
            device=device,
        )

    def evaluate_at_step(step: int, *, final: bool) -> tuple[float, list[dict[str, Any]]]:
        accuracy, records = evaluate_renderer_control(diagnostic, queries, choice_scorer=choice_scorer)
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        evaluation = {
            "optimizer_step": step,
            "final": final,
            "accuracy": accuracy,
            "correct": sum(int(record["correct"]) for record in records),
            "count": len(records),
            "mean_target_nll": sum(float(record["target_mean_nll"]) for record in records) / len(records),
            "passed_threshold": accuracy >= args.threshold,
        }
        _append_jsonl(evaluations_path, evaluation)
        print(json.dumps({"renderer_control_evaluation": evaluation}, sort_keys=True), flush=True)
        return accuracy, records

    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    evaluate_at_step(0, final=False)
    query_order = deterministic_query_order(len(queries), steps=args.steps, seed=args.seed)
    final_accuracy = 0.0
    final_records: list[dict[str, Any]] = []
    for optimizer_step, query_index in enumerate(query_order, start=1):
        item = queries[query_index]
        query = item.query

        def loss_from_image(image: Tensor) -> Tensor:
            return qwen3vl_target_only_ce(
                model=reader,
                processor=processor,
                image=image,
                query=format_mcq_query(query.text, query.choices),
                target=query.target,
                device=device,
                require_image_grad=True,
            ).loss

        result = optimize_target_selected_image(
            diagnostic,
            target_index=query.target_index,
            optimizer=optimizer,
            loss_from_image=loss_from_image,
        )
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        _append_jsonl(
            metrics_path,
            {
                "optimizer_step": optimizer_step,
                "comparison_id": item.comparison_id,
                "target_index": query.target_index,
                "loss": result.loss,
                "state_code_gradient_norm": result.state_code_gradient_norm,
                "rgb_head_gradient_norm": result.rgb_head_gradient_norm,
            },
        )
        if optimizer_step % args.eval_every == 0 or optimizer_step == args.steps:
            final_accuracy, final_records = evaluate_at_step(
                optimizer_step,
                final=optimizer_step == args.steps,
            )

    if not final_records:
        final_accuracy, final_records = evaluate_at_step(args.steps, final=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    forbidden_calls = diagnostic.forbidden_call_counts()
    if any(forbidden_calls.values()):
        raise RuntimeError(f"Forbidden updater modules were called: {forbidden_calls}")
    assert_no_frozen_parameter_grads(reader, "Qwen Reader")
    diagnostic.assert_parameter_contract()
    checkpoint = {
        "schema_version": "vision_memory.target_selected_renderer_checkpoint.v1",
        "target_supervised_label_leak": TARGET_SUPERVISED_LABEL_LEAK,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_disclaimer": DISCLAIMER,
        "optimizer_step": args.steps,
        "state_codes": diagnostic.state_codes.detach().cpu(),
        "rgb_head_state_dict": {
            name: tensor.detach().cpu() for name, tensor in diagnostic.renderer.rgb_head.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {
            "state_channels": args.state_channels,
            "state_size": args.state_size,
            "output_size": args.output_size,
            "state_code_bound": diagnostic.code_bound,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    target_positions = Counter(record["target_index"] for record in final_records)
    state_codes_cpu = diagnostic.state_codes.detach().cpu().contiguous()
    summary = {
        "schema_version": "vision_memory.target_selected_renderer_summary.v1",
        "target_supervised_label_leak": TARGET_SUPERVISED_LABEL_LEAK,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_disclaimer": DISCLAIMER,
        "passed_threshold": final_accuracy >= args.threshold,
        "threshold": args.threshold,
        "accuracy": final_accuracy,
        "accuracy_by_pattern": grouped_accuracy(final_records, "pattern"),
        "accuracy_by_target_position": grouped_accuracy(final_records, "target_index"),
        "target_position_counts": dict(sorted(target_positions.items())),
        "raw_query_count": raw_query_count,
        "comparison_query_count": len(queries),
        "dataset_sha256": sha256_file(args.dataset),
        "reader_revision": locked_revision(args.reader),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "trainable_parameter_names": sorted(diagnostic.trainable_parameter_names()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in diagnostic.parameters() if parameter.requires_grad
        ),
        "state_code_tensor_sha256": sha256_bytes(state_codes_cpu.numpy().tobytes()),
        "forbidden_forward_call_counts": forbidden_calls,
        "frozen_reader_parameter_grad_count": sum(parameter.grad is not None for parameter in reader.parameters()),
        "frozen_non_renderer_parameter_grad_count": sum(
            parameter.grad is not None
            for name, parameter in diagnostic.named_parameters()
            if name not in diagnostic.allowed_trainable_parameter_names()
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "evaluations": str(evaluations_path.resolve()),
            "evaluations_sha256": sha256_file(evaluations_path),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return renderer_control_exit_code(summary["passed_threshold"])


if __name__ == "__main__":
    raise SystemExit(main())
