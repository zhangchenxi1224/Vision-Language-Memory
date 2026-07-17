from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_choice_nll,
    qwen3vl_listwise_choice_ce,
)
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    cuda_peak_memory_report,
    emit_json_report,
    probe_provenance,
    reset_cuda_peak_memory,
)
from vision_memory.training import format_mcq_query  # noqa: E402


NLL_RTOL = 1e-6
NLL_ATOL = 1e-6
REPEAT_RTOL = 0.0
REPEAT_ATOL = 0.0


@dataclass(frozen=True)
class ChoiceView:
    name: str
    family: str
    step: int
    source_indices: tuple[int, ...]
    choices: tuple[str, ...]
    target_index: int


def dihedral_choice_views(choices: Sequence[str], target_index: int) -> tuple[ChoiceView, ...]:
    """Return the four rotations and four reflected rotations of a four-choice query."""

    canonical = tuple(choices)
    if len(canonical) != 4 or len(set(canonical)) != 4:
        raise ValueError("R3-S0 requires exactly four distinct choices.")
    if any(not isinstance(choice, str) or not choice.strip() for choice in canonical):
        raise ValueError("R3-S0 choices must be non-empty strings.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("R3-S0 target_index must be an integer in [0, 3].")

    views: list[ChoiceView] = []
    for family, direction in (("cyclic", 1), ("reverse-cyclic", -1)):
        for step in range(4):
            source_indices = tuple((step + direction * offset) % 4 for offset in range(4))
            mapped_target_index = source_indices.index(target_index)
            views.append(
                ChoiceView(
                    name=f"{family}-{step}",
                    family=family,
                    step=step,
                    source_indices=source_indices,
                    choices=tuple(canonical[index] for index in source_indices),
                    target_index=mapped_target_index,
                )
            )
    return tuple(views)


def validate_choice_view_mappings(
    views: Sequence[ChoiceView],
    *,
    canonical_target: str,
) -> dict[str, Any]:
    expected_names = {
        *(f"cyclic-{step}" for step in range(4)),
        *(f"reverse-cyclic-{step}" for step in range(4)),
    }
    observed_names = {view.name for view in views}
    semantic_targets_preserved = all(view.choices[view.target_index] == canonical_target for view in views)
    per_family_positions = {
        family: sorted(view.target_index for view in views if view.family == family)
        for family in ("cyclic", "reverse-cyclic")
    }
    passed = (
        len(views) == 8
        and observed_names == expected_names
        and semantic_targets_preserved
        and all(positions == [0, 1, 2, 3] for positions in per_family_positions.values())
    )
    return {
        "passed": passed,
        "view_count": len(views),
        "expected_names_present": observed_names == expected_names,
        "semantic_targets_preserved": semantic_targets_preserved,
        "target_positions_by_family": per_family_positions,
    }


def _token_ids(tokenizer: Any, text: str) -> Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("Tokenizer must return input_ids with shape [1, sequence].")
    return input_ids.detach().cpu()


def audit_joint_chat_tokenization(
    *,
    processor: Any,
    query: str,
    choices: Sequence[str],
) -> dict[str, Any]:
    """Independently audit option tokenization in the actual assistant left context."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("Qwen chat template returned an empty or non-string generation prompt.")
    prompt_ids = _token_ids(processor.tokenizer, prompt)
    choice_records: list[dict[str, Any]] = []
    for choice in choices:
        joint_ids = _token_ids(processor.tokenizer, prompt + choice)
        prefix_stable = joint_ids.shape[1] > prompt_ids.shape[1] and torch.equal(
            joint_ids[:, : prompt_ids.shape[1]], prompt_ids
        )
        suffix = joint_ids[:, prompt_ids.shape[1] :] if prefix_stable else torch.empty(1, 0, dtype=torch.long)
        choice_records.append(
            {
                "choice": choice,
                "prefix_stable": prefix_stable,
                "joint_suffix_token_ids": [int(value) for value in suffix.flatten().tolist()],
                "joint_suffix_token_count": int(suffix.numel()),
            }
        )
    passed = bool(choice_records) and all(
        record["prefix_stable"] and record["joint_suffix_token_count"] > 0 for record in choice_records
    )
    return {
        "passed": passed,
        "prompt_token_count": int(prompt_ids.numel()),
        "choices": choice_records,
    }


def compare_nll_vectors(
    left: Sequence[float],
    right: Sequence[float],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    same_length = len(left_values) == len(right_values) == 4
    finite = same_length and all(math.isfinite(value) for value in (*left_values, *right_values))
    raw_absolute_differences = (
        tuple(abs(first - second) for first, second in zip(left_values, right_values, strict=True))
        if same_length
        else ()
    )
    absolute_differences = raw_absolute_differences if finite else ()
    elementwise_close = (
        tuple(
            difference <= atol + rtol * abs(reference)
            for difference, reference in zip(absolute_differences, right_values, strict=True)
        )
        if finite
        else ()
    )
    passed = finite and all(elementwise_close)
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "same_length_four": same_length,
        "all_values_finite": finite,
        "maximum_absolute_difference": max(absolute_differences) if absolute_differences else None,
        "absolute_differences": absolute_differences,
        "elementwise_close": elementwise_close,
    }


def run_scorer_contract(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    choices: Sequence[str],
    target_index: int,
    device: torch.device,
    listwise_scorer: Callable[..., Any] = qwen3vl_listwise_choice_ce,
    eval_scorer: Callable[..., Any] = qwen3vl_choice_nll,
) -> dict[str, Any]:
    """Exercise the real train/eval scorer boundary over all eight D4 choice views."""

    views = dihedral_choice_views(choices, target_index)
    canonical_target = tuple(choices)[target_index]
    mapping_validation = validate_choice_view_mappings(views, canonical_target=canonical_target)
    view_reports: list[dict[str, Any]] = []

    for view in views:
        formatted_query = format_mcq_query(query, view.choices)
        tokenization = audit_joint_chat_tokenization(
            processor=processor,
            query=formatted_query,
            choices=view.choices,
        )
        train_result = listwise_scorer(
            model=model,
            processor=processor,
            image=image,
            query=formatted_query,
            choices=view.choices,
            target_index=view.target_index,
            device=device,
            require_image_grad=True,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=True,
        )
        train_nll = tuple(float(value) for value in train_result.choice_mean_nll.detach().cpu().tolist())
        eval_result = eval_scorer(
            model=model,
            processor=processor,
            image=image,
            query=formatted_query,
            choices=view.choices,
            device=device,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=True,
        )
        repeat_result = eval_scorer(
            model=model,
            processor=processor,
            image=image,
            query=formatted_query,
            choices=view.choices,
            device=device,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=True,
        )
        eval_nll = tuple(float(value) for value in eval_result.mean_nll)
        repeat_nll = tuple(float(value) for value in repeat_result.mean_nll)
        if not all(math.isfinite(value) for value in (*train_nll, *eval_nll, *repeat_nll)):
            raise RuntimeError(f"View {view.name} produced a non-finite train, eval, or repeated-eval NLL.")
        train_eval_comparison = compare_nll_vectors(
            train_nll,
            eval_nll,
            rtol=NLL_RTOL,
            atol=NLL_ATOL,
        )
        repeat_comparison = compare_nll_vectors(
            eval_nll,
            repeat_nll,
            rtol=REPEAT_RTOL,
            atol=REPEAT_ATOL,
        )
        expected_target_ids = tokenization["choices"][view.target_index]["joint_suffix_token_ids"]
        observed_target_ids = [int(value) for value in train_result.target_ids.detach().cpu().flatten().tolist()]
        target_ids_match_joint = observed_target_ids == expected_target_ids
        expected_choice_token_counts = tuple(
            int(record["joint_suffix_token_count"]) for record in tokenization["choices"]
        )
        observed_choice_token_counts = tuple(int(value) for value in train_result.choice_token_counts)
        choice_token_counts_match_joint = observed_choice_token_counts == expected_choice_token_counts
        train_loss_requires_grad = bool(train_result.loss.requires_grad)
        view_passed = bool(
            tokenization["passed"]
            and target_ids_match_joint
            and choice_token_counts_match_joint
            and train_loss_requires_grad
            and train_eval_comparison["passed"]
            and repeat_comparison["passed"]
        )
        view_reports.append(
            {
                "name": view.name,
                "family": view.family,
                "step": view.step,
                "source_indices": view.source_indices,
                "choices": view.choices,
                "target_index": view.target_index,
                "target_text": view.choices[view.target_index],
                "joint_chat_tokenization": tokenization,
                "observed_train_target_token_ids": observed_target_ids,
                "target_ids_match_joint": target_ids_match_joint,
                "expected_joint_choice_token_counts": expected_choice_token_counts,
                "observed_train_choice_token_counts": observed_choice_token_counts,
                "choice_token_counts_match_joint": choice_token_counts_match_joint,
                "train_loss_requires_grad": train_loss_requires_grad,
                "train_choice_mean_nll": train_nll,
                "eval_choice_mean_nll": eval_nll,
                "repeat_eval_choice_mean_nll": repeat_nll,
                "train_eval_comparison": train_eval_comparison,
                "repeat_eval_comparison": repeat_comparison,
                "passed": view_passed,
            }
        )
        # Do not retain four full differentiable Reader graphs while constructing the
        # next view. Detached scalars and token IDs above contain all audit evidence.
        del train_result, eval_result, repeat_result

    passed = bool(mapping_validation["passed"] and all(report["passed"] for report in view_reports))
    return {
        "schema_version": 1,
        "probe": "r3_s0_qwen_scorer_contract",
        "passed": passed,
        "contract": {
            "reader_loss_mode": "listwise-choice",
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "nll_implementation": "fp32-logsumexp-minus-target-score",
            "tokenization": "joint-chat-template-continuation",
            "train_eval_nll_rtol": NLL_RTOL,
            "train_eval_nll_atol": NLL_ATOL,
            "repeat_eval_rtol": REPEAT_RTOL,
            "repeat_eval_atol": REPEAT_ATOL,
            "repeat_scope": "every-cyclic-and-reverse-cyclic-view",
        },
        "canonical_choices": tuple(choices),
        "canonical_target_index": target_index,
        "canonical_target_text": canonical_target,
        "mapping_validation": mapping_validation,
        "views": view_reports,
        "summary": {
            "views_passed": sum(int(report["passed"]) for report in view_reports),
            "views_required": 8,
            "joint_tokenization_views_passed": sum(
                int(report["joint_chat_tokenization"]["passed"]) for report in view_reports
            ),
            "train_eval_views_passed": sum(int(report["train_eval_comparison"]["passed"]) for report in view_reports),
            "repeat_eval_views_passed": sum(int(report["repeat_eval_comparison"]["passed"]) for report in view_reports),
        },
    }


def contract_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("passed") is True else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3-S0 real Qwen listwise train/eval scorer contract probe")
    parser.add_argument("--reader", type=Path, default=ROOT / "models" / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--query", default="Which stored preference should be selected?")
    parser.add_argument("--choice", action="append", help="Repeat exactly four times; defaults to four color options.")
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-small-gpu", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.choice is None:
        args.choice = ["red", "blue", "green", "yellow"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "r3_s0_qwen_scorer_contract",
        "passed": False,
    }
    try:
        choices = tuple(args.choice)
        # Validate before model loading so malformed scientific inputs fail cheaply.
        dihedral_choice_views(choices, args.target_index)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real Qwen scorer contract probe.")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("The real Qwen scorer contract probe requires a CUDA device.")
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if memory_gib < 16 and not args.allow_small_gpu:
            raise RuntimeError(f"Only {memory_gib:.1f} GiB VRAM detected; run this probe on the cluster.")

        strict_determinism = configure_strict_cuda_determinism(seed=args.seed)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        reset_cuda_peak_memory([device])
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
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.reader,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        freeze_module(model)
        model.eval()
        model.config.use_cache = False

        generator = torch.Generator(device=device).manual_seed(args.seed)
        image = torch.rand(
            3,
            1024,
            1024,
            generator=generator,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        report = run_scorer_contract(
            model=model,
            processor=processor,
            image=image,
            query=args.query,
            choices=choices,
            target_index=args.target_index,
            device=device,
        )
        report.update(
            {
                "reader_processor": processor_name,
                "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                "reader_dtype": str(dtype),
                "device": str(device),
                "strict_determinism": strict_determinism,
                "image_sha256": canonical_tensor_sha256(image),
                "frozen_gradients": assert_no_frozen_parameter_grads(
                    {"reader": model},
                    fully_frozen={"reader"},
                ),
                "cuda_peak_memory": cuda_peak_memory_report([device]),
                "provenance": probe_provenance(
                    root=ROOT,
                    arguments=args,
                    models={"reader": args.reader},
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 - the probe must emit JSON for every runtime contract failure
        report = {
            **base_report,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "provenance": probe_provenance(
                root=ROOT,
                arguments=args,
                models={"reader": args.reader},
            ),
        }
    emit_json_report(report, args.output_json)
    return contract_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
