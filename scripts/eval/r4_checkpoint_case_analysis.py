"""Evaluate R4 checkpoints on fixed cases and save intermediate RGB/latent outputs.

Run this on the two-GPU training notebook after a checkpoint run exists.  It
never chooses a best checkpoint; it evaluates the requested checkpoints using
the same fixed dev cases and emits images plus machine-readable state statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import dreamlite_r4_free_pixel as R4  # noqa: E402
from vision_memory.data import REVERSE_CYCLIC4, read_jsonl  # noqa: E402
from vision_memory.training import load_trainable_weights  # noqa: E402


def _turn_field(turn: Any, name: str, default: Any = None) -> Any:
    return turn.get(name, default) if isinstance(turn, dict) else getattr(turn, name, default)


def _turn_type(turn: Any) -> str:
    value = _turn_field(turn, "type", _turn_field(turn, "kind"))
    return str(getattr(value, "value", value)).lower()


def _event_text(turn: Any) -> str:
    return str(_turn_field(turn, "event_text", ""))


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    tensor = value.detach().float().clamp(0, 1).cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float()
    return {
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "rms": float(tensor.square().mean().sqrt()),
        "finite_fraction": float(torch.isfinite(tensor).float().mean()),
    }


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    load_trainable_weights(path, trainable_module=model)


def _runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dreamlite=args.dreamlite,
        reader=args.reader,
        checkpoint_unet=args.checkpoint_unet,
        adapter_seed=args.adapter_seed,
        strict_determinism=False,
    )


def analyze_checkpoint(
    *,
    checkpoint: Path,
    model: Any,
    reader_fn: Any,
    episodes: list[Any],
    output_dir: Path,
    max_cases: int,
) -> dict[str, Any]:
    _load_checkpoint(model, checkpoint)
    checkpoint_dir = output_dir / checkpoint.stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    aggregate = R4.evaluate_r4(model=model, records=episodes, reader_fn=reader_fn)
    case_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_index, episode in enumerate(episodes[:max_cases]):
            episode_id = str(_turn_field(episode, "episode_id"))
            state = model.reset_state()
            case_dir = checkpoint_dir / "cases" / f"{case_index:03d}-{episode_id}"
            turn_rows: list[dict[str, Any]] = []
            for turn_index, turn in enumerate(_turn_field(episode, "turns", ())):
                kind = _turn_type(turn)
                if kind in {"event", "mixed"}:
                    trace = model.updater.forward_with_trace(
                        state,
                        _event_text(turn),
                        episode_id,
                        turn_index,
                        gradient_mode="full",
                        persistent_state="float_rgb",
                        presentation_index=0,
                    )
                    event_dir = case_dir / f"turn-{turn_index:03d}-event"
                    _save_rgb(event_dir / "state.png", trace.output_state)
                    trajectory_rows: list[dict[str, Any]] = []
                    for diffusion_index, latent in enumerate(trace.latent_trajectory):
                        decoded = model.updater.decode_for_reader(latent, persistent_state="latent")
                        _save_rgb(event_dir / f"diffusion-{diffusion_index:02d}.png", decoded)
                        trajectory_rows.append({
                            "diffusion_index": diffusion_index,
                            "latent": _tensor_stats(latent),
                            "decoded_rgb": _tensor_stats(decoded),
                        })
                    turn_rows.append({
                        "turn_index": turn_index,
                        "kind": kind,
                        "event_text": _event_text(turn),
                        "output_state": _tensor_stats(trace.output_state),
                        "source_latents": _tensor_stats(trace.source_latents),
                        "output_latents": _tensor_stats(trace.output_latents),
                        "trajectory": trajectory_rows,
                    })
                    state = trace.output_state
                if kind in {"query", "mixed"}:
                    text, choices, target = R4._query_payload(turn)
                    permutation = REVERSE_CYCLIC4[0]
                    ordered = tuple(choices[index] for index in permutation)
                    ordered_target = permutation.index(target)
                    query_loss = R4._loss_tensor(reader_fn(
                        state,
                        R4.format_mcq_query(text, ordered),
                        ordered,
                        ordered_target,
                    ))
                    turn_rows.append({
                        "turn_index": turn_index,
                        "kind": kind,
                        "query": text,
                        "target_index": target,
                        "choice_ce": float(query_loss),
                        "state": _tensor_stats(state),
                    })
            case_rows.append({"episode_id": episode_id, "case_index": case_index, "turns": turn_rows})
    report = {
        "schema": "vision_memory.r4-checkpoint-case-analysis.v1",
        "checkpoint": checkpoint.name,
        "aggregate_fixed_dev": aggregate,
        "cases": case_rows,
        "notes": [
            "Trajectory index 0 is the initial noisy latent; indices 1-4 are after each mobile denoising step.",
            "Case analysis is diagnostic only and never changes endpoint or checkpoint selection.",
        ],
    }
    (checkpoint_dir / "cases.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("checkpoint case analysis requires two visible CUDA devices")
    episodes = read_jsonl(args.dev)[: args.eval_limit]
    runtime_args = _runtime_args(args)
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    pipe = R4._load_pipeline(runtime_args, updater_device, R4.compute_dtype(updater_device))
    processor, reader = R4._load_reader(runtime_args, reader_device, R4.compute_dtype(reader_device))
    model = R4.R4FreePixelModel(
        pipeline=pipe,
        initial_rgb=R4._initial_rgb_tensor(
            resolution=1024,
            device=updater_device,
            dtype=R4.compute_dtype(updater_device),
        ),
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
    )
    named, _ = R4._force_trainable_fp32(model)
    runtime = SimpleNamespace(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
    )
    reader_fn = R4._reader_fn(runtime, args, require_grad=False)
    checkpoints = args.checkpoints
    if checkpoints is None:
        roots = (args.run_dir / "checkpoints", args.run_dir / "output" / "checkpoints")
        checkpoints = sorted(path for root in roots for path in root.glob("step-*.pt"))
    if not checkpoints:
        raise SystemExit("no checkpoints supplied or found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        analyze_checkpoint(
            checkpoint=path,
            model=model,
            reader_fn=reader_fn,
            episodes=episodes,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
        )
        for path in checkpoints
    ]
    (args.output_dir / "checkpoint_case_index.json").write_text(
        json.dumps(
            [{"checkpoint": report["checkpoint"], "aggregate_fixed_dev": report["aggregate_fixed_dev"]}
             for report in reports],
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint_count": len(reports), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

