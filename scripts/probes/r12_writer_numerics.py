"""R12 two-GPU numerical preflight for the exact fixed scalar objective."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from scripts.train import r12_shared_event_latent_writer as trainer  # noqa: E402
from vision_memory.data import CYCLIC4  # noqa: E402


SCHEMA = "vision_memory.r12-objective-numerics-preflight.v2"
BACKWARD_LOSS_DIVISOR = 1.0
ALLOWED_BACKWARD_LOSS_DIVISORS = (1.0,) + tuple(
    float(2**exponent) for exponent in range(10, 23, 2)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--r11-root", type=Path, required=True)
    parser.add_argument("--r11-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument(
        "--backward-loss-divisor",
        type=float,
        choices=ALLOWED_BACKWARD_LOSS_DIVISORS,
        default=BACKWARD_LOSS_DIVISOR,
        help="Diagnostic power-of-two divisor; the formal trainer remains separately locked.",
    )
    return parser


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _nonfinite_parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        name: int((~torch.isfinite(parameter.detach())).sum())
        for name, parameter in module.named_parameters()
        if not torch.isfinite(parameter.detach()).all()
    }


def _nonfinite_optimizer_state_counts(
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                counts[f"parameter_{parameter_index}.{name}"] = int(
                    (~torch.isfinite(value)).sum()
                )
    return counts


def _tensor_gradient_statistics(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"present": False, "nonfinite": None, "max_abs_finite": None}
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "present": True,
        "nonfinite": int((~finite).sum()),
        "max_abs_finite": (
            float(finite_values.abs().max()) if finite_values.numel() else None
        ),
    }


def _run(cli: argparse.Namespace) -> dict[str, Any]:
    if cli.output.exists():
        raise ValueError("R12 numerical preflight refuses to overwrite its output.")
    with tempfile.TemporaryDirectory(prefix="r12-numerics-") as directory:
        args = trainer.parse_args(
            [
                "--arm",
                "conditioned",
                "--train",
                str(cli.train),
                "--dev",
                str(cli.dev),
                "--dreamlite",
                str(cli.dreamlite),
                "--reader",
                str(cli.reader),
                "--r11-root",
                str(cli.r11_root),
                "--r11-comparison",
                str(cli.r11_comparison),
                "--output-dir",
                directory,
            ]
        )
        validation = trainer._validate_args(args)
        determinism = r8.configure_strict_cuda_determinism(0)
        r8.set_all_seeds(0)
        initial, basis, basis_audit = trainer._load_r11_basis(
            r11_root=args.r11_root,
            config=validation["config"],
        )
        data = r5._load_data(args, optimizer_steps=0)
        selected, _audit = trainer.select_balanced_train_f1(data.train_pools["F1"])
        schedule = trainer.build_training_schedule(selected)
        micros = schedule[: trainer.R12_GRADIENT_ACCUMULATION]
        next_unit = schedule[trainer.R12_GRADIENT_ACCUMULATION]
        writer_device = torch.device(cli.dreamlite_device)
        reader_device = torch.device(cli.reader_device)
        writer_dtype = r5.compute_dtype(writer_device)
        reader_dtype = r5.compute_dtype(reader_device)
        pipe = r5._load_pipeline(args, writer_device, writer_dtype)
        pipe.unet.requires_grad_(False).eval()
        pipe.text_encoder.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()
        processor, reader = r5._load_reader(args, reader_device, reader_dtype)
        cache, embedding_audit = trainer._embedding_cache(
            pipe=pipe,
            segments=[unit.segment for unit in (*micros, next_unit)],
            device=writer_device,
            dtype=writer_dtype,
            output_dir=Path(directory),
        )
        writer = trainer.SharedEventLatentWriter(
            initial_latent=initial.to(writer_device),
            initial_basis=basis.to(writer_device),
        ).to(writer_device)
        network = [
            parameter
            for name, parameter in writer.named_parameters()
            if name != "basis_raw"
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": network,
                    "lr": trainer.R12_WRITER_LEARNING_RATE,
                    "weight_decay": trainer.R12_WEIGHT_DECAY,
                },
                {
                    "params": [writer.basis_raw],
                    "lr": trainer.R12_WRITER_LEARNING_RATE,
                    "weight_decay": 0.0,
                },
            ],
            lr=trainer.R12_WRITER_LEARNING_RATE,
        )
        reader_fn = r8.choice_reader_callable(
            reader=reader,
            processor=processor,
            reader_device=reader_device,
            require_grad=True,
            deterministic_ce=True,
        )
        divisor = float(cli.backward_loss_divisor)
        micro_records = []
        for unit in micros:
            token_states, mask = trainer._tokens(
                cache, unit.segment.segment_id, writer_device
            )
            latent, diagnostics = writer(token_states, mask, conditioned=True)
            latent.retain_grad()
            image = trainer._decode(pipe.vae, latent, writer_dtype)
            image.retain_grad()
            output = trainer._reader_output(
                reader_fn,
                image=image,
                segment=unit.segment,
                permutation=CYCLIC4[unit.forward_cyclic_training_view],
            )
            latent_penalty = trainer._latent_rms_penalty(diagnostics["delta"])
            coefficient_penalty = (
                trainer.R12_COEFFICIENT_L2_PENALTY
                * diagnostics["coefficients"].square().mean()
            )
            objective = (
                output.loss
                + latent_penalty.to(output.loss.device)
                + coefficient_penalty.to(output.loss.device)
            )
            (
                objective
                / trainer.R12_GRADIENT_ACCUMULATION
                / divisor
            ).backward()
            bad_parameters = {
                name: int((~torch.isfinite(parameter.grad)).sum())
                for name, parameter in writer.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            }
            micro_records.append(
                {
                    "global_micro_index": unit.global_micro_index,
                    "segment_id": unit.segment.segment_id,
                    "ce": float(output.loss.detach()),
                    "latent_gradient": _tensor_gradient_statistics(latent.grad),
                    "image_gradient": _tensor_gradient_statistics(image.grad),
                    "writer_nonfinite_parameters": bad_parameters,
                }
            )
            if (
                bad_parameters
                or micro_records[-1]["latent_gradient"]["nonfinite"]
                or micro_records[-1]["image_gradient"]["nonfinite"]
            ):
                raise RuntimeError(
                    "R12 scaled preflight produced non-finite gradients at "
                    f"micro {unit.global_micro_index}: "
                    f"{json.dumps(micro_records[-1], sort_keys=True)}"
                )
        scaled = trainer._gradient_statistics(writer)
        for parameter in writer.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(divisor)
        unscaled = trainer._gradient_statistics(writer)
        ratio = unscaled["gradient_norm"] / scaled["gradient_norm"]
        if not math.isclose(ratio, divisor, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"R12 exact gradient unscale ratio drifted: {ratio}")
        pre_step_parameters = {
            name: parameter.detach().clone()
            for name, parameter in writer.named_parameters()
        }
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        nonfinite_parameters = _nonfinite_parameter_counts(writer)
        nonfinite_optimizer_state = _nonfinite_optimizer_state_counts(optimizer)
        if nonfinite_parameters or nonfinite_optimizer_state:
            raise RuntimeError(
                "R12 first scaled optimizer step produced non-finite optimizer state: "
                f"parameters={nonfinite_parameters}, optimizer={nonfinite_optimizer_state}"
            )
        parameter_change_l2 = math.sqrt(
            sum(
                float(
                    (parameter.detach() - pre_step_parameters[name])
                    .double()
                    .square()
                    .sum()
                )
                for name, parameter in writer.named_parameters()
            )
        )
        if not math.isfinite(parameter_change_l2) or parameter_change_l2 <= 0.0:
            raise RuntimeError("R12 first scaled optimizer step did not update the writer.")
        token_states, mask = trainer._tokens(
            cache, next_unit.segment.segment_id, writer_device
        )
        latent, diagnostics = writer(token_states, mask, conditioned=True)
        image = trainer._decode(pipe.vae, latent, writer_dtype)
        finite_endpoint = bool(torch.isfinite(latent).all() and torch.isfinite(image).all())
        if not finite_endpoint:
            raise RuntimeError("R12 first scaled optimizer step produced a non-finite state.")
        reader_frozen_clean = all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in reader.parameters()
        )
        vae_frozen_clean = all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in pipe.vae.parameters()
        )
        if not reader_frozen_clean or not vae_frozen_clean:
            raise RuntimeError("R12 preflight contaminated a frozen Reader or VAE parameter.")
        return {
            "schema": SCHEMA,
            "status": "passed",
            "source_commit": validation["git_commit"],
            "host": platform.node(),
            "strict_determinism": determinism,
            "backward_loss_divisor": divisor,
            "stable_rms_penalty_value_equivalent": True,
            "gradient_accumulation": trainer.R12_GRADIENT_ACCUMULATION,
            "micro_records": micro_records,
            "scaled_gradient_norm": scaled["gradient_norm"],
            "unscaled_gradient_norm": unscaled["gradient_norm"],
            "exact_unscale_ratio": ratio,
            "post_step_writer_parameter_change_l2": parameter_change_l2,
            "post_step_writer_nonfinite_parameters": nonfinite_parameters,
            "post_step_optimizer_nonfinite_state": nonfinite_optimizer_state,
            "post_step_coefficient_norm": float(diagnostics["coefficients"].norm()),
            "post_step_latent_delta_rms": float(diagnostics["delta_rms"]),
            "post_step_image_saturation_fraction": float(
                ((image <= 0.0) | (image >= 1.0)).double().mean()
            ),
            "post_step_finite": finite_endpoint,
            "basis_audit_passed": basis_audit["passed"],
            "embedding_audit_passed": embedding_audit["passed"],
            "reader_frozen_clean": reader_frozen_clean,
            "vae_frozen_clean": vae_frozen_clean,
        }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "host": platform.node(),
            "backward_loss_divisor": float(args.backward_loss_divisor),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write(args.output, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 1
    _write(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
