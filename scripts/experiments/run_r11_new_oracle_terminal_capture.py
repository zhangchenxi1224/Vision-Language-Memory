"""Run or independently audit the preregistered oracle terminal-capture scan."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_direction_fidelity as direction_runner  # noqa: E402
from scripts.experiments import run_r11_new_reachable_lr_control as low_lr_runner  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import r11_new_direction_fidelity as direction_core  # noqa: E402
from vision_memory.training import r11_new_oracle_terminal_capture as core  # noqa: E402
from vision_memory.training import r11_new_reachable_lr_control as low_lr_core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("technical-preflight", "formal", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-root", type=Path)
    return parser.parse_args(argv)


def validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    direction_config = direction_core.load_config()
    contract = config["base_runtime_contract"]
    core.require(
        direction_core.sha256_file(direction_core.CONFIG_PATH) == contract["config_byte_sha256"]
        and direction_core.canonical_sha(direction_config) == contract["config_canonical_sha256"],
        "Base direction runtime contract drift.",
    )
    low_lr_config = low_lr_core.load_config()
    for key in contract["reuse_exact_sections"]:
        core.require(
            direction_config[key] == low_lr_config[key], f"Reused runtime section drift: {key}."
        )
    base_args = argparse.Namespace(**vars(args))
    base_args.mode = "technical-preflight"
    base_args.preflight_root = None
    base_validation = low_lr_runner.validate_environment(base_args, low_lr_config)
    parent = config["parent_direction_failure"]
    parent_root = Path(parent["source_root"])
    bindings = {
        "result": (parent_root / "result.json", parent["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", parent["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", parent["source_terminal_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", parent["source_inventory_sha256"]),
        "archive": (Path(parent["source_archive_path"]), parent["source_archive_sha256"]),
    }
    fixed = config["fixed_parent_artifacts"]
    bindings.update(
        {
            "target": (
                Path(fixed["source_target_path"]),
                fixed["source_target_artifact_sha256"],
            ),
            "plateau": (
                Path(fixed["source_plateau_checkpoint_path"]),
                fixed["source_plateau_checkpoint_sha256"],
            ),
        }
    )
    for name, (path, expected_hash) in bindings.items():
        core.require(
            path.is_file() and core.sha256_file(path) == expected_hash,
            f"Terminal parent {name} binding drift.",
        )
    parent_audit = direction_core.audit_delivery(parent_root, direction_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == parent["classification"]
        and parent_audit["formal_success"] is False,
        "Wrong parent direction-scan result.",
    )
    prerequisite = None
    if args.mode == "technical-preflight":
        core.require(args.preflight_root is None, "Preflight cannot consume another preflight.")
    else:
        core.require(
            args.preflight_root is not None
            and args.preflight_root.resolve() != args.output_root.resolve(),
            "Formal requires a distinct terminal preflight root.",
        )
        prerequisite = core.audit_delivery(args.preflight_root, config)
        core.require(
            prerequisite["mode"] == "technical-preflight"
            and prerequisite["git_commit"] == args.expected_commit,
            "Wrong terminal preflight prerequisite.",
        )
        prerequisite = {
            **prerequisite,
            "root": str(args.preflight_root.resolve()),
            "terminal_sha256": core.sha256_file(args.preflight_root / "terminal.json"),
            "inventory_sha256": core.sha256_file(
                args.preflight_root / "artifact_inventory.json"
            ),
            "tensor_bundle_sha256": core.sha256_file(
                args.preflight_root / "terminal_path_tensors.pt"
            ),
        }
    return {
        **base_validation,
        "base_direction_config": {
            "path": str(direction_core.CONFIG_PATH),
            "byte_sha256": contract["config_byte_sha256"],
            "canonical_sha256": contract["config_canonical_sha256"],
        },
        "parent_direction_audit": parent_audit,
        "parent_bindings": {
            name: {"path": str(path), "sha256": expected_hash}
            for name, (path, expected_hash) in bindings.items()
        },
        "preflight_prerequisite": prerequisite,
    }


def load_runtime(args: argparse.Namespace, counters: dict[str, int]):
    direction_config = direction_core.load_config()
    return direction_runner.load_runtime(args, direction_config, counters)


def candidate_record(
    *,
    pass_name: str,
    point_index: int,
    point: torch.Tensor,
    endpoint: torch.Tensor,
    teacher_x_t: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    plateau_mse: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    statistics = direction_core.quantization_statistics(point, teacher_x_t)
    binding = config["terminal_path_contract"]["point_bindings"][point_index]
    loss = float((endpoint - teacher_endpoint).square().mean())
    return {
        "schema": f"{core.PREFIX}-scan-row.v1",
        "protocol": core.PROTOCOL,
        "row_id": core.scan_row_id(pass_name, point_index),
        "pass": pass_name,
        "point_index": point_index,
        "requested_remaining_l2": binding["requested_remaining_l2"],
        "actual_fp32_l2_to_teacher": statistics["fp32_l2_to_teacher"],
        "bf16_l2_to_teacher": statistics["bf16_l2_to_teacher"],
        "bf16_equal_fraction_to_teacher": statistics["bf16_equal_fraction_to_teacher"],
        "x_T_fp32_sha256": canonical_tensor_sha256(point),
        "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
        "endpoint_bitwise_equal_to_teacher": torch.equal(endpoint, teacher_endpoint),
        "loss": loss,
        "loss_ratio_to_plateau": loss / plateau_mse,
    }


def scan_formal(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    oracle: phase1a.FrozenDreamLiteOracle,
    points: list[torch.Tensor],
    target: Mapping[str, torch.Tensor],
    plateau_mse: float,
    counters: dict[str, int],
    forward: Any,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    endpoints: dict[str, torch.Tensor] = {}
    orders = {
        "teacher-outward": range(len(points)),
        "plateau-inward": range(len(points) - 1, -1, -1),
    }
    metrics_path = args.output_root / "scan_metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics:
        for pass_name in core.PASSES:
            for point_index in orders[pass_name]:
                point = points[point_index]
                with torch.no_grad():
                    oracle.x_T_fp32.copy_(point.to(oracle.x_T_fp32.device))
                    oracle.initial_x_T_fp32.copy_(point.to(oracle.initial_x_T_fp32.device))
                output = forward(gradient=False)
                endpoint = output.z_t.detach().float().cpu()
                row = candidate_record(
                    pass_name=pass_name,
                    point_index=point_index,
                    point=point,
                    endpoint=endpoint,
                    teacher_x_t=target["teacher_x_T_fp32"],
                    teacher_endpoint=target["teacher_endpoint_fp32"],
                    plateau_mse=plateau_mse,
                    config=config,
                )
                rows.append(row)
                endpoints[row["row_id"]] = endpoint
                metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                metrics.flush()
                os.fsync(metrics.fileno())
                phase1a._save_image(
                    args.output_root / f"scan_images/{row['row_id']}.png", output.image
                )
                print(
                    f"terminal {len(rows)}/34 {row['row_id']} "
                    f"ratio={row['loss_ratio_to_plateau']:.10g}",
                    flush=True,
                )
    return rows, endpoints


def replay_preflight(
    *,
    oracle: phase1a.FrozenDreamLiteOracle,
    target: Mapping[str, torch.Tensor],
    plateau_checkpoint: Mapping[str, Any],
    forward: Any,
    output_root: Path,
) -> dict[str, torch.Tensor]:
    teacher_endpoint = target["teacher_endpoint_fp32"]
    endpoints = {"teacher-replay": teacher_endpoint.clone()}
    plateau_x_t = plateau_checkpoint["student_x_T_fp32"]
    with torch.no_grad():
        oracle.x_T_fp32.copy_(plateau_x_t.to(oracle.x_T_fp32.device))
        oracle.initial_x_T_fp32.copy_(plateau_x_t.to(oracle.initial_x_T_fp32.device))
    output = forward(gradient=False)
    endpoint = output.z_t.detach().float().cpu()
    core.require(
        torch.equal(endpoint, plateau_checkpoint["endpoint_fp32"]),
        "Terminal plateau preflight replay drift.",
    )
    endpoints["plateau-replay"] = endpoint
    phase1a._save_image(output_root / "preflight/plateau-replay.png", output.image)
    return endpoints


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# R11_new oracle terminal-capture scan",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
        f"- Counters: `{json.dumps(result['counters'], sort_keys=True)}`",
    ]
    if result["mode"] == "formal":
        lines += [
            f"- Classification: `{result['classification']}`",
            "- Strong capture prefix max requested remaining L2: "
            f"`{result['scan_summary']['strong_capture_prefix_max_requested_remaining_l2']}`",
        ]
    lines += [
        "",
        "This uses exact oracle teacher xT and is a fixed-target numerical diagnostic.",
        "Formal Picture Memory success and Phase 2 remain false.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(
    args: argparse.Namespace, config: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    phase1a._atomic_json(args.output_root / "config.json", config)
    phase1a._write_environment(args.output_root / "environment.txt")
    runtime = {
        **phase1a._runtime_versions(),
        "python_executable": str(Path(sys.executable).resolve()),
    }
    phase1a._atomic_json(args.output_root / "runtime.json", runtime)
    phase1a._atomic_json(
        args.output_root / "determinism.json", configure_strict_cuda_determinism(0)
    )
    direction_config = direction_core.load_config()
    snapshots_start = {
        name: phase1a.verify_snapshot_binding(value)
        for name, value in direction_config["model_snapshots"].items()
    }
    phase1a._atomic_json(
        args.output_root / "model_snapshot_verification_start.json", snapshots_start
    )
    counters = {
        "full_chain_forward_calls": 0,
        "reader_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    started = time.monotonic()
    (
        reader,
        oracle,
        source_latents,
        context,
        target,
        _alpha_checkpoint,
        plateau_checkpoint,
        target_record,
        initialization,
        forward,
    ) = load_runtime(args, counters)

    def forbidden_reader(_module: Any, _inputs: Any) -> None:
        counters["reader_forward_calls"] += 1
        raise RuntimeError("Reader is forbidden in the terminal-capture scan.")

    reader_hook = reader.model.register_forward_pre_hook(forbidden_reader)
    try:
        core.require(
            low_lr_runner.models_frozen(oracle, reader)
            and low_lr_runner.only_student_trainable(oracle, reader),
            "Initial terminal frozen/trainable contract failed.",
        )
        fixed = config["fixed_parent_artifacts"]
        core.require(
            core.sha256_file(args.output_root / "target/fixed_parent_target.pt")
            == fixed["source_target_artifact_sha256"]
            and core.sha256_file(args.output_root / "parent/lr001-step256.pt")
            == fixed["source_plateau_checkpoint_sha256"],
            "Copied terminal parent bytes drift.",
        )
        teacher_x_t = target["teacher_x_T_fp32"]
        plateau_x_t = plateau_checkpoint["student_x_T_fp32"]
        points = core.construct_points(teacher_x_t, plateau_x_t, config)
        teacher_endpoint = target["teacher_endpoint_fp32"]
        plateau_endpoint = plateau_checkpoint["endpoint_fp32"]
        plateau_mse = float((plateau_endpoint - teacher_endpoint).square().mean())
        core.require(
            math.isclose(
                plateau_mse, fixed["plateau_endpoint_mse"], rel_tol=1e-6, abs_tol=1e-10
            ),
            "Terminal parent plateau MSE drift.",
        )
        manifest = {
            "schema": f"{core.PREFIX}-manifest.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "config_sha256": core.canonical_sha(config),
            "validation": dict(validation),
            "target": target_record,
            "initialization": initialization,
            "target_segment_id": context["target"].segment_id,
            "path_information_boundary": {
                "teacher_x_T_exposed": True,
                "deployable_writer_information": False,
                "new_endpoint_losses_used_to_select_grid": False,
                "optimizer_steps": 0,
                "reader_used": False,
            },
            "formal_success": False,
            "phase2_allowed": False,
        }
        phase1a._atomic_json(args.output_root / "manifest.json", manifest)
        rows: list[dict[str, Any]] = []
        endpoints: dict[str, torch.Tensor]
        if args.mode == "technical-preflight":
            endpoints = replay_preflight(
                oracle=oracle,
                target=target,
                plateau_checkpoint=plateau_checkpoint,
                forward=forward,
                output_root=args.output_root,
            )
        else:
            rows, endpoints = scan_formal(
                args=args,
                config=config,
                oracle=oracle,
                points=points,
                target=target,
                plateau_mse=plateau_mse,
                counters=counters,
                forward=forward,
            )
        tensor_bundle = {
            "schema": f"{core.PREFIX}-tensor-bundle.v1",
            "protocol": core.PROTOCOL,
            "candidates": points,
            "scan_endpoints": endpoints,
        }
        phase1a._atomic_torch_save(
            args.output_root / "terminal_path_tensors.pt", tensor_bundle
        )
        snapshots_end = {
            name: phase1a.verify_snapshot_binding(value)
            for name, value in direction_config["model_snapshots"].items()
        }
        phase1a._atomic_json(
            args.output_root / "model_snapshot_verification_end.json", snapshots_end
        )
        if args.mode == "technical-preflight":
            expected_counters = {
                "full_chain_forward_calls": 2,
                "reader_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            engineering_gate = bool(
                counters == expected_counters
                and torch.equal(endpoints["teacher-replay"], teacher_endpoint)
                and torch.equal(endpoints["plateau-replay"], plateau_endpoint)
                and snapshots_end == snapshots_start
                and low_lr_runner.models_frozen(oracle, reader)
                and low_lr_runner.only_student_trainable(oracle, reader)
            )
            core.require(engineering_gate, "Terminal technical preflight gate failed.")
            scan_summary = None
            classification = None
        else:
            expected_counters = {
                "full_chain_forward_calls": 35,
                "reader_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            engineering_gate = bool(
                counters == expected_counters
                and len(rows) == len(endpoints) == 34
                and snapshots_end == snapshots_start
                and low_lr_runner.models_frozen(oracle, reader)
                and low_lr_runner.only_student_trainable(oracle, reader)
            )
            core.require(engineering_gate, "Terminal formal technical gate failed.")
            scan_summary = core.summarize_scan(rows, config)
            classification = core.classify_outcome(scan_summary, config)
        result = {
            "schema": f"{core.PREFIX}-result.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "engineering_gate": engineering_gate,
            "preflight_gate": engineering_gate if args.mode == "technical-preflight" else None,
            "plateau_endpoint_mse": plateau_mse,
            "scan_summary": scan_summary,
            "classification": classification,
            "counters": counters,
            "target_record": target_record,
            "tensor_bundle_sha256": core.sha256_file(
                args.output_root / "terminal_path_tensors.pt"
            ),
            "models_frozen": True,
            "only_student_x_T_trainable": True,
            "snapshots_unchanged": snapshots_end == snapshots_start,
            "formal_success": False,
            "phase2_allowed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        phase1a._atomic_json(args.output_root / "result.json", result)
        write_report(args.output_root, result)
        return result
    finally:
        reader_hook.remove()
        del source_latents


def artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]
    return {
        "schema": f"{core.PREFIX}-inventory.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = core.load_config()
    if args.mode == "audit":
        print(
            json.dumps(
                core.audit_delivery(args.output_root, config),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    validation = validate_environment(args, config)
    lock_path = Path("/tmp/vision-memory-r11-new-oracle-terminal-capture.lock")
    lock_path.mkdir(exist_ok=False)
    owner = {
        "owner_token": uuid.uuid4().hex,
        "pid": os.getpid(),
        "mode": args.mode,
        "git_commit": args.expected_commit,
        "output_root": str(args.output_root),
        "started_at_utc": safety._utc_now(),
    }
    phase1a._atomic_json(lock_path / "owner.json", owner)
    lock = {
        "path": str(lock_path),
        "owner_path": str(lock_path / "owner.json"),
        "owner": owner,
        "owner_sha256": core.sha256_file(lock_path / "owner.json"),
    }
    claimed = False
    terminal = {
        "schema": f"{core.PREFIX}-terminal.v1",
        "protocol": core.PROTOCOL,
        "mode": args.mode,
        "status": "technical_failed",
        "engineering_gate": False,
        "exit_code": 2,
        "git_commit": args.expected_commit,
        "formal_success": False,
        "phase2_allowed": False,
        "started_at_utc": safety._utc_now(),
    }
    try:
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        phase1a._atomic_json(
            args.output_root / "launch.json", {"owner": owner, "validation": validation}
        )
        with (args.output_root / "stdout.log").open(
            "x", encoding="utf-8"
        ) as stdout, (args.output_root / "stderr.log").open(
            "x", encoding="utf-8"
        ) as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result = run_experiment(args, config, validation)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(
            status="technical_completed",
            engineering_gate=True,
            exit_code=0,
            result_sha256=core.sha256_file(args.output_root / "result.json"),
            classification=result.get("classification"),
        )
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            terminal["lock_release"] = safety._release_lock(lock)
        except Exception as error:
            terminal.update(
                status="technical_failed",
                engineering_gate=False,
                exit_code=2,
                lock_release_error=f"{type(error).__name__}: {error}",
            )
        if claimed:
            terminal["completed_at_utc"] = safety._utc_now()
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            phase1a._atomic_json(
                args.output_root / "artifact_inventory.json",
                artifact_inventory(args.output_root),
            )
            if terminal["exit_code"] == 0:
                core.audit_delivery(args.output_root, config)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
