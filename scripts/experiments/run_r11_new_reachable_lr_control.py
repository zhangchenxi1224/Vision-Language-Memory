"""Run or independently audit the preregistered R11_new reachable low-LR control."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
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

from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from scripts.train import r11_new_identity_condition_binding as condition_binding  # noqa: E402
from scripts.train import r11_new_identity_condition_bridge as identity  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import r11_new_reachable_lr_control as core  # noqa: E402


CHECKPOINT_SCHEMA = f"{core.PREFIX}-checkpoint.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("technical-preflight", "formal", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-root", type=Path)
    return parser.parse_args(argv)


def runtime_args(output_root: Path, config: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        train=Path(config["data"]["train_path"]),
        dev=Path(config["data"]["dev_path"]),
        dreamlite=Path(config["models"]["dreamlite"]),
        reader=Path(config["models"]["reader"]),
        output_dir=output_root,
        dreamlite_device="cuda:0",
        reader_device="cuda:1",
        strict_determinism=True,
        allow_dirty=False,
        target_index=1,
        seed=phase1a.SEED,
        adapter_seed=phase1a.SEED,
        pairing_seed=0,
        split_seed=20260730,
        schedule_seed=phase1a.R10_SELECTION_SEED,
        bootstrap_iterations=10_000,
        resolution=phase1a.RESOLUTION,
        lora_rank=0,
        selected_step_count=0,
        gradient_mode="full",
    )


def validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    deployment = safety._deployment_audit(args.output_root)
    core.require(bool(safety._COMMIT_RE.fullmatch(args.expected_commit or "")), "Full expected Git SHA required.")
    core.require(safety._git("rev-parse", "HEAD") == args.expected_commit
                 and not safety._git("status", "--porcelain") and not safety._git("branch", "--show-current"),
                 "Clean detached expected checkout required.")
    observed_environment = {name: os.environ.get(name) for name in config["environment"]}
    core.require(observed_environment == config["environment"], "Deterministic/offline environment drift.")
    core.require(sys.version_info[:2] == (3, 12) and torch.__version__.startswith("2.7.0a0+ecf3bae40a")
                 and torch.version.cuda == "12.8", "Pinned NGC runtime drift.")
    core.require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported()
                 and all("H200" in torch.cuda.get_device_name(index).upper() for index in range(2)),
                 "Exactly two visible native-BF16 H200 GPUs required.")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    core.require(all(line.strip() in ("", str(os.getpid())) for line in active), "GPU occupied; do not interrupt it.")
    locks = list(Path("/tmp").glob("*r11*lock*"))
    core.require(not locks, f"Existing R11 lock; never remove automatically: {locks}")
    core.require(core.sha256_file(Path(config["data"]["train_path"])) == config["data"]["train_sha256"]
                 and core.sha256_file(Path(config["data"]["dev_path"])) == config["data"]["dev_sha256"],
                 "Fixed train/dev data drift.")
    identity_config = identity._load_config()
    core.require(core.sha256_file(identity.CONFIG_PATH)
                 == config["fixed_inputs"]["identity_parent_config_file_sha256"], "Identity parent config drift.")
    core.require(identity_config["single_changed_solver_factor"]["new_value"]
                 == config["fixed_inputs"]["actual_conditioning_text"], "Identity condition drift.")
    parent = config["parent_basin_failure"]
    parent_root = Path(parent["source_root"])
    parent_archive = parent_root.parent / "r11-new-basin-7d4a845-20260907-round01.tar.gz"
    bindings = {
        "target": (parent_root / "target/fixed_parent_target.pt", parent["source_target_artifact_sha256"]),
        "result": (parent_root / "result.json", parent["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", parent["source_manifest_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", parent["source_inventory_sha256"]),
        "archive": (parent_archive, parent["source_archive_sha256"]),
    }
    for name, (path, expected_hash) in bindings.items():
        core.require(path.is_file() and core.sha256_file(path) == expected_hash, f"Parent {name} binding drift.")
    snapshots = {name: phase1a.verify_snapshot_binding(value) for name, value in config["model_snapshots"].items()}
    prerequisite = None
    if args.mode == "technical-preflight":
        core.require(args.preflight_root is None, "Preflight cannot consume another preflight.")
    else:
        core.require(args.preflight_root is not None and args.preflight_root.resolve() != args.output_root.resolve(),
                     "Formal requires a distinct preflight root.")
        prerequisite = core.audit_delivery(args.preflight_root, config)
        core.require(prerequisite["mode"] == "technical-preflight"
                     and prerequisite["git_commit"] == args.expected_commit, "Wrong preflight prerequisite.")
        preflight_target = torch.load(
            args.preflight_root / "target/fixed_parent_target.pt", map_location="cpu", weights_only=True
        )
        prerequisite = {
            **prerequisite,
            "root": str(args.preflight_root.resolve()),
            "terminal_sha256": core.sha256_file(args.preflight_root / "terminal.json"),
            "inventory_sha256": core.sha256_file(args.preflight_root / "artifact_inventory.json"),
            "target_artifact_sha256": core.sha256_file(args.preflight_root / "target/fixed_parent_target.pt"),
            "target_tensor_sha256": dict(preflight_target["tensor_sha256"]),
        }
    return {
        "deployment": deployment,
        "git_commit": args.expected_commit,
        "git_clean": True,
        "git_detached": True,
        "environment": observed_environment,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(2)],
        "snapshots": snapshots,
        "parent_bindings": {name: {"path": str(path), "sha256": expected_hash}
                            for name, (path, expected_hash) in bindings.items()},
        "preflight_prerequisite": prerequisite,
    }


def models_frozen(oracle: phase1a.FrozenDreamLiteOracle, reader: torch.nn.Module) -> bool:
    modules = (oracle.unet, oracle.vae, oracle.text_encoder, reader)
    return all(not module.training and all(not parameter.requires_grad and parameter.grad is None
                                          for parameter in module.parameters()) for module in modules)


def only_student_trainable(oracle: phase1a.FrozenDreamLiteOracle, reader: torch.nn.Module) -> bool:
    trainable = [(name, parameter) for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    return len(trainable) == 1 and trainable[0][1] is oracle.x_T_fp32 and not any(
        parameter.requires_grad for parameter in reader.parameters()
    )


def save_checkpoint(root: Path, *, condition_name: str, base_learning_rate: float,
                    initialization_alpha: float, step: int,
                    oracle: phase1a.FrozenDreamLiteOracle, optimizer: torch.optim.Optimizer,
                    output: phase1a.OracleForward, teacher_endpoint: torch.Tensor,
                    m0_mse: float) -> dict[str, Any]:
    condition_root = root / "conditions" / condition_name
    student = oracle.x_T_fp32.detach().float().cpu()
    endpoint = output.z_t.detach().float().cpu()
    trajectory = tuple(value.detach().float().cpu() for value in output.trajectory)
    mse = float((endpoint - teacher_endpoint).square().mean())
    distance = core.distance_statistics(mse=mse, m0_mse=m0_mse, tensor_numel=endpoint.numel())
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": core.PROTOCOL,
        "condition": condition_name,
        "base_learning_rate": base_learning_rate,
        "initialization_alpha": initialization_alpha,
        "optimizer_step": step,
        "student_x_T_fp32": student,
        "endpoint_fp32": endpoint,
        "trajectory_fp32": trajectory,
        "effective_sigmas": list(output.effective_sigmas),
        "distance": distance,
        "optimizer": optimizer.state_dict(),
        "tensor_sha256": {
            "student_x_T_fp32": canonical_tensor_sha256(student),
            "endpoint_fp32": canonical_tensor_sha256(endpoint),
            "trajectory_fp32": [canonical_tensor_sha256(value) for value in trajectory],
        },
    }
    checkpoint_path = condition_root / f"checkpoints/step-{step:03d}.pt"
    phase1a._atomic_torch_save(checkpoint_path, payload)
    image_path = condition_root / f"images/step-{step:03d}.png"
    phase1a._save_image(image_path, output.image)
    return {
        "optimizer_step": step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": core.sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "image_path": str(image_path),
        "image_sha256": core.sha256_file(image_path),
        "image_bytes": image_path.stat().st_size,
        "distance": distance,
        "tensor_sha256": payload["tensor_sha256"],
    }


def load_fixed_runtime(args: argparse.Namespace, config: Mapping[str, Any], counters: dict[str, int]):
    identity_config = identity._load_config()
    load_args = runtime_args(args.output_root, config)
    processor, pipe, reader, oracle, source_latents, context = condition_binding.load_runtime(
        load_args, identity_config
    )
    del processor, pipe
    initial_x_t, initialization = identity._source_only_initialization(
        source_latents=source_latents,
        scheduler=oracle.sampler,
        compute_device=oracle.source_latents.device,
        output_dir=args.output_root,
    )
    parent_path = Path(config["parent_basin_failure"]["source_target_path"])
    copied_path = args.output_root / "target/fixed_parent_target.pt"
    copied_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_path, copied_path)
    core.require(core.sha256_file(copied_path)
                 == config["parent_basin_failure"]["source_target_artifact_sha256"],
                 "Copied parent target bytes drifted.")
    target = torch.load(copied_path, map_location="cpu", weights_only=True)
    core.require(target.get("tensor_sha256") == config["parent_basin_failure"]["target_tensor_sha256"],
                 "Copied parent target tensor map drifted.")
    for name in core.TARGET_TENSOR_NAMES:
        tensor = target.get(name)
        core.require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                     and canonical_tensor_sha256(tensor) == target["tensor_sha256"][name],
                     f"Copied parent tensor drifted: {name}.")
    source = target["source_only_x_T_init_fp32"]
    teacher_x_t = target["teacher_x_T_fp32"]
    teacher_endpoint = target["teacher_endpoint_fp32"]
    core.require(torch.equal(initial_x_t.detach().float().cpu(), source),
                 "Current source-only initialization differs from fixed parent target.")
    core.require(torch.equal(teacher_x_t, source + target["normalized_noise_fp32"]),
                 "Fixed parent target formula drifted.")

    def forward(*, gradient: bool) -> phase1a.OracleForward:
        counters["full_chain_forward_calls"] += 1
        if gradient:
            return oracle()
        with torch.no_grad():
            return oracle()

    with torch.no_grad():
        oracle.x_T_fp32.copy_(teacher_x_t.to(oracle.x_T_fp32.device))
        oracle.initial_x_T_fp32.copy_(teacher_x_t.to(oracle.initial_x_T_fp32.device))
    replay = forward(gradient=False)
    replay_equal = bool(torch.equal(replay.z_t.detach().float().cpu(), teacher_endpoint))
    core.require(replay_equal, "Fixed teacher endpoint does not replay bitwise.")
    phase1a._save_image(args.output_root / "target/fixed_parent_target.png", replay.image)
    target_record = {
        "path": str(copied_path),
        "artifact_sha256": core.sha256_file(copied_path),
        "bytes": copied_path.stat().st_size,
        "tensor_sha256": dict(target["tensor_sha256"]),
        "teacher_endpoint_replay_bitwise_equal": replay_equal,
        "source_parent_root": config["parent_basin_failure"]["source_root"],
    }
    return reader, oracle, source_latents, context, target, target_record, initialization, forward


def train_condition(*, args: argparse.Namespace, config: Mapping[str, Any], condition: Mapping[str, Any],
                    oracle: phase1a.FrozenDreamLiteOracle, reader: torch.nn.Module,
                    target: Mapping[str, torch.Tensor], counters: dict[str, int], forward: Any) -> dict[str, Any]:
    name = str(condition["name"])
    base_learning_rate = float(condition["base_learning_rate"])
    alpha = float(config["fixed_target_and_initialization_contract"]["initialization_alpha"])
    teacher_endpoint = target["teacher_endpoint_fp32"]
    start = core.warm_start(target["source_only_x_T_init_fp32"], target["teacher_x_T_fp32"], alpha)
    with torch.no_grad():
        oracle.x_T_fp32.copy_(start.to(oracle.x_T_fp32.device))
        oracle.initial_x_T_fp32.copy_(start.to(oracle.initial_x_T_fp32.device))
    initial_output = forward(gradient=False)
    m0_mse = float((initial_output.z_t.detach().float().cpu() - teacher_endpoint).square().mean())
    core.require(m0_mse >= config["technical_preflight_gate"]["minimum_initial_endpoint_mse"],
                 f"Initial endpoint is trivial or invalid for {name}.")
    optimizer = torch.optim.Adam(
        (oracle.x_T_fp32,),
        lr=base_learning_rate,
        weight_decay=config["optimizer_contract"]["weight_decay"],
    )
    checkpoints = [save_checkpoint(
        args.output_root, condition_name=name, base_learning_rate=base_learning_rate,
        initialization_alpha=alpha, step=0, oracle=oracle, optimizer=optimizer,
        output=initial_output, teacher_endpoint=teacher_endpoint, m0_mse=m0_mse
    )]
    if args.mode == "technical-preflight":
        optimizer.zero_grad(set_to_none=True)
        probe_output = forward(gradient=True)
        loss = (probe_output.z_t.float() - teacher_endpoint.to(probe_output.z_t.device)).square().mean()
        loss.backward()
        counters["backward_calls"] += 1
        gradient = oracle.x_T_fp32.grad
        core.require(gradient is not None and bool(torch.isfinite(gradient).all()), "Preflight gradient invalid.")
        gradient_cpu = gradient.detach().float().cpu()
        probe = {
            "schema": f"{core.PREFIX}-gradient-probe.v1",
            "condition": name,
            "base_learning_rate": base_learning_rate,
            "initialization_alpha": alpha,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient.double().norm()),
            "gradient_nonzero_fraction": float((gradient != 0).double().mean()),
            "gradient_fp32_sha256": canonical_tensor_sha256(gradient_cpu),
        }
        optimizer.zero_grad(set_to_none=True)
        phase1a._atomic_json(args.output_root / "conditions" / name / "gradient_probe.json", probe)
        return {"base_learning_rate": base_learning_rate, "initialization_alpha": alpha,
                "m0_mse": m0_mse, "gradient_probe": probe,
                "checkpoint_records": checkpoints}

    metrics_path = args.output_root / "conditions" / name / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics:
        for update in range(1, config["optimizer_contract"]["optimizer_steps_per_condition"] + 1):
            learning_rate = core.optimizer_learning_rate(update, name, config)
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            output = forward(gradient=True)
            loss = (output.z_t.float() - teacher_endpoint.to(output.z_t.device)).square().mean()
            core.require(loss.numel() == 1 and bool(torch.isfinite(loss)), "Invalid training loss.")
            loss.backward()
            counters["backward_calls"] += 1
            gradient = oracle.x_T_fp32.grad
            core.require(gradient is not None and bool(torch.isfinite(gradient).all()), "Invalid x_T gradient.")
            gradient_norm = float(gradient.double().norm())
            nonzero_fraction = float((gradient != 0).double().mean())
            core.require(gradient_norm > 0.0 and nonzero_fraction > 0.0, "Zero x_T gradient.")
            core.require(models_frozen(oracle, reader), "Frozen model received a gradient.")
            before = oracle.x_T_fp32.detach().clone()
            optimizer.step()
            counters["optimizer_steps"] += 1
            after = oracle.x_T_fp32.detach()
            optimizer.zero_grad(set_to_none=True)
            row = {
                "schema": f"{core.PREFIX}-metrics.v1",
                "protocol": core.PROTOCOL,
                "condition": name,
                "base_learning_rate": base_learning_rate,
                "initialization_alpha": alpha,
                "optimizer_step": update,
                "learning_rate": learning_rate,
                "loss_before_step": float(loss.detach()),
                "gradient_norm": gradient_norm,
                "gradient_nonzero_fraction": nonzero_fraction,
                "x_T_update_norm": float((after - before).double().norm()),
                "gradient_clipping_applied": False,
                "optimizer_step_applied": True,
                "teacher_endpoint_sha256": target["tensor_sha256"]["teacher_endpoint_fp32"],
            }
            metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            if update in config["optimizer_contract"]["checkpoint_steps"][1:]:
                endpoint = forward(gradient=False)
                checkpoints.append(save_checkpoint(
                    args.output_root, condition_name=name, base_learning_rate=base_learning_rate,
                    initialization_alpha=alpha, step=update, oracle=oracle,
                    optimizer=optimizer, output=endpoint, teacher_endpoint=teacher_endpoint, m0_mse=m0_mse
                ))
            print(f"{name} receipt {update}/256 loss={float(loss.detach()):.10g} lr={learning_rate:.10g}",
                  flush=True)
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    technical_audit = core.validate_metrics(rows, config, name)
    endpoint_distance = checkpoints[-1]["distance"]
    gate = core.per_condition_gate(endpoint_distance, technical_gate=True, config=config)
    return {
        "base_learning_rate": base_learning_rate,
        "initialization_alpha": alpha,
        "m0_mse": m0_mse,
        "technical_audit": technical_audit,
        "checkpoint_records": checkpoints,
        "endpoint_distance": endpoint_distance,
        "condition_gate": gate,
    }


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# R11_new reachable low-LR control",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
    ]
    for name, record in result["conditions"].items():
        lines.append(f"- {name} M0 MSE: `{record['m0_mse']:.10g}`")
        if result["mode"] == "technical-preflight":
            lines.append(f"- {name} gradient norm: `{record['gradient_probe']['gradient_norm']:.10g}`")
        else:
            lines.append(f"- {name} raw256 MSE/M0: `{record['endpoint_distance']['mse_ratio_to_m0']:.10g}`")
            lines.append(f"- {name} gate: `{record['condition_gate']}`")
    if result["mode"] == "formal":
        lines.append(f"- Classification: `{result['classification']}`")
    lines += ["", "The target is fixed from the parent reachable control and is non-semantic.",
              "This isolates optimizer scale at the fixed alpha=0.99 initialization.",
              "Formal Picture Memory success and Phase 2 remain false."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(args: argparse.Namespace, config: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    phase1a._atomic_json(args.output_root / "config.json", config)
    phase1a._write_environment(args.output_root / "environment.txt")
    phase1a._atomic_json(args.output_root / "runtime.json", phase1a._runtime_versions())
    phase1a._atomic_json(args.output_root / "determinism.json", configure_strict_cuda_determinism(0))
    snapshots_start = {name: phase1a.verify_snapshot_binding(value)
                       for name, value in config["model_snapshots"].items()}
    phase1a._atomic_json(args.output_root / "model_snapshot_verification_start.json", snapshots_start)
    counters = {"full_chain_forward_calls": 0, "reader_forward_calls": 0,
                "backward_calls": 0, "optimizer_steps": 0}
    started = time.monotonic()
    reader, oracle, source_latents, context, target, target_record, initialization, forward = load_fixed_runtime(
        args, config, counters
    )

    def forbidden_reader(_module: Any, _inputs: Any) -> None:
        counters["reader_forward_calls"] += 1
        raise RuntimeError("Reader is forbidden in the numerical reachable low-LR control.")

    reader_hook = reader.model.register_forward_pre_hook(forbidden_reader)
    try:
        core.require(models_frozen(oracle, reader) and only_student_trainable(oracle, reader),
                     "Frozen/trainable parameter contract failed.")
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
            "actual_conditioning_text": config["fixed_inputs"]["actual_conditioning_text"],
            "condition_order": config["optimizer_contract"]["condition_order"],
            "information_boundary": {
                "target_source": "exact parent reachable_target.pt artifact",
                "target_regeneration": False,
                "teacher_x_T_exposed_to_oracle_warm_start": True,
                "student_loss_inputs": ["condition-specific student endpoint", "fixed detached teacher endpoint"],
                "teacher_x_T_exposed_to_loss": False,
                "reader_used": False,
            },
            "formal_success": False,
            "phase2_allowed": False,
        }
        phase1a._atomic_json(args.output_root / "manifest.json", manifest)
        condition_results = {}
        for condition in config["learning_rate_conditions"]:
            condition_results[condition["name"]] = train_condition(
                args=args, config=config, condition=condition, oracle=oracle, reader=reader,
                target=target, counters=counters, forward=forward
            )
        snapshots_end = {name: phase1a.verify_snapshot_binding(value)
                         for name, value in config["model_snapshots"].items()}
        phase1a._atomic_json(args.output_root / "model_snapshot_verification_end.json", snapshots_end)
        if args.mode == "technical-preflight":
            preflight = config["technical_preflight_gate"]
            expected_counters = {"full_chain_forward_calls": 5, "reader_forward_calls": 0,
                                 "backward_calls": 2, "optimizer_steps": 0}
            initial_hashes = {
                record["checkpoint_records"][0]["tensor_sha256"]["endpoint_fp32"]
                for record in condition_results.values()
            }
            m0_values = {record["m0_mse"] for record in condition_results.values()}
            probe_signatures = {
                (record["gradient_probe"]["loss"], record["gradient_probe"]["gradient_norm"],
                 record["gradient_probe"]["gradient_nonzero_fraction"],
                 record["gradient_probe"]["gradient_fp32_sha256"])
                for record in condition_results.values()
            }
            preflight_gate = bool(
                counters == expected_counters and snapshots_end == snapshots_start
                and len(initial_hashes) == len(m0_values) == len(probe_signatures) == 1
                and all(record["m0_mse"] >= preflight["minimum_initial_endpoint_mse"]
                        and record["gradient_probe"]["gradient_norm"] > preflight["gradient_norm_gt"]
                        and record["gradient_probe"]["gradient_nonzero_fraction"]
                        >= preflight["gradient_nonzero_fraction_gte"]
                        for record in condition_results.values())
                and models_frozen(oracle, reader) and only_student_trainable(oracle, reader)
            )
            result = {
                "schema": f"{core.PREFIX}-result.v1",
                "protocol": core.PROTOCOL,
                "mode": args.mode,
                "git_commit": args.expected_commit,
                "engineering_gate": preflight_gate,
                "preflight_gate": preflight_gate,
                "conditions": condition_results,
                "counters": counters,
                "target_record": target_record,
                "models_frozen": True,
                "only_student_x_T_trainable": True,
                "snapshots_unchanged": snapshots_end == snapshots_start,
                "condition_gates": {},
                "classification": None,
                "formal_success": False,
                "phase2_allowed": False,
                "elapsed_seconds": time.monotonic() - started,
            }
            core.require(preflight_gate, "Technical preflight gate failed.")
        else:
            expected_counters = {"full_chain_forward_calls": 523, "reader_forward_calls": 0,
                                 "backward_calls": 512, "optimizer_steps": 512}
            technical_gate = bool(
                counters == expected_counters and snapshots_end == snapshots_start
                and all([item["optimizer_step"] for item in record["checkpoint_records"]]
                        == config["optimizer_contract"]["checkpoint_steps"]
                        for record in condition_results.values())
                and models_frozen(oracle, reader) and only_student_trainable(oracle, reader)
            )
            core.require(technical_gate, "Formal technical gate failed.")
            gates = {name: record["condition_gate"] for name, record in condition_results.items()}
            classification = core.classify_outcome(gates)
            result = {
                "schema": f"{core.PREFIX}-result.v1",
                "protocol": core.PROTOCOL,
                "mode": args.mode,
                "git_commit": args.expected_commit,
                "engineering_gate": technical_gate,
                "conditions": condition_results,
                "counters": counters,
                "target_record": target_record,
                "models_frozen": True,
                "only_student_x_T_trainable": True,
                "snapshots_unchanged": snapshots_end == snapshots_start,
                "condition_gates": gates,
                "classification": classification,
                "formal_success": False,
                "phase2_allowed": False,
                "elapsed_seconds": time.monotonic() - started,
            }
        phase1a._atomic_json(args.output_root / "result.json", result)
        write_report(args.output_root, result)
        core.require(models_frozen(oracle, reader) and only_student_trainable(oracle, reader),
                     "Final frozen/trainable contract failed.")
        return result
    finally:
        reader_hook.remove()
        del source_latents


def artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": core.sha256_file(path)}
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"
    ]
    return {"schema": f"{core.PREFIX}-inventory.v1", "artifact_count": len(artifacts), "artifacts": artifacts}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = core.load_config()
    if args.mode == "audit":
        print(json.dumps(core.audit_delivery(args.output_root, config), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    validation = validate_environment(args, config)
    lock_path = Path("/tmp/vision-memory-r11-new-reachable-lr-control.lock")
    lock_path.mkdir(exist_ok=False)
    owner = {"owner_token": uuid.uuid4().hex, "pid": os.getpid(), "mode": args.mode,
             "git_commit": args.expected_commit, "output_root": str(args.output_root),
             "started_at_utc": safety._utc_now()}
    phase1a._atomic_json(lock_path / "owner.json", owner)
    lock = {"path": str(lock_path), "owner_path": str(lock_path / "owner.json"), "owner": owner,
            "owner_sha256": core.sha256_file(lock_path / "owner.json")}
    claimed = False
    terminal = {"schema": f"{core.PREFIX}-terminal.v1", "protocol": core.PROTOCOL, "mode": args.mode,
                "status": "technical_failed", "engineering_gate": False, "exit_code": 2,
                "git_commit": args.expected_commit, "formal_success": False, "phase2_allowed": False,
                "started_at_utc": safety._utc_now()}
    try:
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        phase1a._atomic_json(args.output_root / "launch.json", {"owner": owner, "validation": validation})
        with (args.output_root / "stdout.log").open("x", encoding="utf-8") as stdout, (
            args.output_root / "stderr.log"
        ).open("x", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result = run_experiment(args, config, validation)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(status="technical_completed", engineering_gate=True, exit_code=0,
                        result_sha256=core.sha256_file(args.output_root / "result.json"),
                        condition_gates=result.get("condition_gates", {}),
                        classification=result.get("classification"))
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            terminal["lock_release"] = safety._release_lock(lock)
        except Exception as error:
            terminal.update(status="technical_failed", engineering_gate=False, exit_code=2,
                            lock_release_error=f"{type(error).__name__}: {error}")
        if claimed:
            terminal["completed_at_utc"] = safety._utc_now()
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            phase1a._atomic_json(args.output_root / "artifact_inventory.json", artifact_inventory(args.output_root))
            if terminal["exit_code"] == 0:
                core.audit_delivery(args.output_root, config)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
