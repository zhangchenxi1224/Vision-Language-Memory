"""Lock native identity conditioning only; no U-Net, Reader, backward or optimizer."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.inspire import run_r11_new_source_init_bridge as safety  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.dreamlite.conditioning import (  # noqa: E402
    encode_latent_path_condition,
    official_mobile_edit_prompt,
)
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


CONFIG = ROOT / "configs/experiments/r11_new_identity_condition_probe.json"
CONFIG_SHA256 = "5f1db61ac1e7db179dd3cb39c463c891b3b7aba056fafbe9d6618066f4eab73e"
PROTOCOL = "R11-New-Identity-Condition-Technical-Probe-Target01"
SCHEMA_PREFIX = "vision_memory.r11-new-identity-condition-probe"
CONFIG_SCHEMA = f"{SCHEMA_PREFIX}-config.v1"
MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}-manifest.v1"
CONDITION_SCHEMA = f"{SCHEMA_PREFIX}-condition.v1"
SOURCE_SCHEMA = f"{SCHEMA_PREFIX}-source.v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}-result.v1"
TERMINAL_SCHEMA = f"{SCHEMA_PREFIX}-terminal.v1"
INVENTORY_SCHEMA = f"{SCHEMA_PREFIX}-inventory.v1"
LOCK_PATH = Path("/tmp/vision-memory-r11-new-identity-condition-probe.lock")
IDENTITY_TEXT = "no changes"
ZERO_COUNTERS = {
    "unet_forward_calls": 0, "full_dreamlite_forward_calls": 0,
    "reader_forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_config() -> dict[str, Any]:
    _require(safety._sha256(CONFIG) == CONFIG_SHA256, "Probe locked config file hash drifted.")
    config = safety._load(CONFIG)
    _require(config.get("schema") == CONFIG_SCHEMA and config.get("protocol") == PROTOCOL, "Probe identity drifted.")
    condition = config["condition"]
    _require(condition["actual_conditioning_text"] == IDENTITY_TEXT, "Probe permits only 'no changes'.")
    for name in ("original_event_text", "actual_conditioning_text", "full_prompt"):
        _require(_text_sha256(condition[name]) == condition[f"{name}_sha256"], f"Invalid {name} UTF8 hash.")
    _require(condition["full_prompt"] == official_mobile_edit_prompt(IDENTITY_TEXT), "Official template drifted.")
    _require(condition["repeat_count"] == 2 and condition["mode"] == "edit", "Condition call contract drifted.")
    return config


def _read_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, f"Parent hash drifted: {path}")
    value = json.loads(payload)
    _require(isinstance(value, dict), f"Expected parent JSON object: {path}")
    return value


def _validate_parents(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["parent_source_init"]
    root = Path(parent["source_root"])
    paths = {
        "comparison": root / "aggregation-v1/comparison.json",
        "raw_artifacts": root / "aggregation-v1/RAW_ARTIFACTS.json",
        "formal_manifest": root / "formal-target01/run/manifest.json",
        "phase1a_manifest": Path(config["parent_phase1a"]["source_root"]) / "run/manifest.json",
    }
    hashes = {name: parent[f"{name}_sha256"] for name in paths if name != "phase1a_manifest"}
    hashes["phase1a_manifest"] = config["parent_phase1a"]["manifest_sha256"]
    values = {name: _read_bound_json(path, hashes[name]) for name, path in paths.items()}
    comparison, manifest, original = (values[key] for key in ("comparison", "formal_manifest", "phase1a_manifest"))
    _require(comparison.get("protocol") == safety.core.BRIDGE_PROTOCOL, "Wrong source-init parent protocol.")
    _require(comparison.get("raw_artifacts_sha256") == hashes["raw_artifacts"], "Parent comparison/RAW mismatch.")
    for name in ("engineering_gate", "teacher_replay_gate", "bridge_distance_gate", "endpoint_reader_transfer_gate",
                 "formal_success", "phase2_allowed"):
        _require(comparison.get(name) is parent[name], f"Parent decision binding drifted: {name}")
    _require(comparison.get("git_commit") == parent["training_git_commit"], "Parent training commit drifted.")
    _require(manifest.get("protocol") == safety.core.BRIDGE_PROTOCOL, "Wrong source-init manifest protocol.")
    _require(manifest.get("preregistered_config_file_sha256") == parent["config_sha256"], "Parent config drifted.")
    _require(original.get("schema") == phase1a.MANIFEST_SCHEMA, "Wrong original target manifest schema.")
    target = original.get("target_segment", {})
    for value in (original, manifest, comparison):
        _require(value.get("target_index") == config["parent_phase1a"]["target_index"], "Parent target index drifted.")
        _require(value.get("target_segment_id") == config["parent_phase1a"]["target_segment_id"], "Parent target ID drifted.")
    _require(safety.core.canonical_json_sha256(manifest.get("target_segment")) ==
             safety.core.canonical_json_sha256(target), "Original target/event/query content drifted.")
    _require(len(target.get("events", [])) == 1, "Probe requires the fixed single original event.")
    _require(target["events"][0].get("event_text") == config["condition"]["original_event_text"], "Original event drifted.")
    _require(manifest["source_rgb"]["sha256"] == config["source"]["rgb_sha256"], "Parent source RGB drifted.")
    _require(manifest["source_latents"]["sha256"] == config["source"]["latents_fp32_sha256"], "Parent source latent drifted.")
    old = manifest["condition_artifact"]
    _require(old["event_text_sha256"] == config["condition"]["original_event_text_sha256"], "Parent event hash drifted.")
    for key, anchor in (("prompt_embeds", "old_prompt_embeds_sha256"), ("attention_mask", "old_attention_mask_sha256")):
        _require(old["tensor_sha256"][key] == config["condition"][anchor], "Parent conditioning anchor drifted.")
    return {
        "artifacts": {name: {"path": str(path), "sha256": hashes[name], "bytes": path.stat().st_size}
                      for name, path in paths.items()},
        "target_segment": target,
        "target_segment_canonical_sha256": safety.core.canonical_json_sha256(target),
        "original_event_unchanged": True,
    }


def _validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    deployment = safety._deployment_audit(args.output_root)
    _require(bool(safety._COMMIT_RE.fullmatch(args.expected_commit)), "Expected commit must be a full SHA1.")
    _require(safety._git("rev-parse", "HEAD") == args.expected_commit, "Probe checkout mismatch.")
    _require(not safety._git("status", "--porcelain"), "Probe requires a clean source tree.")
    _require(not safety._git("branch", "--show-current"), "Probe requires a detached checkout.")
    expected_environment = config["execution"]["environment"]
    observed = {key: os.environ.get(key) for key in expected_environment}
    _require(observed == expected_environment, "Probe deterministic/offline environment drifted.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "Probe requires exactly two visible H200 GPUs.")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    _require(all("H200" in name.upper() for name in gpu_names), "Probe requires H200 GPUs.")
    _require(torch.cuda.is_bf16_supported(), "Probe requires native CUDA BF16.")
    return {"deployment": deployment, "gpu_names": gpu_names, "environment": observed,
            "git_commit": args.expected_commit, "git_dirty": False, "git_detached": True}


def _claim_root(path: Path) -> None:
    _require(not path.exists(), "Probe requires a fresh output root.")
    path.mkdir(parents=True, exist_ok=False)


def _acquire_lock(output_root: Path, expected_commit: str) -> dict[str, Any]:
    owner = {"schema": f"{SCHEMA_PREFIX}-lock.v1", "owner_token": uuid.uuid4().hex,
             "pid": os.getpid(), "output_root": str(output_root), "git_commit": expected_commit,
             "acquired_at_utc": safety._utc_now()}
    LOCK_PATH.mkdir(exist_ok=False)
    owner_path = LOCK_PATH / "owner.json"
    try:
        phase1a._atomic_json(owner_path, owner)
    except Exception:
        LOCK_PATH.rmdir()
        raise
    return {"path": str(LOCK_PATH), "owner_path": str(owner_path), "owner": owner,
            "owner_sha256": safety._sha256(owner_path)}


def _metadata(config: Mapping[str, Any], parent_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{name: config["condition"][name] for name in (
            "original_event_text", "original_event_text_sha256", "actual_conditioning_text",
            "actual_conditioning_text_sha256", "full_prompt", "full_prompt_sha256", "mode")},
        "target_segment_canonical_sha256": parent_binding["target_segment_canonical_sha256"],
        "original_target_is_provenance_only": True,
        "conditioner_input_names": ["fixed_source_latents", "fixed_identity_text"],
        "compute_device": "cuda:0", "compute_dtype": "torch.bfloat16",
        "source_latents_fp32_sha256": config["source"]["latents_fp32_sha256"],
    }


def _encode_condition(pipe: Any, source_latents: Tensor, fixed_text: str) -> Any:
    # No target, teacher, query, answer or choice object can enter this interface.
    _require(fixed_text == IDENTITY_TEXT, "Condition-only probe permits exactly 'no changes'.")
    return encode_latent_path_condition(pipe, source_latents, fixed_text)


def _condition_tensor_checks(embeds: Tensor, mask: Tensor) -> None:
    _require(isinstance(embeds, Tensor) and isinstance(mask, Tensor), "Condition values must be raw tensors.")
    _require(embeds.dtype == torch.bfloat16 and embeds.ndim == 3 and embeds.shape[0] == 1
             and all(size > 0 for size in embeds.shape), "Invalid condition embedding shape/dtype.")
    _require(mask.dtype in (torch.bool, torch.int64) and mask.shape == embeds.shape[:2], "Invalid attention mask shape/dtype.")
    _require(bool(torch.isfinite(embeds).all()) and bool(torch.count_nonzero(embeds)), "Nonfinite or zero embeddings.")
    _require(bool(((mask == 0) | (mask == 1)).all()) and bool(mask.any()), "Invalid binary attention mask.")
    _require(not embeds.requires_grad and not mask.requires_grad, "Condition tensor retained a gradient graph.")


def _source_checks(rgb: Tensor, latent: Tensor, config: Mapping[str, Any]) -> None:
    _require(rgb.dtype == torch.bfloat16 and tuple(rgb.shape) == (1, 3, 1024, 1024), "Invalid source RGB shape/dtype.")
    _require(latent.dtype == torch.float32 and list(latent.shape) == config["source"]["latent_shape"], "Invalid source latent.")
    _require(bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(latent).all()), "Nonfinite source tensor.")
    _require(canonical_tensor_sha256(rgb) == config["source"]["rgb_sha256"], "Source RGB hash drifted.")
    _require(canonical_tensor_sha256(latent) == config["source"]["latents_fp32_sha256"], "Source FP32 latent hash drifted.")


def _modules(pipe: Any) -> dict[str, nn.Module]:
    modules = {name: module for name, module in pipe.components.items() if isinstance(module, nn.Module)}
    _require(all(name in modules for name in ("unet", "vae", "text_encoder")), "Base pipeline modules missing.")
    return modules


def _frozen_audit(modules: Mapping[str, nn.Module]) -> dict[str, Any]:
    values = {f"{name}.{key}": parameter for name, module in modules.items() for key, parameter in module.named_parameters()}
    _require(bool(values), "Pipeline exposes no parameters for frozen audit.")
    trainable = [name for name, value in values.items() if value.requires_grad]
    gradients = [name for name, value in values.items() if value.grad is not None]
    _require(not trainable and not gradients, "Pipeline parameters must remain frozen without gradients.")
    return {"passed": True, "parameter_tensors": len(values), "parameter_numel": sum(p.numel() for p in values.values()),
            "trainable_names": trainable, "gradient_names": gradients,
            "modules": {name: type(module).__name__ for name, module in modules.items()}}


def _forbid_unet(counters: dict[str, int]) -> Any:
    def guard(_module: nn.Module, _inputs: Any) -> None:
        counters["unet_forward_calls"] += 1
        raise RuntimeError("U-Net forward is forbidden in the condition-only probe.")
    return guard


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": safety._sha256(path)}


def _save_condition(path: Path, condition: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    tensors = {"prompt_embeds": condition.prompt_embeds.detach().cpu(),
               "attention_mask": condition.attention_mask.detach().cpu()}
    phase1a._atomic_torch_save(path, {"schema": CONDITION_SCHEMA, "metadata": dict(metadata), **tensors})
    return {**_file_record(path), "tensor_sha256": {key: canonical_tensor_sha256(value) for key, value in tensors.items()},
            "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
            "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()}}


def _validate_artifacts(root: Path, config: Mapping[str, Any], parent_binding: Mapping[str, Any],
                        records: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = {"source": "source.pt", "condition": "condition.pt", "repeat": "condition-repeat.pt"}
    payloads = {}
    for name, filename in expected_files.items():
        path = root / filename
        _require({key: records[name].get(key) for key in ("path", "bytes", "sha256")} == _file_record(path),
                 f"Probe artifact byte binding drifted: {name}")
        payloads[name] = torch.load(path, map_location="cpu", weights_only=True)
    source = payloads["source"]
    _require(set(source) == {"schema", "source_rgb", "source_latents_fp32"} and source["schema"] == SOURCE_SCHEMA,
             "Source artifact fieldset/schema drifted.")
    _source_checks(source["source_rgb"], source["source_latents_fp32"], config)
    metadata = _metadata(config, parent_binding)
    for name in ("condition", "repeat"):
        value = payloads[name]
        _require(set(value) == {"schema", "metadata", "prompt_embeds", "attention_mask"}
                 and value["schema"] == CONDITION_SCHEMA, "Condition artifact fieldset/schema drifted.")
        _require(value["metadata"] == metadata, "Condition metadata disagrees with locked text/provenance.")
        _condition_tensor_checks(value["prompt_embeds"], value["attention_mask"])
        for key in ("prompt_embeds", "attention_mask"):
            _require(canonical_tensor_sha256(value[key]) == records[name]["tensor_sha256"][key], "Condition tensor hash drifted.")
            _require(list(value[key].shape) == records[name]["tensor_shapes"][key]
                     and str(value[key].dtype) == records[name]["tensor_dtypes"][key], "Condition tensor metadata drifted.")
            _require(torch.equal(value[key], payloads["condition"][key]), "Repeated condition tensor differs.")
    hashes = records["condition"]["tensor_sha256"]
    _require(hashes["prompt_embeds"] != config["condition"]["old_prompt_embeds_sha256"], "Identity intervention did not change embedding.")
    return {"passed": True, "tensor_sha256": dict(hashes),
            "attention_mask_equals_old": hashes["attention_mask"] == config["condition"]["old_attention_mask_sha256"]}


def _collect_condition_artifacts(root: Path, pipe: Any, source_rgb: Tensor, source_latents: Tensor,
                                 config: Mapping[str, Any], parent_binding: Mapping[str, Any],
                                 counters: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Model-independent evidence collection; production device checks live in _run_probe."""
    _require(counters == {**ZERO_COUNTERS, "condition_encoder_calls": 0}, "Probe counters must start at zero.")
    handle = pipe.unet.register_forward_pre_hook(_forbid_unet(counters))
    try:
        with torch.no_grad():
            _source_checks(source_rgb, source_latents.float(), config)
            phase1a._atomic_torch_save(root / "source.pt", {"schema": SOURCE_SCHEMA,
                "source_rgb": source_rgb.detach().cpu(), "source_latents_fp32": source_latents.detach().float().cpu()})
            records: dict[str, Any] = {"source": _file_record(root / "source.pt")}
            for name, filename in (("condition", "condition.pt"), ("repeat", "condition-repeat.pt")):
                counters["condition_encoder_calls"] += 1
                condition = _encode_condition(pipe, source_latents, IDENTITY_TEXT)
                records[name] = _save_condition(root / filename, condition, _metadata(config, parent_binding))
                _condition_tensor_checks(condition.prompt_embeds, condition.attention_mask)
                _require(condition.prompt_embeds.device == source_latents.device
                         and condition.attention_mask.device == source_latents.device, "Condition device drifted.")
                print(f"Condition encode {counters['condition_encoder_calls']}: {records[name]['tensor_sha256']}", flush=True)
        _require(counters == {**ZERO_COUNTERS, "condition_encoder_calls": 2}, "Forbidden probe execution counters.")
        return records, _validate_artifacts(root, config, parent_binding, records)
    finally:
        handle.remove()


def _run_probe(root: Path, config: Mapping[str, Any], validation: Mapping[str, Any],
               parent_binding: Mapping[str, Any], counters: dict[str, int]) -> dict[str, Any]:
    started = time.monotonic()
    determinism = phase1a.r8.configure_strict_cuda_determinism(0)
    phase1a._write_environment(root / "environment.txt")
    phase1a._atomic_json(root / "runtime.json", phase1a._runtime_versions())
    shutil.copyfile(CONFIG, root / "config.json")
    _require(safety._sha256(root / "config.json") == CONFIG_SHA256, "Copied probe config hash drifted.")
    model_dir = Path(config["dreamlite"]["path"])
    snapshot = phase1a.r5._verified_snapshot_payload(
        model_dir=model_dir, model_key="dreamlite_mobile",
        env_name="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256", required=True,
    )
    _require(snapshot["manifest_sha256"] == config["dreamlite"]["snapshot_manifest_sha256"], "DreamLite snapshot drifted.")
    phase1a._atomic_json(root / "model_snapshot_verification_start.json", snapshot)
    print("Snapshot fully verified; loading base DreamLite only.", flush=True)
    try:
        pipe = phase1a._load_pipeline(argparse.Namespace(dreamlite=model_dir), torch.device("cuda:0"), torch.bfloat16)
        modules = _modules(pipe)
        for module in modules.values():
            module.eval().requires_grad_(False)
        frozen_before = _frozen_audit(modules)
        # Install a guard before even source encoding, then retain an inner guard
        # in the independently testable evidence collector.
        source_guard = pipe.unet.register_forward_pre_hook(_forbid_unet(counters))
        with torch.no_grad():
            try:
                source_rgb = phase1a.blank_source_rgb(device=torch.device("cuda:0"), dtype=torch.bfloat16)
                source_latents = phase1a.encode_model_latent(pipe.vae, source_rgb)
                _require(source_latents.device == torch.device("cuda:0") and source_latents.dtype == torch.bfloat16,
                         "Source must be encoded on CUDA:0 in BF16, no CPU fallback.")
                records, artifact_audit = _collect_condition_artifacts(
                    root, pipe, source_rgb, source_latents, config, parent_binding, counters,
                )
            finally:
                source_guard.remove()
        frozen_after = _frozen_audit(modules)
        _require(frozen_after == frozen_before, "Frozen pipeline audit changed.")
        _require(_validate_parents(config) == parent_binding, "Parent evidence changed during probe.")
    finally:
        snapshot_after = phase1a.verify_snapshot_binding(snapshot)
        phase1a._atomic_json(root / "model_snapshot_verification_end.json", snapshot_after)
    gates = dict(config["technical_gate"])
    # These are evidence-backed gates, not trainability or scientific-success claims.
    _require(snapshot_after == snapshot, "DreamLite snapshot changed.")
    manifest = {"schema": MANIFEST_SCHEMA, "protocol": PROTOCOL, "config_sha256": CONFIG_SHA256,
        "probe_script_sha256": safety._sha256(Path(__file__)), "validation": dict(validation),
        "parent_binding": dict(parent_binding), "condition_metadata": _metadata(config, parent_binding),
        "artifacts": records, "model_snapshot_start": snapshot, "model_snapshot_end": snapshot_after,
        "strict_determinism": determinism, "frozen_before": frozen_before, "frozen_after": frozen_after,
        "information_boundary": dict(config["information_boundary"]), "counters": dict(counters)}
    phase1a._atomic_json(root / "manifest.json", manifest)
    result = {"schema": RESULT_SCHEMA, "protocol": PROTOCOL, "status": "completed",
        "engineering_gate": True, "technical_gate": gates, "counters": dict(counters),
        "artifact_audit": artifact_audit, "manifest_sha256": safety._sha256(root / "manifest.json"),
        "git_commit": validation["git_commit"], "config_sha256": CONFIG_SHA256,
        "information_boundary": dict(config["information_boundary"]), "formal_success": False, "full_success": False,
        "phase2_allowed": False, "bridge_result_evaluated": False, "elapsed_seconds": time.monotonic() - started,
        "next_step": config["next_step_if_pass"]}
    phase1a._atomic_json(root / "result.json", result)
    return result


def _write_inventory(root: Path) -> None:
    rows = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": safety._sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"]
    phase1a._atomic_json(root / "artifact_inventory.json", {"schema": INVENTORY_SCHEMA,
        "artifact_count": len(rows), "artifacts": rows})
    safety._validate_inventory(root, schema=INVENTORY_SCHEMA)


def _fail_delivery(root: Path, terminal: dict[str, Any], error: Exception, scope: str) -> None:
    detail = f"{type(error).__name__}: {error}"
    terminal.update(status="technical_failed", engineering_gate=False, exit_code=2)
    terminal[f"{scope}_error"] = detail
    result_path = root / "result.json"
    if result_path.is_file():
        result = safety._load(result_path)
        result.update(status="technical_failed", engineering_gate=False, delivery_audit_passed=False,
                      formal_success=False, full_success=False, phase2_allowed=False)
        result[f"{scope}_error"] = detail
        phase1a._atomic_json(result_path, result)
        terminal["result_sha256"] = safety._sha256(result_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = None
    claimed = False
    counters = {**ZERO_COUNTERS, "condition_encoder_calls": 0}
    terminal: dict[str, Any] = {"schema": TERMINAL_SCHEMA, "protocol": PROTOCOL,
        "status": "technical_failed", "engineering_gate": False, "formal_success": False, "full_success": False,
        "phase2_allowed": False, "bridge_result_evaluated": False, "git_commit": args.expected_commit,
        "config_sha256": CONFIG_SHA256, "exit_code": 2, "started_at_utc": safety._utc_now()}
    try:
        config = _load_config()
        validation = _validate_environment(args, config)
        parent_binding = _validate_parents(config)
        lock = _acquire_lock(args.output_root, args.expected_commit)
        _claim_root(args.output_root)
        claimed = True
        phase1a._atomic_json(args.output_root / "launch.json", {**terminal, "validation": validation,
            "parent_binding": parent_binding, "lock": lock, "argv": list(argv) if argv is not None else sys.argv[1:]})
        with (args.output_root / "stdout.log").open("x", encoding="utf-8") as stdout, (
            args.output_root / "stderr.log"
        ).open("x", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                _run_probe(args.output_root, config, validation, parent_binding, counters)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(status="technical_completed", engineering_gate=True, exit_code=0,
                        manifest_sha256=safety._sha256(args.output_root / "manifest.json"),
                        result_sha256=safety._sha256(args.output_root / "result.json"))
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
        print(terminal["error"], file=sys.stderr, flush=True)
    finally:
        if lock is not None:
            try:
                terminal["lock_release"] = safety._release_lock(lock)
            except Exception as error:
                terminal.update(status="technical_failed", engineering_gate=False, exit_code=2,
                                lock_release_error=f"{type(error).__name__}: {error}")
                if claimed:
                    _fail_delivery(args.output_root, terminal, error, "lock_release")
        if claimed:
            terminal.update(counters=dict(counters), completed_at_utc=safety._utc_now())
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            try:
                _write_inventory(args.output_root)
            except Exception as error:
                _fail_delivery(args.output_root, terminal, error, "inventory")
                phase1a._atomic_json(args.output_root / "terminal.json", terminal)
                try:
                    _write_inventory(args.output_root)
                except Exception as retry_error:
                    terminal["inventory_recovery_error"] = f"{type(retry_error).__name__}: {retry_error}"
                    phase1a._atomic_json(args.output_root / "terminal.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
