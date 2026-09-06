"""Shared read-only probe binding and explicitly separated identity conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.experiments import probe_r11_new_identity_condition as probe
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a
from vision_memory.dreamlite.conditioning import encode_latent_path_condition
from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_identity_condition_bridge as core


CONDITION_SCHEMA = "vision_memory.r11-new-identity-bridge-condition.v1"
PROBE_BINDING_SCHEMA = "vision_memory.r11-new-identity-bridge-probe-binding.v1"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def validate_probe_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read immutable original probe evidence; no model, teacher or GPU calls."""
    core.validate_bridge_config(config)
    binding = config["condition_probe_binding"]
    root = Path(binding["source_root"])
    for key, name in (("manifest", "manifest.json"), ("result", "result.json"),
                      ("terminal", "terminal.json"), ("inventory", "artifact_inventory.json")):
        _require(phase1a._sha256(root / name) == binding[key + "_sha256"], f"Identity probe {key} bytes drifted.")
    probe.safety._validate_inventory(root, schema=probe.INVENTORY_SCHEMA)
    manifest = probe.safety._load(root / "manifest.json")
    result = probe.safety._load(root / "result.json")
    terminal = probe.safety._load(root / "terminal.json")
    probe_config = probe._load_config()
    _require(probe.safety._load(root / "config.json") == probe_config, "Identity probe config drifted.")
    _require(binding["config_file_sha256"] == probe.CONFIG_SHA256, "Wrong identity probe config lock.")
    _require(manifest.get("schema") == probe.MANIFEST_SCHEMA and result.get("schema") == probe.RESULT_SCHEMA
             and terminal.get("schema") == probe.TERMINAL_SCHEMA, "Identity probe schemas drifted.")
    counters = {**probe.ZERO_COUNTERS, "condition_encoder_calls": 2}
    for record in (manifest, result, terminal):
        _require(record.get("protocol") == probe.PROTOCOL and record.get("counters") == counters,
                 "Identity probe protocol/counters drifted.")
    _require(manifest["validation"]["git_commit"] == result["git_commit"] == terminal["git_commit"]
             == binding["training_git_commit"], "Identity probe commit drifted.")
    _require(result["manifest_sha256"] == terminal["manifest_sha256"] == binding["manifest_sha256"]
             and terminal["result_sha256"] == binding["result_sha256"], "Identity probe result hash chain drifted.")
    _require(terminal["status"] == "technical_completed" and terminal["engineering_gate"] is True
             and terminal["exit_code"] == 0 and terminal["lock_release"]["released"] is True
             and result["engineering_gate"] is True and result["bridge_result_evaluated"] is False,
             "Identity probe was not technically completed.")
    _require(result["formal_success"] is False and terminal["formal_success"] is False
             and result["phase2_allowed"] is False and terminal["phase2_allowed"] is False,
             "Identity probe carries an invalid scientific-success claim.")
    parent = probe._validate_parents(probe_config)
    _require(parent == manifest["parent_binding"], "Identity probe parent evidence changed.")
    audit = probe._validate_artifacts(root, probe_config, parent, manifest["artifacts"])
    _require(audit == result["artifact_audit"], "Identity probe raw tensor audit changed.")
    for name, key in (("source", "source_artifact"), ("condition", "condition_artifact"), ("repeat", "repeat_artifact")):
        record = manifest["artifacts"][name]
        _require({k: record[k] for k in ("path", "sha256", "bytes")} == binding[key], "Identity probe raw file lock drifted.")
    for name in ("prompt_embeds", "attention_mask"):
        expected = binding[name]
        record = manifest["artifacts"]["condition"]
        _require(expected == {"sha256": record["tensor_sha256"][name], "shape": record["tensor_shapes"][name],
                             "dtype": record["tensor_dtypes"][name]}, "Identity probe tensor lock drifted.")
    _require(parent["target_segment_canonical_sha256"] == binding["full_target_canonical_sha256"],
             "Identity probe original target changed.")
    return {"schema": PROBE_BINDING_SCHEMA, "passed": True, "source_root": str(root),
            "probe_commit": binding["training_git_commit"], "manifest_sha256": binding["manifest_sha256"],
            "result_sha256": binding["result_sha256"], "terminal_sha256": binding["terminal_sha256"],
            "inventory_sha256": binding["inventory_sha256"], "tensor_sha256": dict(audit["tensor_sha256"])}


def expected_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    changed = config["single_changed_solver_factor"]
    return {
        "original_event_text": changed["parent_value"],
        "original_event_text_sha256": config["condition_probe_binding"]["original_event_text_sha256"],
        "original_event_is_provenance_only": True,
        "actual_conditioning_text": changed["new_value"],
        "actual_conditioning_text_sha256": changed["actual_conditioning_text_sha256"],
        "full_prompt": changed["full_prompt"], "full_prompt_sha256": changed["full_prompt_sha256"],
        "mode": "edit", "conditioner_input_names": ["fixed_source_latents", "fixed_identity_text"],
        "probe_manifest_sha256": core.BRIDGE_PROBE_MANIFEST_SHA256,
        "source_latents_fp32_sha256": core.BRIDGE_SOURCE_LATENTS_SHA256,
    }


def _validate_tensor_pair(embeds: torch.Tensor, mask: torch.Tensor) -> None:
    probe._condition_tensor_checks(embeds, mask)
    _require(tuple(embeds.shape) == (1, 293, 2048) and tuple(mask.shape) == (1, 293), "Identity condition shapes drifted.")
    _require(canonical_tensor_sha256(embeds) == core.BRIDGE_CONDITION_EMBEDS_SHA256
             and canonical_tensor_sha256(mask) == core.BRIDGE_CONDITION_MASK_SHA256,
             "Reencoded condition differs from locked first probe; no fallback or prompt selection.")


def save_condition_pair(output_dir: Path, first: Any, second: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Store BOTH raw encodings before rejecting numerical or repeat mismatches."""
    metadata = expected_metadata(config)
    records = []
    for condition, filename in ((first, "official_full_condition.pt"), (second, "identity_condition_repeat.pt")):
        path = output_dir / "condition" / filename
        tensors = {"prompt_embeds": condition.prompt_embeds.detach().cpu(),
                   "attention_mask": condition.attention_mask.detach().cpu()}
        payload = {"schema": CONDITION_SCHEMA, **tensors, "metadata": metadata,
                   "event_text_sha256": metadata["original_event_text_sha256"],
                   "tensor_sha256": {k: canonical_tensor_sha256(v) for k, v in tensors.items()}}
        phase1a._atomic_torch_save(path, payload)
        records.append({"path": str(path), "sha256": phase1a._sha256(path), "bytes": path.stat().st_size,
                        "tensor_sha256": payload["tensor_sha256"], "event_text_sha256": payload["event_text_sha256"],
                        "metadata": dict(metadata)})
    _validate_tensor_pair(first.prompt_embeds, first.attention_mask)
    _validate_tensor_pair(second.prompt_embeds, second.attention_mask)
    _require(torch.equal(first.prompt_embeds, second.prompt_embeds)
             and torch.equal(first.attention_mask, second.attention_mask), "Identity condition repeat mismatch.")
    record = {**records[0], "repeat": records[1], "recompute_matches": True}
    _require(verify_condition_record(record, config=config), "Saved identity condition verification failed.")
    return record


def verify_condition_record(record: Mapping[str, Any], *, config: Mapping[str, Any]) -> bool:
    """Recompute both tensors and fixed metadata, rejecting a re-signed false record."""
    metadata = expected_metadata(config)
    loaded = []
    try:
        first_path, repeat_path = Path(record["path"]), Path(record["repeat"]["path"])
        _require(first_path.name == "official_full_condition.pt" and repeat_path.name == "identity_condition_repeat.pt"
                 and first_path.parent.resolve() == repeat_path.parent.resolve(),
                 "Identity condition must retain two separately named sibling artifacts.")
        for item in (record, record["repeat"]):
            path = Path(item["path"])
            _require(phase1a._sha256(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                     "Identity condition bytes drifted.")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            _require(payload.get("schema") == CONDITION_SCHEMA, "Identity condition schema drifted.")
            _require(payload.get("metadata") == item.get("metadata") == metadata, "Identity actual/original text binding drifted.")
            _require(payload.get("event_text_sha256") == item.get("event_text_sha256")
                     == metadata["original_event_text_sha256"], "Identity original event provenance drifted.")
            _validate_tensor_pair(payload["prompt_embeds"], payload["attention_mask"])
            _require(payload["tensor_sha256"] == item["tensor_sha256"] == {
                "prompt_embeds": core.BRIDGE_CONDITION_EMBEDS_SHA256,
                "attention_mask": core.BRIDGE_CONDITION_MASK_SHA256}, "Identity condition hashes drifted.")
            loaded.append(payload)
        _require(torch.equal(loaded[0]["prompt_embeds"], loaded[1]["prompt_embeds"])
                 and torch.equal(loaded[0]["attention_mask"], loaded[1]["attention_mask"]), "Identity repeat tensor mismatch.")
        return record.get("recompute_matches") is True
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        return False


def load_runtime(args: Any, config: Mapping[str, Any]):
    """Retain the original target; only actual native conditioner text changes."""
    _require(torch.cuda.is_available(), "Identity full-chain runtime requires CUDA.")
    probe_binding = validate_probe_binding(config)
    updater_device, reader_device = torch.device(args.dreamlite_device), torch.device(args.reader_device)
    updater_dtype, reader_dtype = phase1a.r5.compute_dtype(updater_device), phase1a.r5.compute_dtype(reader_device)
    data = phase1a.r5._load_data(args, optimizer_steps=0)
    selected = phase1a.select_f1_targets(data.train_pools)
    selected_sha = phase1a.r5.canonical_sha256([segment.to_dict() for segment in selected])
    _require(tuple(s.segment_id for s in selected) == phase1a.R11_NEW_TARGET_IDS
             and selected_sha == phase1a.R11_NEW_TARGETS_PAYLOAD_SHA256, "Identity fixed eight-target selection drifted.")
    target = selected[args.target_index]
    _require(target.family == "F1" and len(target.events) == 1
             and target.segment_id == core.BRIDGE_TARGET_SEGMENT_ID, "Identity original target identity drifted.")
    _require(core.canonical_json_sha256(target.to_dict()) == config["condition_probe_binding"]["full_target_canonical_sha256"],
             "Identity original answer-bearing target content drifted.")
    _require(target.events[0].event_text == config["single_changed_solver_factor"]["parent_value"],
             "Identity original event changed.")
    pipe = phase1a._load_pipeline(args, updater_device, updater_dtype)
    processor, reader = phase1a._load_reader(args, reader_device, reader_dtype)
    source_rgb = phase1a.blank_source_rgb(device=updater_device, dtype=updater_dtype)
    with torch.no_grad():
        source_latents = phase1a.encode_model_latent(pipe.vae, source_rgb)
        _require(canonical_tensor_sha256(source_rgb) == core.BRIDGE_UNCHANGED_PARITY_BINDINGS["blank_source_rgb_sha256"]
                 and canonical_tensor_sha256(source_latents.float()) == core.BRIDGE_SOURCE_LATENTS_SHA256,
                 "Identity source hash drifted before condition encoding.")
        first = encode_latent_path_condition(pipe, source_latents, core.BRIDGE_CONDITION_TEXT)
        second = encode_latent_path_condition(pipe, source_latents, core.BRIDGE_CONDITION_TEXT)
    condition_record = save_condition_pair(args.output_dir, first, second, config)
    oracle = phase1a.FrozenDreamLiteOracle(
        unet=pipe.unet, scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=pipe.text_encoder,
        source_latents=source_latents, prompt_embeds=first.prompt_embeds,
        prompt_attention_mask=first.attention_mask, initial_x_t=source_latents.detach().float().clone(),
        compute_dtype=updater_dtype, checkpoint_unet=True, vae_scale_factor=int(pipe.vae_scale_factor),
    )
    context = {"data": data, "selected": selected, "selected_sha": selected_sha, "target": target, "pipe": pipe,
               "source_rgb": source_rgb, "source_latents": source_latents, "condition_record": condition_record,
               "identity_condition_probe_binding": probe_binding,
               "updater_device": updater_device, "reader_device": reader_device}
    return processor, pipe, reader, oracle, source_latents, context


def information_boundary(target: Any) -> dict[str, Any]:
    base = phase1a._writer_information_boundary(target)
    return {**base, **core.bridge_information_boundary(),
            "conditioner_input_names": ["fixed_source_latents", "fixed_identity_text"],
            "sampler_input_names": ["source_latents", "identity_prompt_embeds", "identity_prompt_attention_mask",
                                    "x_T_fp32", "fixed_scheduler"],
            "oracle_initialization_key_names": ["source_latents"],
            "original_event_retained_for_provenance_only": True}


def single_changed_factor(config: Mapping[str, Any]) -> dict[str, Any]:
    changed = config["single_changed_solver_factor"]
    return {"factor": "actual_conditioning_text", "from": changed["parent_value"], "to": changed["new_value"],
            "actual_conditioning_text_sha256": changed["actual_conditioning_text_sha256"],
            "full_prompt_sha256": changed["full_prompt_sha256"],
            "initialization_teacher_assisted": False, "conditioning_teacher_assisted": False,
            "optimization_teacher_supervised": True, "answer_independent_writer_usable": False}


def validate_manifest_identity(manifest: Mapping[str, Any], *, config: Mapping[str, Any]) -> bool:
    binding = validate_probe_binding(config)
    return bool(
        manifest.get("identity_condition_probe_binding") == binding
        and manifest.get("single_changed_factor") == single_changed_factor(config)
        and core.validate_bridge_information_boundary(manifest.get("information_boundary", {}))
        and verify_condition_record(manifest.get("condition_artifact", {}), config=config)
    )
