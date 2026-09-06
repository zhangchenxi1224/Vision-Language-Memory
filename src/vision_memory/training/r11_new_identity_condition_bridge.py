"""Immutable single-text intervention contracts; no model execution on import."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from vision_memory.training import r11_new_source_init_bridge as source


BRIDGE_PROTOCOL = "R11-New-Identity-Conditioning-Bridge-Source-Init-Target01"
BRIDGE_CONFIG_SCHEMA = "vision_memory.r11-new-identity-conditioning-bridge-config.v1"
BRIDGE_CONFIG_FILE_SHA256 = "e4d375f10796d70f64ed91e201aebdbaa6a3939d5c7e37fd90f7034525fa60c8"
BRIDGE_CONFIG_CANONICAL_SHA256 = "9ecd610e7c001d001af38cd04d4950f8470d3d8cfef7a6315cbd8ef40fd76f17"
BRIDGE_PARENT_ENDPOINT_MSE = 0.11177215725183487
BRIDGE_PARENT_ENDPOINT_READER_CE = 25.765637596946004
BRIDGE_CONDITION_TEXT = "no changes"
BRIDGE_CONDITION_TEXT_SHA256 = "44f9161c3a252925f55022d60f68918abf8fdc1a9a3fadbf5ab766549f6b3461"
BRIDGE_CONDITION_PROMPT_SHA256 = "4620b977a7aff51c9ee9a6e2901555b39ee4a3e4bd705d1402818acf330d22e6"
BRIDGE_CONDITION_EMBEDS_SHA256 = "dbb23b166711849739432ba815002e889253da78e099e59c7adbd351a8fb9f4d"
BRIDGE_CONDITION_MASK_SHA256 = "c3325ac723d7cc08cba37d46e1710b5067019e47236ef55c41c66cba27c519f1"
BRIDGE_PROBE_COMMIT = "55c355f769c21ceb9b86e5090cdafa4090751d21"
BRIDGE_PROBE_DELIVERY_COMMIT = "66d20121ad2277429b8f61ce7666a636c345b70e"
BRIDGE_PROBE_MANIFEST_SHA256 = "6dc0652d6d1750c51341a55849757bcc3fa86b201a451eb020803e0912e0aaa2"

# Share only unchanged numerical and data contracts. In particular, do NOT
# re-export source.bridge_initialization_hypothesis_audit: its globals bind the
# Gaussian parent, which is not this experiment's single-factor comparator.
BRIDGE_BASE_LEARNING_RATE = source.BRIDGE_BASE_LEARNING_RATE
BRIDGE_CHECKPOINT_STEPS = source.BRIDGE_CHECKPOINT_STEPS
BRIDGE_L2_RATIO_MAX = source.BRIDGE_L2_RATIO_MAX
BRIDGE_LR_COSINE_DENOMINATOR = source.BRIDGE_LR_COSINE_DENOMINATOR
BRIDGE_LR_INTERVENTION_FIRST_UPDATE = source.BRIDGE_LR_INTERVENTION_FIRST_UPDATE
BRIDGE_MODEL_SNAPSHOT_BINDINGS = dict(source.BRIDGE_MODEL_SNAPSHOT_BINDINGS)
BRIDGE_MSE_RATIO_MAX = source.BRIDGE_MSE_RATIO_MAX
BRIDGE_OPTIMIZER_STEPS = source.BRIDGE_OPTIMIZER_STEPS
BRIDGE_PARENT_TARGET_MANIFEST_SHA256 = source.BRIDGE_PARENT_TARGET_MANIFEST_SHA256
BRIDGE_PHASE1A_SOURCE_ROOT = source.BRIDGE_PHASE1A_SOURCE_ROOT
BRIDGE_PRIMARY_ENDPOINT = source.BRIDGE_PRIMARY_ENDPOINT
BRIDGE_SOURCE_LATENTS_SHA256 = source.BRIDGE_SOURCE_LATENTS_SHA256
BRIDGE_TARGET_INDEX = source.BRIDGE_TARGET_INDEX
BRIDGE_TARGET_SEGMENT_ID = source.BRIDGE_TARGET_SEGMENT_ID
BRIDGE_TEACHER_FILE_SHA256 = source.BRIDGE_TEACHER_FILE_SHA256
BRIDGE_TEACHER_NRMSE_MAX = source.BRIDGE_TEACHER_NRMSE_MAX
BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX = source.BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX
BRIDGE_TEACHER_STD = source.BRIDGE_TEACHER_STD
BRIDGE_TEACHER_TENSOR_SHA256 = source.BRIDGE_TEACHER_TENSOR_SHA256
BRIDGE_INITIALIZATION_SCHEMA = source.BRIDGE_INITIALIZATION_SCHEMA
BRIDGE_INITIALIZATION_FORMULA = source.BRIDGE_INITIALIZATION_FORMULA
BRIDGE_INITIALIZATION_FILENAME = source.BRIDGE_INITIALIZATION_FILENAME
BRIDGE_UNCHANGED_PARITY_BINDINGS = {
    **source.BRIDGE_UNCHANGED_PARITY_BINDINGS,
    "condition_prompt_embeds_sha256": BRIDGE_CONDITION_EMBEDS_SHA256,
    "condition_attention_mask_sha256": BRIDGE_CONDITION_MASK_SHA256,
}
BRIDGE_REQUIRED_TECHNICAL_FIELDS = (
    *source.BRIDGE_REQUIRED_TECHNICAL_FIELDS,
    "identity_condition_probe_binding_valid",
    "actual_conditioning_text_binding_valid",
)
canonical_json_sha256 = source.canonical_json_sha256
bridge_distance_gate = source.bridge_distance_gate
bridge_distance_statistics = source.bridge_distance_statistics
bridge_optimizer_learning_rate = source.bridge_optimizer_learning_rate
endpoint_reader_transfer_gate = source.endpoint_reader_transfer_gate
reader_checkpoint_statistics = source.reader_checkpoint_statistics
teacher_replay_gate = source.teacher_replay_gate


def validate_bridge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != BRIDGE_CONFIG_SCHEMA or config.get("protocol") != BRIDGE_PROTOCOL:
        raise ValueError("Identity bridge protocol/schema drifted.")
    if canonical_json_sha256(config) != BRIDGE_CONFIG_CANONICAL_SHA256:
        raise ValueError("Identity bridge canonical config drifted.")
    expected = {
        "target_selection": {"target_index": BRIDGE_TARGET_INDEX, "target_segment_id": BRIDGE_TARGET_SEGMENT_ID},
        "parent_bridge": {
            "protocol": source.BRIDGE_PROTOCOL,
            "endpoint_mse": BRIDGE_PARENT_ENDPOINT_MSE,
            "endpoint_reader_mean_ce": BRIDGE_PARENT_ENDPOINT_READER_CE,
            "config_sha256": source.BRIDGE_CONFIG_FILE_SHA256,
        },
        "single_changed_solver_factor": {
            "factor": "actual_conditioning_text", "new_value": BRIDGE_CONDITION_TEXT,
            "actual_conditioning_text_sha256": BRIDGE_CONDITION_TEXT_SHA256,
            "full_prompt_sha256": BRIDGE_CONDITION_PROMPT_SHA256,
            "official_template_unchanged": True,
            "original_event_preserved_as_provenance": True,
            "initialization_teacher_assisted": False, "conditioning_teacher_assisted": False,
            "optimization_teacher_supervised": True, "answer_independent_writer_usable": False,
        },
        "condition_probe_binding": {
            "training_git_commit": BRIDGE_PROBE_COMMIT,
            "delivery_git_commit": BRIDGE_PROBE_DELIVERY_COMMIT,
            "manifest_sha256": BRIDGE_PROBE_MANIFEST_SHA256,
            "prompt_embeds": {"sha256": BRIDGE_CONDITION_EMBEDS_SHA256, "shape": [1, 293, 2048], "dtype": "torch.bfloat16"},
            "attention_mask": {"sha256": BRIDGE_CONDITION_MASK_SHA256, "shape": [1, 293], "dtype": "torch.int64"},
        },
        "initialization_binding": {
            "artifact_schema": BRIDGE_INITIALIZATION_SCHEMA,
            "source_latents_fp32_sha256": BRIDGE_SOURCE_LATENTS_SHA256,
            "initial_x_t_equals_source": True,
            "no_cpu_fallback_for_cuda_recomputation": True,
        },
        "unchanged_contract": {
            "only_trainable": "x_T_fp32", "optimizer_steps": BRIDGE_OPTIMIZER_STEPS,
            "checkpoint_steps": list(BRIDGE_CHECKPOINT_STEPS), "primary_endpoint": BRIDGE_PRIMARY_ENDPOINT,
            "base_learning_rate": BRIDGE_BASE_LEARNING_RATE, "gradient_clipping": None,
        },
        "primary_bridge_gate": {
            "mse_ratio_to_m0_lte": BRIDGE_MSE_RATIO_MAX, "l2_distance_ratio_to_m0_lte": BRIDGE_L2_RATIO_MAX,
            "teacher_normalized_rmse_lte": BRIDGE_TEACHER_NRMSE_MAX, "endpoint_reverse_cyclic_accuracy_eq": 1,
        },
    }
    for section, bindings in expected.items():
        for key, value in bindings.items():
            observed = config[section].get(key)
            if type(observed) is not type(value) or observed != value:
                raise ValueError(f"Identity bridge implementation constant drifted: {section}.{key}")
    if hashlib.sha256(BRIDGE_CONDITION_TEXT.encode("utf-8")).hexdigest() != BRIDGE_CONDITION_TEXT_SHA256:
        raise ValueError("Identity bridge actual text byte definition drifted.")
    return dict(config)


def load_bridge_config(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BRIDGE_CONFIG_FILE_SHA256:
        raise ValueError("Identity bridge config file bytes drifted.")
    return validate_bridge_config(json.loads(payload))


def bridge_information_boundary() -> dict[str, bool]:
    return {
        **source.bridge_information_boundary(),
        "conditioning_teacher_assisted": False,
        "original_event_used_as_condition": False,
        "fixed_identity_text_used_as_condition": True,
        "event_to_state_learning_evaluated": False,
    }


def validate_bridge_information_boundary(boundary: Mapping[str, Any]) -> bool:
    return all(boundary.get(name) is value for name, value in bridge_information_boundary().items())


def bridge_technical_gate(audit: Mapping[str, Any]) -> bool:
    return all(audit.get(name) is True for name in BRIDGE_REQUIRED_TECHNICAL_FIELDS)


def bridge_decision(*, distance_pass: bool, reader_transfer_pass: bool) -> str:
    if distance_pass and reader_transfer_pass:
        return "distance_pass_reader_pass_restore_event_revalidate_original_qa_solver"
    if distance_pass:
        return "distance_pass_reader_fail_test_teacher_neighborhood"
    if reader_transfer_pass:
        return "distance_fail_reader_pass_restore_event_revalidate_original_qa_solver"
    return "distance_fail_reader_fail_preregister_one_discriminating_diagnostic"


def bridge_conditioning_hypothesis_audit(
    *, endpoint_mse: float, endpoint_reader_mean_ce: float,
    technical_gate: bool, teacher_replay_gate: bool, distance_pass: bool, reader_transfer_pass: bool,
) -> dict[str, bool]:
    values = (endpoint_mse, endpoint_reader_mean_ce)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Identity conditioning audit received an invalid metric.")
    flags = (technical_gate, teacher_replay_gate, distance_pass, reader_transfer_pass)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("Identity conditioning audit requires exact boolean gates.")
    eligible = technical_gate and teacher_replay_gate and not distance_pass and not reader_transfer_pass
    mse_improves = endpoint_mse < BRIDGE_PARENT_ENDPOINT_MSE
    ce_improves = endpoint_reader_mean_ce < BRIDGE_PARENT_ENDPOINT_READER_CE
    return {
        "eligible": eligible, "absolute_endpoint_mse_improves_parent": mse_improves,
        "endpoint_reader_ce_improves_parent": ce_improves,
        "passed": eligible and mse_improves and ce_improves,
    }


def bridge_conditioning_hypothesis_decision(
    *, distance_pass: bool, reader_transfer_pass: bool, audit: Mapping[str, Any],
) -> str:
    if distance_pass or reader_transfer_pass:
        return "primary_branch_" + bridge_decision(distance_pass=distance_pass, reader_transfer_pass=reader_transfer_pass)
    if audit.get("eligible") is not True:
        return "technical_or_teacher_invalid_conditioning_audit_unevaluated"
    if audit.get("passed") is True:
        return "distance_fail_reader_fail_secondary_condition_improves"
    return "distance_fail_reader_fail_secondary_condition_not_improve"
