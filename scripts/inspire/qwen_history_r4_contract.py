from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_snapshot_manifest import verify_snapshot_binding, verify_snapshot_manifest  # noqa: E402
from r3_dag_contract import (  # noqa: E402
    COMMIT_PATTERN,
    SHA256_PATTERN,
    load_json_object,
    require_file_sha,
    require_json_values,
    sha256_file,
    verify_sha_sidecar,
)
from vision_memory.data import REVERSE_CYCLIC4, permutation_family_sha256  # noqa: E402
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_RAW_HISTORY,
    R4_HISTORY_TASK_INSTRUCTION,
    representation_contract_sha256,
)


AMENDMENT_SCHEMA = "vision_memory.r4-qwen-history-comparison-amendment.v1"
PLAN_PROTOCOL = "r4-inspire-qwen-history-comparison-dag.v1"
STAGE_SPEC_PROTOCOL = "r4-inspire-qwen-history-comparison-stage.v1"
STAGE_EVIDENCE_PROTOCOL = "r4-inspire-qwen-history-comparison-evidence.v1"
WRAPPER_FAILURE_PROTOCOL = "r4-inspire-qwen-history-comparison-wrapper-failure.v1"
PREDICTION_SCHEMA = "vision_memory.qwen_r4_history_predictions.v1"
PREDICTION_REPORT_SCHEMA = "vision_memory.qwen_r4_history_report.v1"
SCORE_SCHEMA = "vlm.qwen-history-r4-score.v1"
COMPARISON_SCHEMA = "vlm.qwen-history-r4-comparison.v1"

STAGES = ("BH0", "BH1", "BH2", "BH3")
DEVICES = {"A": "cuda:0", "B": "cuda:1"}
ARM_ORDER = ("raw", "tagged", "last_effective")
ARM_METHODS = {
    "raw": "qwen_r4_raw_history",
    "tagged": "qwen_r4_operation_tagged_history",
    "last_effective": "qwen_r4_last_effective_event",
}
ALLOWED_CONDITIONS = {"standard", "reset", "shuffle", "state_swap"}
SCORE_SUITES = {"smoke", "transition32", "formal"}
COMPARISON_SUITES = {"bh1", "formal"}

EXPECTED_READER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_INSTRUCTION_SHA256 = "95c5223e70075b8e57c04f7b0bd016b98e51c20be4cdc0584b48a9b433228e74"
EXPECTED_REVERSE_CYCLIC4_SHA256 = "4cd725a443d8661dccccbff2d714876aee317c654ef2aa1bcb79aee307d64bbd"
EXPECTED_REPRESENTATION_CONTRACT_SHA256 = {
    "raw": "096f874e4ee9cce819fe3e193d96bef40ee4b2f11310dc0313ee9449f770d381",
    "tagged": "b75b0354d546c96f065b21b814cdcf8c3e522dd2eb7cde9ac1ee4ce3d70964be",
    "last_effective": "aa28a9d2ce7c1260cbd1ab0df30058bbd0c99ad2eb94735d8ecdd74818f15a8f",
}
EXPECTED_VISUAL_INPUT = {
    "input_mode": "blank_image",
    "shape_chw": [3, 1024, 1024],
    "dtype": "float32",
    "rgb_value": 0.5,
    "constant_visual_input_bytes": 3 * 1024 * 1024 * 4,
    "reader_resize_contract": "r3-qwen-reader-1024-to-256-bicubic-antialias-cpu-adjoint.v1",
}
EXPECTED_THRESHOLDS = {
    "bh0_last_effective": {
        "overall": "15/16",
        "per_position": "3/4",
        "per_event_kind": "3/4",
        "rotation_consistent_states": "4/4",
        "clean_noop_memory_prompt_prediction_nll_exact": "4/4",
    },
    "bh1_raw_and_tagged_report_only": {
        "overall": "116/128",
        "per_position": "28/32",
        "per_event_kind": "28/32",
        "mixed": "58/64",
        "per_kind_form_cell": "14/16",
        "rotation_consistent_states": "30/32",
        "state_swap_donor": "28/32",
        "reset_drop": "32/128",
        "shuffle_drop": "32/128",
        "clean_noop_prediction_agreement": "30/32",
    },
    "bh1_last_effective_blocking": {
        "overall": "122/128",
        "per_position": "30/32",
        "per_event_kind": "30/32",
        "mixed": "61/64",
        "per_kind_form_cell": "15/16",
        "rotation_consistent_states": "31/32",
        "state_swap_donor": "30/32",
        "reset_drop": "32/128",
        "shuffle_drop": "32/128",
        "clean_noop_memory_prompt_prediction_nll_exact": "32/32",
    },
    "bh2_last_effective_dev_standard_accuracy": 0.95,
}
EXPECTED_STAGE_GATE_POLICY = {
    "BH0": True,
    "BH1": True,
    "BH2": True,
    "BH3": False,
}
EXPECTED_PRE_BH2_TEST_POLICY = {
    "scope": ["formal_test_id", "formal_test_ood"],
    "allowed_before_bh2_unlock": ["sha256_byte_binding"],
    "json_semantic_parse_before_bh2_unlock": False,
    "scoring_before_bh2_unlock": False,
    "metric_access_before_bh2_unlock": False,
    "evaluation_before_bh2_unlock": False,
    "unlock_condition": "BH2_last_effective_dev_gate_passed",
}
EXPECTED_FORBIDDEN = [
    "DreamLite training",
    "PrefEval",
    "text-only R4 arm",
    "teacher or ledger sidecar",
    "pre-BH2 test JSON semantic parsing, scoring, metric access, or evaluation execution",
    "post-hoc prompt, threshold, data, or permutation changes",
]

SCIENTIFIC_SCRIPTS = {
    "qwen_history_r4.py",
    "score_qwen_history_r4.py",
    "compare_qwen_history_r4.py",
}
FORBIDDEN_COMMAND_FRAGMENTS = (
    "dreamlite",
    "scripts/train/",
    "scripts\\train\\",
    "teacher",
    "oracle",
    "ledger",
    "target_selected",
    "optimizer",
    "backward(",
)


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _validate_arm_contract(arms: Mapping[str, Any]) -> None:
    if tuple(arms) != ARM_ORDER:
        raise ValueError(f"R4 arms must be ordered exactly as {list(ARM_ORDER)}")
    representations = {
        "raw": "raw_chronological_event_history",
        "tagged": "operation_tagged_event_history",
        "last_effective": "last_effective_event",
    }
    implementation_methods = {
        "raw": QWEN_R4_RAW_HISTORY,
        "tagged": QWEN_R4_OPERATION_TAGGED_HISTORY,
        "last_effective": QWEN_R4_LAST_EFFECTIVE_EVENT,
    }
    for arm in ARM_ORDER:
        entry = arms[arm]
        if not isinstance(entry, Mapping):
            raise ValueError(f"R4 arm {arm} must be an object")
        implementation_sha = representation_contract_sha256(implementation_methods[arm])
        if implementation_sha != EXPECTED_REPRESENTATION_CONTRACT_SHA256[arm]:
            raise ValueError(f"R4 arm {arm} implementation representation contract SHA256 drifted")
        expected_entry = {
            "method": ARM_METHODS[arm],
            "history_representation": representations[arm],
            "representation_contract_sha256": EXPECTED_REPRESENTATION_CONTRACT_SHA256[arm],
            "training": False,
            "teacher_access": False,
            "oracle_ledger_access": False,
            "target_access": False,
        }
        require_json_values(
            entry,
            expected_entry,
            f"R4 arm {arm}",
        )
        expected_router_metadata = arm in {"tagged", "last_effective"}
        if entry.get("oracle_router_metadata") is not expected_router_metadata:
            raise ValueError(f"R4 arm {arm} has the wrong oracle_router_metadata declaration")
        expected_role = "data_readability_gate" if arm == "last_effective" else "descriptive_nonblocking"
        if entry.get("performance_role") != expected_role:
            raise ValueError(f"R4 arm {arm} has the wrong performance_role")
        expected_keys = set(expected_entry) | {"oracle_router_metadata", "performance_role"}
        if set(entry) != expected_keys:
            raise ValueError(f"R4 arm {arm} contains missing or unexpected contract fields")


def _validate_inventory(amendment: Mapping[str, Any]) -> None:
    inventory = amendment.get("expected_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("R4 amendment expected_inventory must be a non-empty object")
    data = amendment.get("data")
    if not isinstance(data, Mapping) or not data:
        raise ValueError("R4 amendment data must be a non-empty object")
    if any(SHA256_PATTERN.fullmatch(str(value)) is None for value in data.values()):
        raise ValueError("Every R4 data lock must be a lowercase SHA256 digest")

    for name, entry in inventory.items():
        if not isinstance(name, str) or not name or not isinstance(entry, Mapping):
            raise ValueError("R4 inventory names must map to objects")
        data_key = entry.get("data_key")
        if not isinstance(data_key, str) or data_key not in data:
            raise ValueError(f"R4 inventory {name} references an unknown data_key")
        query_states = _require_positive_int(entry.get("query_states"), label=f"{name}.query_states")
        conditions = entry.get("conditions")
        if (
            not isinstance(conditions, list)
            or not conditions
            or len(set(conditions)) != len(conditions)
            or any(value not in ALLOWED_CONDITIONS for value in conditions)
        ):
            raise ValueError(f"R4 inventory {name} has invalid conditions")
        probe_role = entry.get("probe_role")
        if probe_role not in {"all", "delayed"}:
            raise ValueError(f"R4 inventory {name} has an invalid probe_role")
        limit = entry.get("limit")
        if limit is not None:
            _require_positive_int(limit, label=f"{name}.limit")
        expected_per_condition = query_states * 4
        expected_total = expected_per_condition * len(conditions)
        require_json_values(
            entry,
            {
                "records_per_condition": expected_per_condition,
                "prediction_records_per_arm": expected_total,
            },
            f"R4 inventory {name}",
        )

    stages = amendment.get("stages")
    if not isinstance(stages, Mapping) or tuple(stages) != STAGES:
        raise ValueError(f"R4 stages must be ordered exactly as {list(STAGES)}")
    seen_datasets: set[str] = set()
    for stage in STAGES:
        entry = stages[stage]
        if not isinstance(entry, Mapping):
            raise ValueError(f"R4 stage {stage} must be an object")
        datasets = entry.get("datasets")
        if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets):
            raise ValueError(f"R4 stage {stage} must list unique datasets")
        for dataset in datasets:
            if dataset not in inventory:
                raise ValueError(f"R4 stage {stage} references unknown inventory {dataset!r}")
            if dataset in seen_datasets:
                raise ValueError(f"R4 inventory {dataset!r} appears in more than one stage")
            seen_datasets.add(dataset)
        suite = entry.get("score_suite")
        if suite not in SCORE_SUITES:
            raise ValueError(f"R4 stage {stage} has unsupported score_suite")
        comparison_suite = entry.get("comparison_suite")
        if comparison_suite is not None and comparison_suite not in COMPARISON_SUITES:
            raise ValueError(f"R4 stage {stage} has unsupported comparison_suite")
        expected_gate = EXPECTED_STAGE_GATE_POLICY[stage]
        if _require_bool(entry.get("last_effective_gate_required"), label=f"{stage}.last_effective_gate_required") != expected_gate:
            raise ValueError(f"R4 stage {stage} last-effective gate policy drifted")
    if seen_datasets != set(inventory):
        raise ValueError("Every R4 inventory entry must belong to exactly one stage")

    bh0_names = stages["BH0"]["datasets"]
    if len(bh0_names) != 1:
        raise ValueError("R4 BH0 must contain exactly one smoke dataset")
    bh0 = inventory[bh0_names[0]]
    if bh0["query_states"] != 4 or bh0["conditions"] != ["standard"] or bh0["prediction_records_per_arm"] != 16:
        raise ValueError(
            "R4 BH0 must be smoke4: set/overwrite/clear/noop query states, reverse-cyclic4, standard only"
        )
    bh1_names = stages["BH1"]["datasets"]
    if len(bh1_names) != 1:
        raise ValueError("R4 BH1 must contain exactly one Transition32 dataset")
    bh1 = inventory[bh1_names[0]]
    if bh1["query_states"] != 32 or bh1["conditions"] != ["standard", "reset", "shuffle", "state_swap"]:
        raise ValueError("R4 BH1 must be Transition32 with the four locked conditions")
    if stages["BH0"]["score_suite"] != "smoke" or stages["BH1"]["score_suite"] != "transition32":
        raise ValueError("R4 BH0/BH1 score suites drifted")
    if stages["BH2"]["score_suite"] != "formal" or stages["BH3"]["score_suite"] != "formal":
        raise ValueError("R4 formal stage score suites drifted")


def load_amendment(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    amendment = load_json_object(path)
    require_json_values(
        amendment,
        {
            "schema": AMENDMENT_SCHEMA,
            "status": "prospective_before_any_r4_gpu_prediction",
        },
        "R4 amendment",
    )
    role = amendment.get("research_role")
    if not isinstance(role, Mapping):
        raise ValueError("R4 amendment research_role must be an object")
    require_json_values(
        role,
        {
            "name": "qwen_history_r4_three_arm_comparison",
            "training": False,
            "dreamlite_loaded": False,
            "teacher_or_oracle_state_used": False,
            "oracle_router_metadata_used_by_tagged_and_reducer": True,
        },
        "R4 research role",
    )
    reader = amendment.get("reader")
    if not isinstance(reader, Mapping):
        raise ValueError("R4 amendment reader must be an object")
    choice_family_sha256 = permutation_family_sha256(REVERSE_CYCLIC4)
    if choice_family_sha256 != EXPECTED_REVERSE_CYCLIC4_SHA256:
        raise ValueError("R4 reverse-cyclic4 implementation SHA256 drifted")
    expected_reader = {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": EXPECTED_READER_REVISION,
        "snapshot_key": "qwen_reader",
        "loss_mode": "listwise-choice",
        "choice_family": "reverse-cyclic4",
        "choice_family_sha256": EXPECTED_REVERSE_CYCLIC4_SHA256,
        "parameters_frozen": True,
        "visual_input": "fixed_blank_1024x1024_rgb_float32",
    }
    require_json_values(
        reader,
        expected_reader,
        "R4 Reader contract",
    )
    if set(reader) != set(expected_reader):
        raise ValueError("R4 Reader contract contains missing or unexpected fields")
    blank_image = amendment.get("blank_image")
    if blank_image != EXPECTED_VISUAL_INPUT:
        raise ValueError("R4 fixed blank-image visual input contract drifted")
    prompt = amendment.get("prompt_contract")
    if not isinstance(prompt, Mapping):
        raise ValueError("R4 amendment prompt_contract must be an object")
    instruction_sha256 = hashlib.sha256(R4_HISTORY_TASK_INSTRUCTION.encode("utf-8")).hexdigest()
    if instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise ValueError("R4 common instruction implementation SHA256 drifted")
    expected_prompt = {
        "instruction": R4_HISTORY_TASK_INSTRUCTION,
        "instruction_sha256": EXPECTED_INSTRUCTION_SHA256,
        "query_choices_or_target_in_memory": False,
        "future_event_access": False,
    }
    require_json_values(
        prompt,
        expected_prompt,
        "R4 prompt contract",
    )
    if set(prompt) != set(expected_prompt):
        raise ValueError("R4 prompt contract contains missing or unexpected fields")
    arms = amendment.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("R4 amendment arms must be an object")
    _validate_arm_contract(arms)
    execution = amendment.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("R4 amendment execution must be an object")
    require_json_values(
        execution,
        {
            "strict_order": list(STAGES),
            "arm_order": list(ARM_ORDER),
            "replica_order": ["A", "B"],
            "replica_a_device": "cuda:0",
            "replica_b_device": "cuda:1",
            "sequential_within_arm": True,
            "scientific_prediction_payload_exact": True,
            "raw_tagged_performance_failure_blocks": False,
            "integrity_replication_leakage_context_failure_blocks": True,
        },
        "R4 execution contract",
    )
    if amendment.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("R4 threshold contract drifted")
    lockbox = amendment.get("lockbox")
    if not isinstance(lockbox, Mapping):
        raise ValueError("R4 amendment lockbox must be an object")
    test_policy = lockbox.get("pre_bh2_test_policy")
    if "test_contents_unread_before_bh2_unlock" in lockbox:
        raise ValueError("R4 lockbox contains the obsolete misleading test-unread declaration")
    if test_policy != EXPECTED_PRE_BH2_TEST_POLICY:
        raise ValueError(
            "R4 pre-BH2 test policy must permit only SHA256 byte binding and forbid "
            "JSON semantic parsing, scoring, metric access, and evaluation"
        )
    expected_lockbox_keys = {
        "seed",
        "manifest_sha256",
        "lockbox_contract_sha256",
        "source_contract_sha256",
        "independent_generations_bitwise_identical",
        "pre_bh2_test_policy",
    }
    if set(lockbox) != expected_lockbox_keys:
        raise ValueError("R4 lockbox contains missing or unexpected fields")
    forbidden = amendment.get("forbidden")
    if forbidden != EXPECTED_FORBIDDEN:
        raise ValueError("R4 forbidden list does not match the SHA-only pre-BH2 test policy")
    _validate_inventory(amendment)
    return amendment, sha256_file(path)


def verify_formal_preflight(path: Path, *, expected_commit: str) -> tuple[dict[str, Any], str]:
    if COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase full Git commit")
    path = path.resolve()
    digest = verify_sha_sidecar(path)
    preflight = load_json_object(path)
    require_json_values(preflight, {"passed": True, "formal_ready": True}, "formal preflight")
    git = preflight.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit:
        raise ValueError("Formal preflight is not bound to the R4 Git commit")
    return preflight, digest


def verified_reader_snapshot(
    *,
    repo: Path,
    model_root: Path,
    preflight: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    lock = load_json_object(repo / "models.lock.json")
    specification = lock.get("models", {}).get("qwen_reader")
    if not isinstance(specification, Mapping):
        raise ValueError("models.lock.json does not contain qwen_reader")
    reader = amendment["reader"]
    require_json_values(
        specification,
        {"repo_id": reader["repo_id"], "revision": reader["revision"]},
        "locked R4 Qwen Reader",
    )
    model_dir = model_root.resolve() / Path(str(specification["local_dir"])).name
    if model_root.resolve().is_symlink() or model_dir.is_symlink():
        raise ValueError("R4 model root and Qwen snapshot must not be symlinks")
    current = verify_snapshot_manifest(
        manifest_path=model_dir / str(specification["snapshot_manifest"]),
        model_dir=model_dir,
        expected_repo_id=str(specification["repo_id"]),
        expected_revision=str(specification["revision"]),
    )
    preflight_models = preflight.get("models")
    reported_model = preflight_models.get("qwen_reader") if isinstance(preflight_models, Mapping) else None
    reported = reported_model.get("snapshot_manifest") if isinstance(reported_model, Mapping) else None
    if not isinstance(reported, Mapping) or dict(reported) != current:
        raise ValueError("Formal preflight Qwen snapshot binding drifted")
    verify_snapshot_binding(current)
    return current


def bind_data_files(amendment: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    expected = amendment["data"]
    if set(paths) != set(expected):
        raise ValueError("R4 data paths must exactly match the prospective data lock")
    return {
        name: require_file_sha(path.resolve(), str(expected[name]), name)
        for name, path in paths.items()
    }


def validate_scientific_command(command: Sequence[str]) -> None:
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("Every R4 scientific command must be a non-empty argv string list")
    if len(command) < 2 or Path(command[1]).name not in SCIENTIFIC_SCRIPTS:
        raise ValueError("R4 scientific command must invoke an allowlisted evaluator/scorer")
    flattened = " ".join(command).lower()
    for fragment in FORBIDDEN_COMMAND_FRAGMENTS:
        if fragment in flattened:
            raise ValueError(f"R4 command contains forbidden training/privileged fragment: {fragment}")
    script = Path(command[1]).name
    argv = list(command)
    if script == "qwen_history_r4.py":
        if "--method" not in argv or argv[argv.index("--method") + 1] not in ARM_METHODS.values():
            raise ValueError("R4 evaluator must declare one locked method")
        if "--choice-view-family" not in argv or argv[argv.index("--choice-view-family") + 1] != "reverse-cyclic4":
            raise ValueError("R4 evaluator must use reverse-cyclic4")
        if "--strict-determinism" not in argv:
            raise ValueError("R4 evaluator must enable strict determinism")
    elif script == "score_qwen_history_r4.py":
        if "--method" not in argv or argv[argv.index("--method") + 1] not in ARM_METHODS.values():
            raise ValueError("R4 scorer must declare one locked method")
        if "--suite" not in argv or argv[argv.index("--suite") + 1] not in SCORE_SUITES:
            raise ValueError("R4 scorer has an unsupported suite")
    elif script == "compare_qwen_history_r4.py":
        if "--suite" not in argv or argv[argv.index("--suite") + 1] not in COMPARISON_SUITES:
            raise ValueError("R4 comparison has an unsupported suite")


def validate_replica_pair(pair: Sequence[Mapping[str, Any]], *, method: str) -> None:
    if method not in ARM_METHODS.values():
        raise ValueError("Unknown R4 method")
    if len(pair) != 2 or [entry.get("replica") for entry in pair] != ["A", "B"]:
        raise ValueError("Every R4 replica pair must be ordered exactly A then B")
    for entry in pair:
        replica = str(entry["replica"])
        expected_device = DEVICES[replica]
        if entry.get("device") != expected_device:
            raise ValueError(f"R4 replica {replica} must use {expected_device}")
        argv = entry.get("argv")
        if not isinstance(argv, list):
            raise ValueError("R4 replica argv must be a list")
        validate_scientific_command(argv)
        if argv[argv.index("--method") + 1] != method:
            raise ValueError("R4 replica method differs from its arm")
        if "--device" not in argv or argv[argv.index("--device") + 1] != expected_device:
            raise ValueError(f"R4 replica {replica} argv must explicitly use {expected_device}")
        if "--replica-id" not in argv or argv[argv.index("--replica-id") + 1] != replica:
            raise ValueError(f"R4 replica {replica} argv must bind its replica id")


def verify_prediction_jsonl(output: Mapping[str, Any]) -> None:
    path = Path(str(output["path"])).resolve()
    expected_records = _require_positive_int(output.get("expected_records"), label="expected_records")
    expected_method = output.get("expected_method")
    expected_conditions = output.get("expected_conditions")
    expected_probe_role = output.get("expected_probe_role")
    if expected_method not in ARM_METHODS.values():
        raise ValueError("Prediction output has an unknown expected method")
    if not isinstance(expected_conditions, list) or not expected_conditions:
        raise ValueError("Prediction output requires expected_conditions")
    records = 0
    views: dict[tuple[str, str], set[int]] = {}
    conditions_by_query: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            require_json_values(
                row,
                {
                    "schema_version": PREDICTION_SCHEMA,
                    "method": expected_method,
                    "choice_view_family": "reverse-cyclic4",
                    "context_truncated": False,
                },
                f"{path.name}:{line_number}",
            )
            if row.get("condition") not in expected_conditions:
                raise ValueError(f"{path}:{line_number} has an unexpected condition")
            if expected_probe_role == "delayed" and row.get("probe_role") != "delayed":
                raise ValueError(f"{path}:{line_number} is not a delayed probe")
            query_id = row.get("base_query_id", row.get("query_id"))
            view = row.get("choice_view_index")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number} is missing a query identity")
            if not isinstance(view, int) or isinstance(view, bool) or not 0 <= view < 4:
                raise ValueError(f"{path}:{line_number} has an invalid view index")
            key = (query_id, str(row["condition"]))
            bucket = views.setdefault(key, set())
            if view in bucket:
                raise ValueError(f"{path}:{line_number} duplicates reverse-cyclic view {view}")
            bucket.add(view)
            conditions_by_query.setdefault(query_id, set()).add(str(row["condition"]))
            records += 1
    if records != expected_records:
        raise ValueError(f"{path} expected {expected_records} records, observed {records}")
    expected_views = {0, 1, 2, 3}
    if any(bucket != expected_views for bucket in views.values()):
        raise ValueError(f"{path} has an incomplete reverse-cyclic4 query-condition group")
    expected_condition_set = set(expected_conditions)
    if any(bucket != expected_condition_set for bucket in conditions_by_query.values()):
        raise ValueError(f"{path} has incomplete conditions for at least one query")
    expected_queries = expected_records // (4 * len(expected_conditions))
    if len(conditions_by_query) != expected_queries:
        raise ValueError(f"{path} expected {expected_queries} query states, observed {len(conditions_by_query)}")


def verify_score_report(
    path: Path,
    *,
    method: str,
    suite: str,
    require_data_readability: bool,
) -> dict[str, Any]:
    report = load_json_object(path)
    require_json_values(
        report,
        {"schema": SCORE_SCHEMA, "method": method, "suite": suite, "passed": True},
        f"R4 score {path.name}",
    )
    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("passed") is not True:
        raise ValueError(f"R4 score {path.name} failed integrity")
    replication = report.get("replication")
    if (
        not isinstance(replication, Mapping)
        or replication.get("passed") is not True
        or replication.get("bitwise_scientific_payload_match") is not True
    ):
        raise ValueError(f"R4 score {path.name} failed exact A/B replication")
    gate = report.get("scientific_gate")
    if not isinstance(gate, Mapping):
        raise ValueError(f"R4 score {path.name} has no scientific_gate")
    if bool(gate.get("data_readability_required")) != require_data_readability:
        raise ValueError(f"R4 score {path.name} data-readability policy drifted")
    if require_data_readability and gate.get("passed") is not True:
        raise ValueError(f"R4 score {path.name} failed the preregistered data-readability gate")
    return report


def verify_comparison_report(path: Path, *, suite: str) -> dict[str, Any]:
    report = load_json_object(path)
    require_json_values(
        report,
        {"schema": COMPARISON_SCHEMA, "suite": suite, "passed": True},
        f"R4 comparison {path.name}",
    )
    pairing = report.get("identity_pairing")
    if not isinstance(pairing, Mapping) or pairing.get("passed") is not True:
        raise ValueError(f"R4 comparison {path.name} failed identity pairing")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, (list, Mapping)) or not comparisons:
        raise ValueError(f"R4 comparison {path.name} is empty")
    return report


def verify_bh2_last_effective_dev_gate(
    path: Path, *, expected_records: int = 5008, minimum_accuracy: float = 0.95
) -> dict[str, Any]:
    report = verify_score_report(
        path,
        method=ARM_METHODS["last_effective"],
        suite="formal",
        require_data_readability=False,
    )
    metrics = report.get("descriptive_metrics")
    standard = metrics.get("standard") if isinstance(metrics, Mapping) else None
    if not isinstance(standard, Mapping):
        raise ValueError("R4 BH2 last-effective score has no standard metrics")
    count = standard.get("count")
    accuracy = standard.get("accuracy")
    if count != expected_records:
        raise ValueError(
            f"R4 BH2 last-effective dev expected {expected_records} standard records, got {count}"
        )
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ValueError("R4 BH2 last-effective dev accuracy is not numeric")
    if float(accuracy) < minimum_accuracy:
        raise ValueError(
            "R4 BH2 last-effective dev data-readability gate failed: "
            f"accuracy={float(accuracy):.6f} < {minimum_accuracy:.6f}"
        )
    return {
        **report,
        "bh2_last_effective_dev_gate": {
            "passed": True,
            "accuracy": float(accuracy),
            "minimum_accuracy": minimum_accuracy,
            "records": count,
        },
    }


def verify_declared_output(output: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(str(output["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"Required R4 output is missing: {path}")
    required = output.get("required_values", {})
    if required:
        value = load_json_object(path)
        if not isinstance(required, Mapping):
            raise ValueError("required_values must be an object")
        require_json_values(value, required, str(output.get("label", path.name)))
    validator = output.get("validator")
    if validator is None:
        return None
    if validator == "prediction_jsonl":
        verify_prediction_jsonl(output)
        return None
    if validator == "r4_score":
        return verify_score_report(
            path,
            method=str(output["expected_method"]),
            suite=str(output["suite"]),
            require_data_readability=bool(output.get("require_data_readability")),
        )
    if validator == "r4_comparison":
        return verify_comparison_report(path, suite=str(output["suite"]))
    if validator == "r4_bh2_last_effective_dev":
        return verify_bh2_last_effective_dev_gate(
            path,
            expected_records=int(output.get("expected_records", 5008)),
            minimum_accuracy=float(output.get("minimum_accuracy", 0.95)),
        )
    raise ValueError(f"Unknown R4 output validator: {validator}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return load_json_object(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON artifact {path}: {exc}") from exc


__all__ = [
    "ALLOWED_CONDITIONS",
    "AMENDMENT_SCHEMA",
    "ARM_METHODS",
    "ARM_ORDER",
    "COMPARISON_SCHEMA",
    "DEVICES",
    "PLAN_PROTOCOL",
    "PREDICTION_REPORT_SCHEMA",
    "PREDICTION_SCHEMA",
    "SCORE_SCHEMA",
    "STAGES",
    "STAGE_EVIDENCE_PROTOCOL",
    "STAGE_SPEC_PROTOCOL",
    "WRAPPER_FAILURE_PROTOCOL",
    "bind_data_files",
    "load_amendment",
    "read_json",
    "validate_replica_pair",
    "validate_scientific_command",
    "verified_reader_snapshot",
    "verify_comparison_report",
    "verify_declared_output",
    "verify_formal_preflight",
    "verify_bh2_last_effective_dev_gate",
    "verify_prediction_jsonl",
    "verify_score_report",
]
