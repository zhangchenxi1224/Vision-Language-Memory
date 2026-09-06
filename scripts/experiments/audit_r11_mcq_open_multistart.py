"""CPU-only audit of downloaded R11 multistart artifacts; never edits the run.

An audit cross-checks saved records and tensors, not a second model execution. Model
weight immutability is checked through the recorded start/end snapshot bindings;
the remote weights are not available to this offline auditor. Historical comparisons
are descriptive and never add observations to the eight independent random starts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.reader.open_answer import score_short_answer  # noqa: E402
from vision_memory.repro import canonical_object_sha256, canonical_tensor_sha256  # noqa: E402

STEPS = [0, 64, 128, 192, 256]
INITIAL_SHA = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coverage(rows: list[dict[str, Any]], key: str, expected: list[Any], label: str) -> dict[Any, dict[str, Any]]:
    actual = [row[key] for row in rows]
    require(len(actual) == len(expected) and len(set(actual)) == len(expected) and set(actual) == set(expected),
            f"{label}: missing, duplicate, or unexpected {key} values")
    return {row[key]: row for row in rows}


def tensor(value: Any, label: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and tuple(value.shape) == (1, 4, 128, 128) and torch.isfinite(value).all(),
            f"{label}: expected finite FP32 [1,4,128,128] tensor")
    return value


def load(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if expected_sha is not None:
        require(sha(path) == expected_sha, f"file SHA256 mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-8) -> None:
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{label}: recorded={actual!r}, recomputed={expected!r}")


def check_mcq(row: dict[str, Any], permutations: list[list[int]], target: dict[str, Any], *, probe: bool) -> None:
    permutation = permutations[row["view_index"]]
    require(row["permutation"] == permutation, "MCQ permutation mismatch")
    gold_position = permutation.index(target["original"]["answer_index"])
    logits = torch.tensor(row["choice_logits"], dtype=torch.float32)
    require(logits.shape == (4,) and torch.isfinite(logits).all(), "MCQ scores must be four finite values")
    require(row["correct"] is (int(logits.argmax()) == gold_position), "MCQ correctness disagrees with candidate scores")
    alternative = torch.cat((logits[:gold_position], logits[gold_position + 1:]))
    close(row["margin"], float(logits[gold_position] - alternative.max()), "MCQ margin", 1e-6)
    # CPU FP32 logsumexp need not match the CUDA kernel bit-for-bit.
    close(row["ce"], float(torch.logsumexp(logits, 0) - logits[gold_position]), "MCQ CE", 1e-5)
    if not probe:
        require(row["target_index"] == gold_position and row["condition"] == "matched", "MCQ target/condition mismatch")
        require(row["choices"] == [target["original"]["choices"][i] for i in permutation], "MCQ choice text mismatch")
        require(torch.equal(-torch.tensor(row["choice_mean_nll"], dtype=torch.float32), logits), "MCQ NLL/logit mismatch")


def check_generation(row: dict[str, Any], target: dict[str, Any], expected_image_sha: str) -> None:
    require(row["optimizer_step"] == 256, "Generation did not use the fixed endpoint")
    require(row["query"] == target["inputs"][row["prompt_id"]], "Generation query differs from preregistration")
    require(row["image_sha256"] == expected_image_sha, "Generation image does not match its saved condition tensor")
    require(row["scorer"] == score_short_answer(row["raw"], target["scorer_metadata"]["gold"]), "Strict raw rescoring differs")
    require(isinstance(row["chat_prompt"], str) and row["query"] in row["chat_prompt"], "Serialized prompt is missing its question")
    tokens, inputs, eos = row["generated_token_ids"], row["input_token_ids"], row["eos_token_ids"]
    require(all(isinstance(t, int) and t >= 0 for t in tokens + inputs + eos), "Invalid saved token IDs")
    require(len(tokens) == row["generated_token_count"] <= 32 and len(inputs) == row["prompt_token_count"], "Token count mismatch")
    reached = any(token in eos for token in tokens)
    reason = "eos" if reached else "token_limit" if len(tokens) >= 32 else "other_stop"
    require(row["eos_reached"] is reached and row["truncated"] is (not reached) and row["finish_reason"] == reason,
            "EOS/truncation metadata disagrees with token IDs")
    require(isinstance(row["raw_with_special_tokens"], str), "Missing raw output with special tokens")


def audit_run(root: Path, spec: dict[str, Any], config: dict[str, Any], initial: dict[str, Any],
              controls: dict[str, Any]) -> dict[str, Any]:
    directory = root / "runs" / spec["run_id"]
    manifest, terminal = read_json(directory / "manifest.json"), read_json(directory / "terminal.json")
    initial_sha = canonical_tensor_sha256(initial["latent_fp32"])
    for record in (manifest, terminal):
        require(all(record[key] == value for key, value in spec.items()), "Run identity mismatch")
        require(record["initial_latent_sha256"] == initial_sha, "Run initial hash differs from its materialized initial tensor")
        require(record["optimizer_steps"] == 256, "Run has not completed 256 updates")
    require(terminal["status"] == "completed", "Run terminal is not completed")
    require(manifest["reference_sha256"] == INITIAL_SHA and manifest["gold_eos_appended"] is False, "Run reference/EOS contract mismatch")
    repeated = read_json(directory / "initial_reproducibility.json")
    require(repeated["loss_exactly_equal"] is True and repeated["gradient_exactly_equal"] is True
            and repeated["loss_max_abs_difference"] == 0 and repeated["gradient_max_abs_difference"] == 0
            and len(repeated["gradient_sha256"]) == 2
            and repeated["gradient_sha256"][0] == repeated["gradient_sha256"][1], "Repeated-gradient record failed")
    metric_rows = read_rows(directory / "metrics.jsonl")
    metrics = coverage(metric_rows, "optimizer_step", list(range(1, 257)), "metrics")
    indices = coverage(read_rows(directory / "latent_index.jsonl"), "optimizer_step", list(range(257)), "latent index")
    checkpoints = coverage(read_rows(directory / "checkpoint_index.jsonl"), "optimizer_step", STEPS, "checkpoint index")
    require({p.name for p in (directory / "latents").glob("*.pt")} == {f"step-{i:03d}.pt" for i in range(257)}, "Actual latent file coverage mismatch")
    require({p.name for p in (directory / "checkpoints").glob("*.pt")} == {f"step-{i:03d}.pt" for i in STEPS}, "Actual checkpoint file coverage mismatch")
    history, rng_sha, zero_gradient_steps = [], None, []
    for step in range(257):
        index = indices[step]
        require(index["path"] == f"latents/step-{step:03d}.pt", "Unexpected latent index path")
        payload = load(directory / index["path"], index["file_sha256"])
        current = tensor(payload["latent_fp32"], f"latent step {step}")
        require(payload["optimizer_step"] == step and canonical_tensor_sha256(current) == index["latent_sha256"], "Latent step/canonical hash mismatch")
        if step == 0:
            require(torch.equal(current, initial["latent_fp32"]), "Step zero differs from saved initialization")
        else:
            metric = metrics[step]
            require(all(metric[key] == value for key, value in spec.items()), "Metric run identity mismatch")
            for key in config["logging"]["scalar_training_metrics"]:
                if key != "training_view_if_mcq":
                    require(math.isfinite(metric[key]), f"Nonfinite metric {key} at step {step}")
            require(0 <= metric["gradient_nonzero_fraction"] <= 1 and metric["gradient_l2"] >= 0, "Invalid gradient statistics")
            close(metric["gradient_l2"], metric["gradient_rms"] * 256, "Gradient L2/RMS relation")
            zero = metric["gradient_nonzero_fraction"] == 0
            require(metric["gradient_all_zero"] is zero, "All-zero gradient diagnostic mismatch")
            require(zero == (metric["gradient_l2"] == 0 and metric["gradient_rms"] == 0), "Zero-gradient fraction/norm mismatch")
            if zero:
                zero_gradient_steps.append(step)
            require(metric["training_view_if_mcq"] == ((step - 1 + manifest["target_phase"]) % 4 if spec["arm"] == "mcq" else None), "Training view schedule mismatch")
            update = current.double() - history[-1].double()
            close(metric["actual_update_l2"], float(update.norm()), "Actual update L2")
            close(metric["actual_update_rms"], float(update.square().mean().sqrt()), "Actual update RMS")
            close(metric["latent_rms"], float(current.double().square().mean().sqrt()), "Latent RMS")
            for key, other in (("delta_from_own_start_rms", initial["latent_fp32"]),
                               ("delta_from_shared_reference_rms", controls["reference_fp32"])):
                close(metric[key], float((current.double() - other.double()).square().mean().sqrt()), key)
            if step >= 4:
                close(metric["four_step_update_rms"], float((current.double() - history[0].double()).square().mean().sqrt()), "Four-step movement")
            else:
                require(metric["four_step_update_rms"] is None, "Four-step diagnostic is defined too early")
        if step in STEPS:
            index = checkpoints[step]
            require(index["path"] == f"checkpoints/step-{step:03d}.pt", "Unexpected checkpoint index path")
            checkpoint = load(directory / index["path"], index["file_sha256"])
            require(checkpoint["optimizer_step"] == step and torch.equal(checkpoint["latent_fp32"], current)
                    and torch.equal(checkpoint["initial_latent_fp32"], initial["latent_fp32"]), "Checkpoint latent/state mismatch")
            require(all(checkpoint[key] == value for key, value in spec.items()), "Checkpoint run identity mismatch")
            optimizer = checkpoint["optimizer"]
            require(len(optimizer["param_groups"]) == 1 and len(optimizer["param_groups"][0]["params"]) == 1, "Adam must optimize one tensor")
            group = optimizer["param_groups"][0]
            require(group["lr"] == 0.05 and tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
                    and group["weight_decay"] == 0 and group["amsgrad"] is False, "Adam hyperparameters changed")
            require(set(checkpoint["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "Missing checkpoint RNG state")
            if step == 0:
                require(not optimizer["state"], "Adam state was not reset")
                rng_sha = canonical_object_sha256(checkpoint["rng"])
                require(canonical_tensor_sha256(checkpoint["rng"]["torch_cpu"]) == manifest["initial_rng_sha256"], "Initial CPU RNG hash mismatch")
            else:
                require(len(optimizer["state"]) == 1, "Missing Adam state")
                state = next(iter(optimizer["state"].values()))
                require(float(state["step"]) == step, "Adam step counter mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor(state[key], f"Adam {key}")
                require((state["exp_avg_sq"] >= 0).all(), "Adam variance is negative")
        history = (history + [current])[-4:]
    endpoint = load(directory / "endpoint_raw.pt", terminal["endpoint_file_sha256"])
    require(endpoint["optimizer_step"] == 256 and torch.equal(endpoint["latent_fp32"], history[-1]), "Endpoint differs from step 256")
    require(endpoint["initial_latent_sha256"] == initial_sha
            and canonical_tensor_sha256(endpoint["latent_fp32"]) == terminal["endpoint_latent_sha256"], "Endpoint tensor/initial hash mismatch")
    image = endpoint["image"]
    require(tuple(image.shape) == (1, 3, 1024, 1024) and image.is_floating_point()
            and torch.isfinite(image).all() and (image >= 0).all() and (image <= 1).all(), "Invalid endpoint image")
    probes = read_rows(directory / "fixed_probes.jsonl")
    probe_keys = [(row["optimizer_step"], row["probe"], row.get("view_index")) for row in probes]
    expected = {(s, "mcq_forward", v) for s in STEPS for v in range(4)} | {(s, "open_gold_ce", None) for s in STEPS}
    require(len(probe_keys) == 25 and set(probe_keys) == expected, "Fixed probe coverage mismatch")
    for row in probes:
        require(all(row[key] == value for key, value in spec.items()) and math.isfinite(row["ce"]), "Fixed probe identity/value mismatch")
        if row["probe"] == "mcq_forward":
            check_mcq(row, config["training"]["arms"]["mcq"]["train_choice_permutations"], config["target"], probe=True)
        else:
            require(row["append_eos"] is False and row["target_token_ids"], "Open probe target convention mismatch")
    generations, mcq = read_rows(directory / "generations.jsonl"), read_rows(directory / "mcq_endpoint.jsonl")
    keys = [(r["condition"], r["prompt_id"]) for r in generations]
    expected = {(c, p) for c in ("matched", "blank", "fixed_donor") for p in ("original_open", "paraphrase_open")}
    require(len(keys) == 6 and set(keys) == expected, "Endpoint generation coverage mismatch")
    mcq_by_view = coverage(mcq, "view_index", list(range(4)), "endpoint MCQ")
    image_hashes = {"matched": canonical_tensor_sha256(image), **{
        name: canonical_tensor_sha256(controls[name]) for name in ("blank", "fixed_donor")}}
    for row in generations + mcq:
        require(all(row[key] == value for key, value in spec.items()), "Endpoint row identity mismatch")
    for row in generations:
        check_generation(row, config["target"], image_hashes[row["condition"]])
    for row in mcq_by_view.values():
        require(row["optimizer_step"] == 256, "MCQ endpoint step mismatch")
        check_mcq(row, config["endpoint_evaluation"]["mcq_choice_permutations"], config["target"], probe=False)
    for key, count in (("latent_count", 257), ("checkpoint_count", 5), ("probe_count", 25), ("generation_count", 6), ("mcq_count", 4)):
        require(terminal[key] == count, f"Terminal {key} disagrees with artifacts")
    return {"run_id": spec["run_id"], "passed": True, "metrics": 256, "latent_files_hashed": 257,
            "checkpoint_files_hashed": 5, "probes": 25, "generations": 6, "mcq": 4,
            "initial_latent_sha256": initial_sha, "initial_complete_rng_sha256": rng_sha,
            "endpoint_latent_sha256": terminal["endpoint_latent_sha256"], "endpoint_image_sha256": image_hashes["matched"],
            "zero_gradient_steps": zero_gradient_steps, "repeated_gradient": repeated}


def audit(root: Path, expected_config: Path, old_root: Path | None, report: dict[str, Any]) -> None:
    manifest, summary, terminal = (read_json(root / name) for name in ("manifest.json", "summary.json", "terminal.json"))
    config = manifest["config"]
    require(config == read_json(expected_config), "Manifest configuration differs from expected preregistration")
    require(terminal["status"] == "completed" and terminal["technical_passed"] is True
            and summary["technical_passed"] is True and summary["complete_records"] is True, "Overall run has not passed completion")
    require(summary["formal_success"] is False and terminal["formal_success"] is False, "Exploratory result mislabeled formal success")
    require(terminal["run_count"] == 18 and terminal["optimizer_steps"] == 4608, "Overall terminal count mismatch")
    specs = config["training"]["run_order"]
    require(len(specs) == 18 and {p.name for p in (root / "runs").iterdir() if p.is_dir()} == {s["run_id"] for s in specs}, "Actual run directory coverage differs from 18 preregistered runs")
    report["provenance"] = {"training_git_commit": manifest["git_commit"], "manifest_sha256": sha(root / "manifest.json"),
                            "config_recorded_sha256": manifest["config_sha256"], "configuration_semantically_matches": True}
    snapshots = read_json(root / "model_snapshot_verification_end.json")
    require(snapshots["passed"] is True and summary["snapshots_unchanged"] is True
            and snapshots["bindings"] == manifest["model_snapshot_payloads_start"]
            == manifest["prerequisite"]["model_snapshot_payloads"], "Recorded snapshot start/end/prerequisite bindings differ")
    require(set(snapshots["bindings"]) == {"dreamlite_mobile", "qwen_reader"}
            and all(b["passed"] is True for b in snapshots["bindings"].values()), "Incomplete model snapshot bindings")
    for name in ("vae", "reader"):
        require(snapshots["frozen_parameters"][name] == {"trainable_parameters": 0, "parameter_gradient_tensors": 0, "training": False}, "Frozen parameter audit failed")
    determinism = manifest["determinism"]
    require(determinism["deterministic_algorithms"] is True and determinism["deterministic_warn_only"] is False
            and determinism["sdpa"] == {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}, "Recorded strict determinism disabled")
    report["recorded_model_and_determinism_audit"] = {"passed": True, "snapshot_bindings": snapshots["bindings"],
                                                   "frozen_parameters": snapshots["frozen_parameters"]}
    controls, control_audit = load(root / "fixed_controls.pt"), read_json(root / "control_audit.json")
    reference = tensor(controls["reference_fp32"], "shared reference")
    require(canonical_tensor_sha256(reference) == control_audit["reference_sha256"] == INITIAL_SHA, "Common reference hash mismatch")
    for condition in ("blank", "fixed_donor"):
        require(canonical_tensor_sha256(controls[condition]) == control_audit["image_sha256"][condition], "Fixed control tensor hash mismatch")
    initials, run_audits = {}, []
    init_specs = config["initialization"]["starts"]
    init_index = coverage(read_rows(root / "initial_index.jsonl"), "init_id", [s["init_id"] for s in init_specs], "initial index")
    for spec in init_specs:
        item = load(root / "initials" / f"{spec['init_id']}.pt", init_index[spec["init_id"]]["file_sha256"])
        value = tensor(item["latent_fp32"], "materialized initial")
        require(canonical_tensor_sha256(value) == item["latent_sha256"] == init_index[spec["init_id"]]["latent_sha256"], "Initial tensor hash mismatch")
        require(torch.equal(item["reference_fp32"], reference), "Initial uses a different shared reference")
        for field in ("raw_noise_fp32", "epsilon_fp64"):
            hash_key = "noise_sha256" if field == "raw_noise_fp32" else "epsilon_sha256"
            require(canonical_tensor_sha256(item[field]) == item[hash_key], f"Initial {field} hash mismatch")
        require(item["seed"] == spec["init_seed"] and item["rho"] == spec["rho"], "Initial seed/radius metadata mismatch")
        expected = (reference.double() + spec["rho"] * reference.double().square().mean().sqrt() * item["epsilon_fp64"]).float()
        require(torch.equal(value, expected), "Initial tensor does not follow saved epsilon/reference formula")
        if spec["init_seed"] is not None:
            close(float(item["epsilon_fp64"].square().mean().sqrt()), 1.0, "Normalized direction RMS")
        initials[spec["init_id"]] = item
    for spec in specs:
        try:
            result = audit_run(root, spec, config, initials[spec["init_id"]], controls)
        except Exception as error:
            result = {"run_id": spec["run_id"], "passed": False, "error_type": type(error).__name__, "error": str(error)}
        run_audits.append(result)
        print(json.dumps({"run_id": spec["run_id"], "audit_passed": result["passed"]}), flush=True)
    report["runs"] = run_audits
    require(all(row["passed"] for row in run_audits), "One or more individual artifact audits failed")
    by_run = {r["run_id"]: r for r in run_audits}
    for spec in init_specs:
        pair = [by_run[f"{spec['init_id']}-{arm}"] for arm in ("mcq", "open")]
        require(pair[0]["initial_latent_sha256"] == pair[1]["initial_latent_sha256"]
                and pair[0]["initial_complete_rng_sha256"] == pair[1]["initial_complete_rng_sha256"], "Paired initial tensor/RNG mismatch")
    generations, mcq = read_rows(root / "generations.jsonl"), read_rows(root / "mcq_endpoint.jsonl")
    local_generations = sum((read_rows(root / "runs" / s["run_id"] / "generations.jsonl") for s in specs), [])
    local_mcq = sum((read_rows(root / "runs" / s["run_id"] / "mcq_endpoint.jsonl") for s in specs), [])
    require(generations == local_generations and mcq == local_mcq, "Root endpoint records differ from individual run records")
    require(len(generations) == 108 and len(mcq) == 72, "Root endpoint count mismatch")
    control_repeats = {}
    for condition in ("blank", "fixed_donor"):
        control_repeats[condition] = {}
        for prompt in ("original_open", "paraphrase_open"):
            selected = [r for r in generations if r["condition"] == condition and r["prompt_id"] == prompt]
            require(len(selected) == 18, "A fixed control lacks 18 technical repeats")
            distinct = {key: len({json.dumps(r[key], sort_keys=True) for r in selected}) for key in (
                "image_sha256", "query", "chat_prompt", "input_token_ids", "raw", "raw_with_special_tokens",
                "generated_token_ids", "finish_reason", "truncated")}
            control_repeats[condition][prompt] = {"count": 18, "distinct_value_counts": distinct,
                                                "raw": selected[0]["raw"], "passed": all(n == 1 for n in distinct.values())}
    report["fixed_control_repeats"] = control_repeats
    require(all(item["passed"] for prompts in control_repeats.values() for item in prompts.values()), "Repeated fixed control image/prompt/output inconsistency")
    rebuilt = {}
    for arm in ("mcq", "open"):
        ids = [f"noise-seed-{seed:02d}-{arm}" for seed in range(8)]
        matched = {(r["run_id"], r["prompt_id"]): r["scorer"]["strict_correct"] for r in generations if r["condition"] == "matched"}
        rebuilt[arm] = {
            "open_original": sum(matched[(run, "original_open")] for run in ids) / 8,
            "open_robust": sum(all(matched[(run, prompt)] for prompt in ("original_open", "paraphrase_open")) for run in ids) / 8,
            "mcq_all4": sum(all(r["correct"] for r in mcq if r["run_id"] == run) for run in ids) / 8,
        }
        require(all(summary["by_arm"][arm][key] == value for key, value in rebuilt[arm].items()), "Summary accuracy differs from independent raw rescoring")
    report["independent_behavior_rebuild"] = rebuilt
    report["actual_counts"] = {"runs": 18, "metrics": sum(r["metrics"] for r in run_audits),
                               "latents_hashed": sum(r["latent_files_hashed"] for r in run_audits),
                               "checkpoints_hashed": sum(r["checkpoint_files_hashed"] for r in run_audits),
                               "fixed_probes": sum(r["probes"] for r in run_audits), "generations": 108, "mcq": 72}
    require(summary["optimizer_steps"] == report["actual_counts"]["metrics"] == 4608
            and summary["run_count"] == 18 and summary["random_start_count"] == 8, "Summary counts differ from real records")
    if old_root is not None:
        historical = {"descriptive_only": True, "adds_independent_samples": False}
        try:
            directory = old_root / "target-01" / "run"
            old = load(directory / "endpoint_raw.pt", config["target"]["artifact"]["endpoint_file_sha256"])
            require(canonical_tensor_sha256(old["latent_fp32"]) == config["target"]["artifact"]["endpoint_tensor_sha256"], "Historical endpoint hash mismatch")
            new = load(root / "runs/blank-mcq/endpoint_raw.pt")
            difference = new["latent_fp32"].double() - old["latent_fp32"].double()
            old_metrics = coverage(read_rows(directory / "metrics.jsonl"), "optimizer_step", list(range(1, 257)), "historical metrics")
            new_metrics = read_rows(root / "runs/blank-mcq/metrics.jsonl")
            loss_differences = [abs(r["loss_before_step"] - old_metrics[r["optimizer_step"]]["loss_before_step"]) for r in new_metrics]
            historical.update({"available_and_valid": True, "old_endpoint_file_sha256": sha(directory / "endpoint_raw.pt"),
                               "blank_mcq_endpoint_exactly_equal": torch.equal(new["latent_fp32"], old["latent_fp32"]),
                               "endpoint_image_exactly_equal": torch.equal(new["image"], old["image"]),
                               "endpoint_rmse": float(difference.square().mean().sqrt()),
                               "loss_steps_compared": len(loss_differences), "loss_steps_exact": sum(d == 0 for d in loss_differences),
                               "max_absolute_loss_difference": max(loss_differences)})
        except Exception as error:
            historical.update({"available_and_valid": False, "error": str(error)})
        report["historical_blank_mcq_comparison"] = historical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/r11_mcq_open_multistart.json")
    parser.add_argument("--old-run-root", type=Path)
    args = parser.parse_args()
    root, output = args.run_root.resolve(strict=True), args.output.resolve()
    require(output != root and root not in output.parents, "Audit output must be outside the source result directory")
    require(not output.exists(), "Refusing to overwrite an existing audit output")
    started = time.perf_counter()
    torch.set_num_threads(1)
    report: dict[str, Any] = {
        "schema": "vision_memory.r11-mcq-open-multistart-offline-audit.v1", "run_root": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(Path(__file__)),
        "passed": False, "formal_success": False, "source_results_modified": False,
        "limitations": ["No GPU/model rerun; snapshot verification records are cross-checked, remote weights are not rehashed.",
                        "Recorded gradients are checked for reproducibility and consistency; gradient tensors are not retained for every update.",
                        "Fixed blank/donor repeats are technical repeats, not additional independent starts."],
    }
    try:
        audit(root, args.config, args.old_run_root, report)
        report["passed"] = True
    except Exception as error:
        report["error_type"], report["error"] = type(error).__name__, str(error)
    report["elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "audit": str(output), "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
