"""CPU audit and set-valued teacher export for the fresh EOS oracle campaigns.

The bank contains observed, successful step-256 latents, never an unverified
centroid, best step, old MCQ endpoint, or answer-prefix surrogate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from vision_memory.reader.open_eos import generation_diagnostics
from vision_memory.repro import canonical_tensor_sha256

SCHEMA = "latent-teacher-bank/v1"
SUFFIX = "\nUse the memory image to answer.\nAnswer with a short phrase only."
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
SELECTION = {
    "endpoint_step": 256, "objective": "mean_answer_ce + eos_ce", "lambda_eos": 1.0,
    "primary": "original_open raw greedy max_new_tokens=32 exact match",
    "robustness": "all five fixed-instruction question variants, reported separately",
    "controls": "blank and different-answer donor reported; never assumed zero",
    "deduplication": "same question and exact canonical final-latent SHA; keep every source run",
    "membership": "all unique primary-success endpoints; no mean or best-step substitution",
    "stage_gate": "every planned fresh EOS run and campaign end audit must complete",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def local_artifact(root: Path, name: str, expected_sha: str | None = None) -> Path:
    root = Path(root).resolve()
    path = Path(name)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root) and path.is_file(), f"Missing or escaping artifact: {path}")
    if expected_sha is not None:
        require(file_sha256(path) == expected_sha, f"Artifact SHA mismatch: {path}")
    return path


def tensor(value: Any, *, latent: bool = True) -> torch.Tensor:
    require(isinstance(value, torch.Tensor), "Expected a tensor")
    cpu = value.detach().cpu().contiguous()
    require(cpu.dtype == torch.float32 and bool(torch.isfinite(cpu).all()), "Expected finite raw FP32 values")
    if latent:
        require(tuple(cpu.shape) == (1, 4, 128, 128), f"Unexpected VAE coordinates: {tuple(cpu.shape)}")
    else:
        require(cpu.ndim == 4 and cpu.shape[0] == 1 and cpu.shape[1] == 3, "Expected batch-one RGB")
    return cpu


def source_audit(repo: Path, commit: str, bindings: dict[str, str]) -> None:
    require(len(commit) == 40 and bindings, "Missing exact source identity")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    require(observed == commit, "Oracle checkout commit changed")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip(), "Oracle checkout is dirty")
    for name, sha in bindings.items():
        local_artifact(repo, name, sha)


def validate_prompts(prompts: dict[str, str]) -> None:
    require(set(prompts) == {"original_open", "paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4"}, "Incomplete question-variant panel")
    require(len(set(prompts.values())) == 5, "Question variants are duplicated")
    for prompt in prompts.values():
        require(prompt.endswith(SUFFIX) and prompt.count(SUFFIX) == 1, "Fixed two-line instructions changed")
        sentence = prompt[:-len(SUFFIX)]
        require(bool(sentence.strip()) and "\n" not in sentence and "Choose exactly one option" not in sentence, "Invalid Open question sentence")


def direct_prompt_protocol(config: dict) -> dict:
    """Accept the two deployed protocols without treating held-out prompts as training."""
    base = {"optimizer":"Adam", "steps":256, "lr":.05, "lambda_eos":1.}
    original = ["original_open"]
    multiple = ["original_open", "paraphrase_1", "paraphrase_2"]
    if config["training"] == {**base, "prompts":original}:
        require(not config.get("heldout_prompts"), "Legacy Direct heldout declaration changed")
        return {"training_prompts":original, "heldout_prompts":[f"paraphrase_{i}" for i in range(1,5)],
                "prompt_schedule":"constant_original"}
    require(config["training"] == {**base, "prompts":multiple, "prompt_schedule":"round_robin_zero_based"},
            "Direct EOS training contract changed")
    require(config.get("heldout_prompts") == ["paraphrase_3", "paraphrase_4"],
            "Direct heldout prompt contract changed")
    return {"training_prompts":multiple, "heldout_prompts":config["heldout_prompts"],
            "prompt_schedule":"round_robin_zero_based"}


def audit_direct_prompt_receipts(config: dict, manifest: dict, metrics: list[dict], evaluations: list[dict]) -> dict:
    protocol = direct_prompt_protocol(config)
    trained = protocol["training_prompts"]
    require(manifest.get("training_prompts") == trained, "Direct training prompt manifest mismatch")
    require([r.get("optimizer_step") for r in metrics] == list(range(1,257)), "Incomplete optimizer receipt grid")
    if protocol["prompt_schedule"] == "round_robin_zero_based":
        expected = [trained[i % len(trained)] for i in range(256)]
        require(manifest.get("heldout_prompts") == protocol["heldout_prompts"]
                and manifest.get("prompt_schedule") == protocol["prompt_schedule"], "Direct prompt split/schedule mismatch")
        require(manifest.get("optimizer_prompt_counts") == dict(Counter(expected)), "Direct prompt exposure counts changed")
        require([r.get("training_prompt_id") for r in metrics] == expected,
                "Direct optimizer prompt schedule contains missing, reordered or held-out prompts")
    else:
        require(all(r.get("training_prompt_id", "original_open") == "original_open" for r in metrics),
                "Legacy Direct optimizer used another prompt")
    require(all(r.get("question_trained") is (r.get("prompt_id") in trained) for r in evaluations),
            "Direct evaluation training/heldout label mismatch")
    return protocol


def audit_evaluations(rows: list[dict], prompts: dict[str, str], answer: str, *, direct: bool) -> dict:
    validate_prompts(prompts)
    donor = "fixed_donor" if direct else "donor"
    expected = {(condition, prompt) for condition in ("matched", "blank", donor) for prompt in prompts}
    require(len(rows) == 15 and {(r["condition"], r["prompt_id"]) for r in rows} == expected, "Endpoint generation grid must be exactly 3 x 5")
    scores = {}
    image_hashes: dict[str, set] = {}
    for row in rows:
        require(row["query"] == prompts[row["prompt_id"]] and row.get("gold_eos_appended") is True, "Endpoint EOS/template mismatch")
        if direct:
            require(row.get("optimizer_step") == 256, "Endpoint evaluation uses a different step")
        else:
            require(row.get("gold") == answer, "Wrong question gold")
        original = row["scorer"] if direct else row
        require(0 < len(row["generated_token_ids"]) <= 32, "Raw generation token budget violated")
        recomputed = generation_diagnostics(row, answer, original["gold_token_ids"])
        for key, value in recomputed.items():
            require(original.get(key) == value, f"Unreproducible endpoint score: {key}")
        scores[(row["condition"], row["prompt_id"])] = recomputed
        image_hashes.setdefault(row["condition"], set()).add(row["image_sha256"])
    require(all(len(values) == 1 for values in image_hashes.values()), "Image changed across question variants")
    matched = {prompt: scores[("matched", prompt)] for prompt in prompts}
    return {
        "qa_pass": matched["original_open"]["strict_correct"],
        "robust_qa_pass": all(value["strict_correct"] for value in matched.values()),
        "prompt_correct": {prompt: value["strict_correct"] for prompt, value in matched.items()},
        "answer_prefix_correct": matched["original_open"]["answer_prefix_token_exact"],
        "overgeneration": matched["original_open"]["overgeneration"],
        "blank_correct": sum(scores[("blank", p)]["strict_correct"] for p in prompts),
        "donor_correct": sum(scores[(donor, p)]["strict_correct"] for p in prompts),
        "controls_denominator": 5, "image_sha256": {k: next(iter(v)) for k, v in image_hashes.items()},
    }


def planned_runs(config: dict, planned_manifest: dict | None, route: str) -> list[dict]:
    require(route in ("direct", "frozen"), "Unknown oracle route")
    specs = config["runs"] if route == "direct" else (planned_manifest or {})["runs"]
    require(bool(specs) and len({r["run_id"] for r in specs}) == len(specs), "Empty or duplicate planned run panel")
    require(all(r["run_id"] and all(c.isalnum() or c in "_.-" for c in r["run_id"]) for r in specs), "Unsafe run ID")
    return specs


def campaign_progress(root: Path, route: str, specs: list[dict]) -> dict:
    """Cheap status check; never seals from a partial count or a paused sentinel."""
    statuses, missing, failures = [], [], []
    if route == "direct":
        campaign_terminal_path = root / "terminal.json"
        campaign_terminal = read_json(campaign_terminal_path) if campaign_terminal_path.exists() else {}
        if campaign_terminal.get("status") == "failed":
            failures.append(str(campaign_terminal_path))
        for lane in range(2):
            p = root / f"lane-{lane}" / "terminal.json"
            t = read_json(p) if p.exists() else {}
            statuses.append(t.get("status", "running_or_not_started"))
            if t.get("status") == "failed":
                failures.append(str(p))
        for index, spec in enumerate(specs):
            p = root / f"lane-{index % 2}" / "runs" / spec["run_id"] / "terminal.json"
            t = read_json(p) if p.exists() else {}
            if t.get("status") == "failed":
                failures.append(str(p))
            if t.get("status") != "completed" or t.get("optimizer_steps") != 256:
                missing.append(spec["run_id"])
        complete = (not missing and statuses == ["completed", "completed"]
                    and campaign_terminal.get("status") == "completed")
    else:
        p = root / "status.json"
        t = read_json(p) if p.exists() else {}
        statuses.append(t.get("state", "running_or_not_started"))
        if t.get("state") == "failed":
            failures.append(str(p))
        for spec in specs:
            p = root / "runs" / spec["run_id"] / "campaign_terminal.json"
            item = read_json(p) if p.exists() else {}
            if item and item.get("returncode") != 0:
                failures.append(str(p))
            if not item or item.get("returncode") != 0:
                missing.append(spec["run_id"])
        complete = not missing and t.get("state") == "completed" and t.get("stage") == "oracle_bank_complete"
    return {"route": route, "planned": len(specs), "completed": len(specs)-len(missing), "complete": complete,
            "oracle_states": statuses, "missing_run_ids": missing, "failures": failures,
            "state": "failed" if failures else "ready_for_audit" if complete else "waiting_for_oracle"}


def _event(target: dict) -> str:
    require(len(target["events"]) == 1, "This Writer protocol requires the real single-event F1 condition")
    value = target["events"][0]["event_text"]
    require(isinstance(value, str) and bool(value.strip()), "Missing real visible event text")
    return value


def _termination(contract: dict) -> None:
    require(contract.get("assistant_end_token_id") == 151645 and contract.get("assistant_end_token_text") == "<|im_end|>", "Unexpected actual Qwen assistant EOS contract")
    require(151645 in contract.get("generation_eos_token_ids", []), "Assistant terminator is not a generation stop")


def audit_direct_run(root: Path, spec: dict, lane: int, config: dict, commit: str) -> dict:
    parent = root / f"lane-{lane}"
    lane_manifest = read_json(parent / "manifest.json")
    directory = parent / "runs" / spec["run_id"]
    terminal, manifest = read_json(directory / "terminal.json"), read_json(directory / "manifest.json")
    require(lane_manifest["commit"] == commit and lane_manifest["protocol"] == config, "Direct lane source/protocol identity mismatch")
    require(spec.get("arm") == "open_gold_eos" and all(terminal.get(k) == v and manifest.get(k) == v for k,v in spec.items()), "Direct run identity/EOS arm mismatch")
    require(terminal.get("status") == "completed" and terminal.get("optimizer_steps") == 256, "Incomplete Direct trajectory")
    require(manifest.get("fresh_start") is True and manifest.get("gold_eos_appended") is True and manifest.get("loss") == SELECTION["objective"], "Historical or wrong-objective teacher")
    require(lane_manifest.get("only_trainable") == "final_latent_fp32" and lane_manifest.get("unet_forward_count") == 0, "Direct route executed U-Net or updated a different variable")
    _termination(manifest["termination_contract"])
    metrics = json_rows(directory / "metrics.jsonl")
    require([r["optimizer_step"] for r in metrics] == list(range(1,257)), "Incomplete optimizer receipt grid")
    latent_index = json_rows(directory / "latent_index.jsonl")
    checkpoint_index = json_rows(directory / "checkpoint_index.jsonl")
    require([r["optimizer_step"] for r in latent_index] == list(range(257)), "Incomplete latent grid")
    require([r["optimizer_step"] for r in checkpoint_index] == CHECKPOINTS, "Incomplete checkpoint grid")
    for row in latent_index + checkpoint_index:
        local_artifact(directory, row["path"], row["file_sha256"])
    endpoint_path = local_artifact(directory, "endpoint_raw.pt", terminal["endpoint_file_sha256"])
    payload = torch.load(endpoint_path, map_location="cpu", weights_only=True)
    end = tensor(payload["latent_fp32"])
    start = tensor(torch.load(directory / latent_index[0]["path"], map_location="cpu", weights_only=True)["latent_fp32"])
    endpoint_sha = canonical_tensor_sha256(end)
    require(endpoint_sha == terminal["endpoint_latent_sha256"] == latent_index[-1]["latent_sha256"], "Endpoint tensor disagrees with terminal trajectory")
    require(canonical_tensor_sha256(start) == manifest["initial_latent_sha256"] == latent_index[0]["latent_sha256"], "Initial tensor changed")
    rows = json_rows(directory / "generations.jsonl")
    probes = json_rows(directory / "checkpoint_generations.jsonl")
    require([r["optimizer_step"] for r in probes] == CHECKPOINTS, "Incomplete checkpoint generation grid")
    prompt_protocol = audit_direct_prompt_receipts(config, manifest, metrics, rows + probes)
    target = config["target"]
    qa = audit_evaluations(rows, target["inputs"], target["scorer_metadata"]["gold"], direct=True)
    qa["train_qa_pass"] = all(qa["prompt_correct"][p] for p in prompt_protocol["training_prompts"])
    qa["heldout_qa_pass"] = all(qa["prompt_correct"][p] for p in prompt_protocol["heldout_prompts"])
    require(qa["image_sha256"]["matched"] == canonical_tensor_sha256(payload["image"]), "Endpoint image is not evaluated image")
    controls = torch.load(parent / "reference_and_controls.pt", map_location="cpu", weights_only=True)
    source = tensor(controls["reference"])
    require(canonical_tensor_sha256(source) == lane_manifest["reference_sha256"], "Reference latent changed")
    require(canonical_tensor_sha256(controls["fixed_donor"]) == lane_manifest["donor_sha256"] == qa["image_sha256"]["fixed_donor"], "Donor RGB changed")
    require(canonical_tensor_sha256(controls["blank"]) == qa["image_sha256"]["blank"], "Blank RGB changed")
    target_data = lane_manifest["target_from_data"]
    require(target_data["segment_id"] == target["segment_id"], "Target dataset identity mismatch")
    return dict(spec=spec, question_id=target["segment_id"], answer=target["scorer_metadata"]["gold"],
                event_text=_event(target_data), question_variants=target["inputs"], endpoint=end, initial=start, source=source,
                donor={"kind":"image", "value":controls["fixed_donor"].float(), "answer":"orange"},
                qa=qa, rows=rows, source_run=str(directory.resolve()), termination=manifest["termination_contract"],
                models=lane_manifest["snapshots_start"], data=lane_manifest["protocol"]["data_sha256"],
                provenance={"prompt_protocol":prompt_protocol, "metrics_sha256":file_sha256(directory/"metrics.jsonl"),
                            "endpoint_file_sha256":file_sha256(endpoint_path), "endpoint_tensor_sha256":endpoint_sha,
                            "manifest_sha256":file_sha256(directory/"manifest.json"), "terminal_sha256":file_sha256(directory/"terminal.json"),
                            "generation_file_sha256":file_sha256(directory/"generations.jsonl"),
                            "latent_index_sha256":file_sha256(directory/"latent_index.jsonl"), "checkpoint_index_sha256":file_sha256(directory/"checkpoint_index.jsonl")})


def audit_frozen_run(root: Path, spec: dict, config: dict, commit: str) -> dict:
    directory = root / "runs" / spec["run_id"]
    terminal, summary, manifest = (read_json(directory / p) for p in ("campaign_terminal.json", "summary.json", "manifest.json"))
    require(terminal.get("returncode") == 0 and terminal.get("summary_sha256") == file_sha256(directory / "summary.json"), "Frozen summary completion/SHA mismatch")
    require(summary.get("manifest_sha256") == file_sha256(directory / "manifest.json"), "Frozen manifest SHA mismatch")
    require(manifest.get("git_commit") == commit and manifest.get("git_dirty") is False and manifest.get("run_spec") == spec and manifest.get("config") == config, "Frozen source/protocol/run identity mismatch")
    require(manifest.get("mode") == "optimize" and summary.get("status") == "completed" and summary.get("optimizer_steps") == 256, "Not a completed optimization")
    require(summary.get("technical_pass") and summary.get("model_snapshot_end_verified"), "Missing Frozen end audit")
    require(summary.get("training_objective") == SELECTION["objective"] and summary.get("gold_eos_appended") is True and summary.get("choice_forward_calls") == 0, "Historical or MCQ Frozen run is forbidden")
    require(manifest["contract"]["only_trainable"] == "x_T_fp32", "Frozen route optimized a different variable")
    _termination(manifest["termination_contract"])
    metrics = json_rows(directory / "metrics.jsonl")
    require([r["optimizer_step"] for r in metrics] == list(range(1,257)), "Missing Frozen optimizer receipts")
    require(all(r.get("gold_eos_appended") is True and r.get("prompt_id") == "original_open" and r.get("lambda_eos") == 1.0 for r in metrics), "Mixed objective within Frozen trajectory")
    index_path = local_artifact(directory, summary["trajectory_index_path"])
    checkpoint_path = local_artifact(directory, summary["checkpoint_index_path"])
    index, checkpoints = read_json(index_path), read_json(checkpoint_path)
    require([r["step"] for r in index] == list(range(257)), "Incomplete Frozen trajectory grid")
    require([r["step"] for r in checkpoints] == CHECKPOINTS, "Incomplete Frozen checkpoint grid")
    for row in index:
        for space in ("xT", "z"):
            local_artifact(directory, row[space]["path"], row[space]["file_sha256"])
    for row in checkpoints:
        local_artifact(directory, row["path"], row["file_sha256"])
        local_artifact(directory, row["png_path"], row["png_sha256"])
    endpoint_path = local_artifact(directory, summary["endpoint_z_path"], summary["endpoint_z_file_sha256"])
    end = tensor(torch.from_numpy(np.load(endpoint_path, allow_pickle=False)))
    start = tensor(torch.from_numpy(np.load(local_artifact(directory, index[0]["z"]["path"]), allow_pickle=False)))
    require(canonical_tensor_sha256(end) == summary["endpoint_z_tensor_sha256"] == index[-1]["z"]["sha256"], "Frozen final z tensor changed")
    require(canonical_tensor_sha256(start) == index[0]["z"]["sha256"], "Frozen initial final-z tensor changed")
    rows = read_json(directory / "evaluation.json")
    require(rows == summary["evaluation_rows"], "Frozen evaluation and summary disagree")
    qa = audit_evaluations(rows, manifest["open_prompts"], manifest["gold"], direct=False)
    require(qa["qa_pass"] == summary["qa_pass"], "Frozen success gate differs from raw generation")
    endpoint_checkpoint = torch.load(local_artifact(directory, checkpoints[-1]["path"]), map_location="cpu", weights_only=True)
    require(canonical_tensor_sha256(endpoint_checkpoint["rgb"]) == qa["image_sha256"]["matched"], "Frozen endpoint RGB not evaluated")
    require(canonical_tensor_sha256(endpoint_checkpoint["z"]) == canonical_tensor_sha256(end), "Frozen endpoint checkpoint z mismatch")
    source_record = manifest["condition"]["source_latent"]
    source = tensor(torch.from_numpy(np.load(local_artifact(directory, source_record["path"], source_record["file_sha256"]), allow_pickle=False)))
    require(canonical_tensor_sha256(source) == source_record["sha256"], "Frozen source tensor changed")
    donor_binding = manifest["donor_binding"]
    donor_path = Path(donor_binding["path"])
    require(file_sha256(donor_path) == donor_binding["sha256"] and donor_binding["gold"] != manifest["gold"], "Donor binding invalid")
    donor = tensor(torch.from_numpy(np.load(donor_path, allow_pickle=False)))
    return dict(spec=spec, question_id=manifest["task_id"], answer=manifest["gold"], event_text=_event(manifest["target_segment"]),
                question_variants=manifest["open_prompts"], endpoint=end, initial=start, source=source,
                donor={"kind":"latent", "value":donor, "answer":donor_binding["gold"]}, qa=qa, rows=rows,
                source_run=str(directory.resolve()), termination=manifest["termination_contract"], models=manifest["models"], data=manifest["data"],
                provenance={"endpoint_file_sha256":file_sha256(endpoint_path), "endpoint_tensor_sha256":canonical_tensor_sha256(end),
                            "manifest_sha256":file_sha256(directory/"manifest.json"), "terminal_sha256":file_sha256(directory/"campaign_terminal.json"),
                            "generation_file_sha256":file_sha256(directory/"evaluation.json"),
                            "trajectory_index_sha256":file_sha256(index_path), "checkpoint_index_sha256":file_sha256(checkpoint_path)})


def geometry_description(values: np.ndarray, *, seed: int = 20260908) -> dict:
    """Descriptive sample geometry; no inferred cluster count or manifold claim."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and bool(np.isfinite(values).all()), "Invalid geometry matrix")
    n, dimension = values.shape
    result = {"samples": n, "ambient_dimensions": dimension, "sample_rank_cap": min(max(n-1,0), dimension),
              "cluster_count": None, "cluster_count_reason": "No cluster count is imposed from a small success-selected sample"}
    if n == 0:
        return result
    centered = values - values.mean(axis=0, keepdims=True)
    def pairwise(matrix):
        gram = matrix @ matrix.T
        distance = np.sqrt(np.maximum(np.diag(gram)[:,None]+np.diag(gram)[None,:]-2*gram, 0))
        if n > 1:
            np.fill_diagonal(distance, np.inf)
            nearest = distance.min(axis=1)
            triangle = distance[np.triu_indices(n, 1)]
            return {"min":float(triangle.min()), "median":float(np.median(triangle)), "max":float(triangle.max()),
                    "nearest_neighbor_median":float(np.median(nearest))}
        return None
    def spectral(matrix):
        eig = np.maximum(np.linalg.eigvalsh(matrix @ matrix.T)[::-1], 0)
        total = eig.sum()
        fraction = eig/total if total else eig
        return {"eigenvalues":eig.tolist(), "explained_fraction":fraction.tolist(),
                "r95":int(np.searchsorted(np.cumsum(fraction), .95)+1) if total else 0,
                "participation_ratio":float(total*total / (eig @ eig)) if eig @ eig else 0.0}
    # Centering all observations by the same mean leaves pairwise distances unchanged.
    result.update(raw_pairwise_l2=pairwise(values), centered_pairwise_l2=pairwise(centered),
                  raw_norms=np.linalg.norm(values,axis=1).tolist(), centered_norms=np.linalg.norm(centered,axis=1).tolist(),
                  raw_spectrum=spectral(values), centered_pca=spectral(centered))
    isotropic=np.random.default_rng(seed).standard_normal(values.shape)
    isotropic-=isotropic.mean(axis=0,keepdims=True)
    result["matched_n_d_isotropic_centered_pca"] = spectral(isotropic)
    result["interpretation_limits"] = ["rank <= N-1 is a sample-size fact, not evidence for a low-dimensional manifold",
        "successful optimization endpoints are a selected empirical set, not the entire readable region",
        "common centering preserves pairwise distance; raw versus centered cosine/norm and spectra differ",
        "no untested centroid replaces a successful endpoint"]
    return result


def _save_tensor(path: Path, value: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = value.detach().cpu().contiguous()
    torch.save(value, path)
    return {"path":str(path.resolve()), "file_sha256":file_sha256(path), "sha256":canonical_tensor_sha256(value), "shape":list(value.shape)}


def export_bank(audited: list[dict], out: Path, *, route: str, provenance: dict) -> dict:
    """Export all and only primary-success endpoints, keeping provenance on duplicates."""
    out.mkdir(parents=True, exist_ok=True)
    require(bool(audited), "No audited planned runs")
    group_map: dict[str, dict] = {}
    teachers: dict[tuple[str,str], dict] = {}
    for row in audited:
        question_id = row["question_id"]
        if question_id not in group_map:
            source = _save_tensor(out / "conditions" / f"{question_id}-source.pt", row["source"])
            donor = _save_tensor(out / "controls" / f"{question_id}-donor.pt", row["donor"]["value"])
            kind = row["donor"]["kind"]
            group_map[question_id] = dict(question_id=question_id, event_text=row["event_text"], answer=row["answer"],
                question_variants=row["question_variants"], teacher_ids=[], source_kind="blank_gray_1024",
                source_latent_path=source["path"],source_latent_file_sha256=source["file_sha256"],source_latent_sha256=source["sha256"],
                donor_control={f"{kind}_path":donor["path"], f"{kind}_file_sha256":donor["file_sha256"],f"{kind}_sha256":donor["sha256"],"answer":row["donor"]["answer"]},
                termination_contract=row["termination"], planned_count=0, successful_run_count=0)
        group = group_map[question_id]
        require(group["event_text"] == row["event_text"] and group["answer"] == row["answer"] and group["question_variants"] == row["question_variants"], "Inconsistent conditions within one question")
        require(group["source_latent_sha256"] == canonical_tensor_sha256(row["source"]), "Different source frames within one question")
        group["planned_count"] += 1
        if not row["qa"]["qa_pass"]:
            continue
        group["successful_run_count"] += 1
        key = (question_id, canonical_tensor_sha256(row["endpoint"]))
        record = {"run_id":row["spec"]["run_id"], "run_spec":row["spec"], "source_run":row["source_run"],
                  "qa":row["qa"], "artifact_bindings":row["provenance"]}
        if key in teachers:
            teachers[key]["source_runs"].append(record)
            continue
        teacher_id=f"{question_id}-{key[1][:16]}"
        latent = _save_tensor(out / "latents" / f"{teacher_id}.pt",row["endpoint"])
        teachers[key] = dict(teacher_id=teacher_id, question_id=question_id, answer=row["answer"],
            endpoint_step=256, latent_path=latent["path"],latent_file_sha256=latent["file_sha256"],latent_sha256=latent["sha256"],shape=latent["shape"],
            source_run=row["source_run"], source_runs=[record], primary_success=True, robust_qa_pass=row["qa"]["robust_qa_pass"],
            target={"strict_correct":True,"prompt_id":"original_open","answer":row["answer"]},
            gold_eos_appended=True,evaluation_generation={"do_sample":False,"max_new_tokens":32},
            question_variants=row["question_variants"],qa=row["qa"])
        group["teacher_ids"].append(teacher_id)
    records=[dict(spec=r["spec"],question_id=r["question_id"],source_run=r["source_run"],qa=r["qa"],provenance=r["provenance"],evaluation_rows=r["rows"]) for r in audited]
    write_json(out / "oracle_audit.json", records)
    geometry = {}
    for question_id, group in group_map.items():
        same = [r for r in audited if r["question_id"] == question_id]
        unique = list({canonical_tensor_sha256(r["endpoint"]):r for r in same}.values())
        good = [r for r in unique if r["qa"]["qa_pass"]]
        matrix = np.asarray([r["endpoint"].numpy().reshape(-1) for r in unique],dtype=np.float32)
        initial = np.asarray([r["initial"].numpy().reshape(-1) for r in unique],dtype=np.float32)
        success=np.asarray([r["qa"]["qa_pass"] for r in unique], dtype=bool)
        geometry[question_id]={"all_unique_endpoints":geometry_description(matrix), "successful_unique_endpoints":geometry_description(matrix[success]),
            "successful_displacements":geometry_description((matrix-initial)[success]),"successful_initials":geometry_description(initial[success]),
            "unique_success_count":len(good),"planned_including_repeat_count":len(same)}
        np.savez_compressed(out / f"{question_id}-geometry.npz", endpoint=matrix, initial=initial, success=success,
                            run_ids=np.asarray([r["spec"]["run_id"] for r in unique]))
    write_json(out / "geometry.json",geometry)
    plot_geometry(out, geometry)
    usable=[group for group in group_map.values() if group["teacher_ids"]]
    excluded=[group for group in group_map.values() if not group["teacher_ids"]]
    status="sealed" if teachers else "blocked_no_success"
    result={"schema":SCHEMA,"route":route,"status":status,"bank_status":status,"selection":SELECTION,
            "oracle_complete":True,"planned_count":len(audited),"audited_count":len(audited),
            "successful_run_count":sum(r["qa"]["qa_pass"] for r in audited),"unique_teacher_count":len(teachers),
            "groups":usable,"excluded_groups_without_success":excluded,"teachers":list(teachers.values()),
            "planned_question_ids":list(group_map),"successful_question_ids":[g["question_id"] for g in usable],
            "excluded_question_ids":[g["question_id"] for g in excluded],
            "question_coverage":{"successful":len(usable),"planned":len(group_map),"fraction":len(usable)/len(group_map)},
            "models":audited[0]["models"],"snapshots":audited[0]["models"],"data":audited[0]["data"],
            "provenance":provenance,"oracle_audit_sha256":file_sha256(out/"oracle_audit.json"),"geometry_sha256":file_sha256(out/"geometry.json"),
            "unet_training_started":False,"shared_writer_success":False,
            "limitations":["Questions with no successful endpoint are explicitly excluded from Writer training, never counted as learned",
                           "Single-question Direct bank supports in-question set learning only; no held-out-question generalization claim"],
            "reason":None if teachers else "No step-256 endpoint passed original_open raw exact match; no valid U-Net target exists"}
    write_json(out/"manifest.json",result)
    write_json(out/"seal.json",{"status":status,"manifest_sha256":file_sha256(out/"manifest.json"),"oracle_complete":True})
    return result


def plot_geometry(out: Path, geometry: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for question_id, payload in geometry.items():
        arrays=np.load(out/f"{question_id}-geometry.npz",allow_pickle=False)
        values=arrays["endpoint"].astype(np.float64)
        centered=values-values.mean(0,keepdims=True)
        gram=centered@centered.T
        eigen,vectors=np.linalg.eigh(gram)
        order=np.argsort(eigen)[::-1]
        xy=np.zeros((len(values),2))
        take=min(2,len(values))
        xy[:,:take]=vectors[:,order[:take]]*np.sqrt(np.maximum(eigen[order[:take]],0))
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        good=arrays["success"]
        axes[0].scatter(xy[~good,0],xy[~good,1],label="EM failed",marker="x",color="#a7a7a7")
        axes[0].scatter(xy[good,0],xy[good,1],label="EM passed",color="#1768ac")
        axes[0].set(xlabel="PC1",ylabel="PC2",title="All unique endpoints (projection only)")
        axes[0].legend()
        diag=np.diag(values@values.T)
        distance=np.sqrt(np.maximum(diag[:,None]+diag[None,:]-2*(values@values.T),0))
        fig.colorbar(axes[1].imshow(distance,aspect="auto",cmap="viridis"),ax=axes[1],label="Raw L2")
        axes[1].set(title="Pairwise endpoint distance",xlabel="Observed endpoint",ylabel="Observed endpoint")
        good_geo=payload["successful_unique_endpoints"]
        if good_geo["samples"]:
            for key,label in (("centered_pca","Successful endpoints"),("matched_n_d_isotropic_centered_pca","Same N,D isotropic reference")):
                vals=good_geo[key]["explained_fraction"]
                axes[2].plot(np.arange(1,len(vals)+1),np.cumsum(vals),label=label)
            axes[2].legend(fontsize=8)
        axes[2].set(xlabel="Sample component",ylabel="Cumulative variance",ylim=(0,1.04),title=f"PCA sample rank <= {good_geo['sample_rank_cap']}")
        fig.suptitle(question_id+" | no inferred cluster count or manifold claim",fontsize=10)
        fig.savefig(out/f"{question_id}-geometry.png",dpi=140)
        plt.close(fig)


def build_bank(*, route: str, oracle_root: Path, oracle_repo: Path, oracle_config: Path,
               oracle_commit: str, output_dir: Path, planned_manifest: Path | None=None, progress_only: bool=False) -> dict:
    config=read_json(oracle_config)
    planned=read_json(planned_manifest) if planned_manifest else None
    specs=planned_runs(config,planned,route)
    progress=campaign_progress(oracle_root,route,specs)
    write_json(output_dir/"progress.json",progress)
    if not progress["complete"] or progress_only:
        return progress
    if (output_dir/"seal.json").exists():
        seal=read_json(output_dir/"seal.json")
        require(seal["manifest_sha256"] == file_sha256(output_dir/"manifest.json"), "Previously sealed bank manifest changed")
        previous=read_json(output_dir/"manifest.json")
        require(previous["provenance"]["oracle_commit"] == oracle_commit and previous["provenance"]["oracle_root"] == str(oracle_root.resolve()), "Bank belongs to another campaign")
        require(previous["provenance"]["config_sha256"] == file_sha256(oracle_config), "Sealed bank configuration changed")
        if planned_manifest:
            require(previous["provenance"]["planned_manifest_sha256"] == file_sha256(planned_manifest), "Sealed bank planned manifest changed")
        for teacher in previous["teachers"]:
            local_artifact(output_dir,teacher["latent_path"],teacher["latent_file_sha256"])
        return previous
    if route == "direct":
        direct_prompt_protocol(config)
        for lane in range(2):
            parent=oracle_root/f"lane-{lane}"
            manifest=read_json(parent/"manifest.json")
            identity=read_json(parent/"identity.json")
            terminal=read_json(parent/"terminal.json")
            require(identity["commit"] == oracle_commit and identity["config_sha256"] == file_sha256(oracle_config),"Direct identity mismatch")
            require(terminal["completed"] == len(specs[lane::2]) and terminal["source_end_verified"] and terminal["snapshots_end_verified"],"Direct lane end audit absent")
            source_audit(oracle_repo,oracle_commit,manifest["source_hashes"])
        audited=[audit_direct_run(oracle_root,spec,i%2,config,oracle_commit) for i,spec in enumerate(specs)]
    else:
        identity=read_json(oracle_root/"identity.json")
        require(identity["commit"] == oracle_commit and identity["config_sha256"] == file_sha256(oracle_config) and identity["manifest_sha256"] == file_sha256(planned_manifest),"Frozen campaign identity mismatch")
        audited=[]
        checked_sources=None
        for spec in specs:
            manifest=read_json(oracle_root/"runs"/spec["run_id"]/"manifest.json")
            if checked_sources is None:
                source_audit(oracle_repo,oracle_commit,manifest["source_files_sha256"])
                checked_sources=manifest["source_files_sha256"]
            else:
                require(checked_sources == manifest["source_files_sha256"], "Mixed Frozen source snapshots")
            audited.append(audit_frozen_run(oracle_root,spec,config,oracle_commit))
    require(all(r["models"] == audited[0]["models"] and r["data"] == audited[0]["data"] for r in audited),"Mixed model/data provenance within route")
    provenance={"oracle_root":str(oracle_root.resolve()),"oracle_repo":str(oracle_repo.resolve()),"oracle_commit":oracle_commit,
                "config_path":str(oracle_config.resolve()),"config_sha256":file_sha256(oracle_config),
                "planned_manifest_sha256":file_sha256(planned_manifest) if planned_manifest else None,
                "run_ids":[r["run_id"] for r in specs],"fresh_EOS_only":True,"selection":SELECTION}
    if route == "direct":
        provenance["oracle_prompt_protocol"] = direct_prompt_protocol(config)
    return export_bank(audited,output_dir,route=route,provenance=provenance)
