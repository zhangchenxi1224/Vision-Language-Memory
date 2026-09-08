"""Evidence-gated canonical bank and frozen-DreamLite controller, schema v1.

The bank manifest is an *a priori* task inventory, not a list of successful runs.
Required top-level keys: ``schema_version=1``, ``tasks``, ``runs`` and
``checkpoint_hashes={dreamlite,vae,reader}``. Each task has ``task_id``,
``source_episode_id``, ``source_event_id`` (event identities local to an episode),
``split=train|heldout``, ``source_latent_path``, ``event_embedding_path``, and
``conditioning_provenance={input_fields:[source_latent,event],
query_answer_excluded:true}``. An optional ``event_attention_mask_path`` permits
masked pooling of event token embeddings. Task/query identifiers are metadata
only: they are never model inputs. Paths resolve against the manifest file.

Each run has ``run_id,task_id,status=complete,optimized_xT_path,endpoint_z_path,
qa_pass,reader_margin,prior_penalty,perturbation={n,passed}``. All paths and arrays
are hash-bound in the output bank. ``qa_pass`` must come from actual frozen
DreamLite -> VAE -> Reader evaluation (including choice permutations). The
caller must also supply ``functional_chain=frozen_dreamlite_vae_reader``.
The prior penalty is a preregistered proxy, not a Gaussian likelihood estimate.

The geometry decision must contain ``status=complete,bank_action=go,
canonicalization=medoid|cluster_medoids,controller_supervision=single_mse|set_mse``
and ``canonical_policy`` with explicit margin/prior/perturbation thresholds and
selection weights. A behavior-only decision is deliberately not converted into
an arbitrary coordinate MSE. Failed/missing tasks remain in the ledger.

Stage C only fits a small Controller. It does not load or unfreeze DreamLite,
VAE or Reader. Its predictions require a separate complete-chain functional
evaluation before any success claim or capacity escalation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


CHAIN = "frozen_dreamlite_vae_reader"
CAPACITY_LADDER = (
    "controller", "lora_r4", "lora_r8", "lora_r16", "lora_r32", "lora_r64",
    "expanded_lora", "partial_finetune", "full_finetune_upper_bound",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def resolve_artifact(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected a tensor-only artifact: {path}")
        value = value.detach().float().numpy()
    else:
        raise ValueError(f"Expected .npy or tensor-only .pt: {path}")
    value = np.asarray(value, dtype=np.float32)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"Empty/nonfinite artifact: {path}")
    return value


def image_latent(value: np.ndarray) -> np.ndarray:
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("Latents must have shape [C,H,W] or [1,C,H,W]")
    return value


def event_pool(embedding: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding[0]
    if embedding.ndim == 1:
        return embedding
    # [1,D] is the worker's already-pooled artifact; its optional companion
    # mask describes the original token matrix and is not pooled a second time.
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        return embedding[0]
    if embedding.ndim != 2 or mask is None:
        raise ValueError("Token event embeddings require an explicit attention mask")
    weights = np.asarray(mask, dtype=np.float64).reshape(-1)
    if len(weights) != len(embedding) or not np.isin(weights, [0, 1]).all() or weights.sum() == 0:
        raise ValueError("Invalid event attention mask")
    return ((embedding.astype(np.float64) * weights[:, None]).sum(0) / weights.sum()).astype(np.float32)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate_task_inventory(tasks: Sequence[Mapping[str, Any]], *, min_train: int = 32) -> None:
    _require(min_train >= 32, "The independent training bank cannot be reduced below 32 tasks")
    _require(32 <= len(tasks) <= 64, "Bank needs 32-64 independent tasks, not repeated seeds")
    ids = [row["task_id"] for row in tasks]
    _require(all(isinstance(task_id, str) and bool(task_id) for task_id in ids), "Task identities must be nonempty strings")
    _require(len(ids) == len(set(ids)), "Duplicate task_id in preregistered inventory")
    _require(all(row.get("split") in {"train", "heldout"} for row in tasks), "Every task needs train/heldout split")
    _require(sum(row["split"] == "train" for row in tasks) >= min_train, "Insufficient independent training tasks")
    _require(any(row["split"] == "heldout" for row in tasks), "An independent heldout group is required")
    # Turn/event ids can repeat in unrelated episodes. Split by episode, and
    # identify an event by (episode,event), never by a local turn number alone.
    split_by_group: dict[str, str] = {}
    for row in tasks:
        group = str(row.get("source_episode_id", ""))
        _require(bool(group) and bool(str(row.get("source_event_id", ""))), "Missing stable episode/event identity")
        previous = split_by_group.setdefault(group, row["split"])
        _require(previous == row["split"], f"Source leakage across split: source_episode_id={group}")
    event_keys = [(row["source_episode_id"], row["source_event_id"]) for row in tasks]
    _require(len(event_keys) == len(set(event_keys)), "Repeated source event is not an independent bank task")


def validate_geometry_decision(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(decision.get("status") == "complete", "Stage A geometry is incomplete")
    _require(decision.get("bank_action") == "go", "Stage A did not authorize a coordinate oracle bank")
    _require(decision.get("canonicalization") in {"medoid", "cluster_medoids"}, "Missing canonicalization decision")
    _require(decision.get("controller_supervision") in {"single_mse", "set_mse"}, "Behavior-only/unknown loss cannot become MSE")
    if decision["canonicalization"] == "cluster_medoids":
        _require(decision["controller_supervision"] == "set_mse", "Multiple basins require set supervision")
    policy = decision.get("canonical_policy", {})
    keys = ("min_margin", "max_prior_penalty", "min_perturbations", "min_perturbation_success",
            "prior_weight", "margin_weight", "min_passing_endpoints")
    _require(all(key in policy for key in keys), "Geometry decision lacks preregistered canonical thresholds")
    _require(all(math.isfinite(float(policy[key])) for key in keys), "Nonfinite canonical threshold")
    _require(policy["min_margin"] > 0 and policy["max_prior_penalty"] >= 0, "Invalid margin/prior thresholds")
    _require(policy["min_perturbations"] >= 1 and 0 <= policy["min_perturbation_success"] <= 1,
             "Invalid perturbation thresholds")
    _require(policy["min_passing_endpoints"] >= 2, "Canonicalization needs multiple passing endpoints")
    _require(policy["prior_weight"] >= 0 and policy["margin_weight"] >= 0, "Selection weights must be nonnegative")
    return policy


def next_capacity_stage(bank: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> str | None:
    """Capacity rises only after a valid bank and audited functional failure."""
    _require(bank.get("status") == "ready", "Capacity escalation blocked: invalid bank")
    for index, record in enumerate(history):
        _require(index < len(CAPACITY_LADDER) and record.get("stage") == CAPACITY_LADDER[index],
                 "Capacity stages cannot be skipped")
        _require(record.get("bank_sha256") == bank.get("bank_sha256"), "Capacity history uses another bank")
        _require(record.get("functional_chain") == CHAIN and record.get("functional_evaluation_complete") is True,
                 "MSE alone does not establish functional failure")
        _require(record.get("optimization_audit_pass") is True and record.get("data_audit_pass") is True,
                 "Fix optimization/data evidence before increasing capacity")
        _require(record.get("outcome") in {"pass", "fail"}, "Unresolved stage cannot authorize more capacity")
        if record["outcome"] == "pass":
            _require(index == len(history) - 1, "A successful stage cannot authorize escalation")
            return None
    return CAPACITY_LADDER[len(history)] if len(history) < len(CAPACITY_LADDER) else None


def _candidate(run: Mapping[str, Any], root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    _require(run.get("status") in {"complete", "completed"}, "run incomplete/failed")
    _require(run.get("qa_pass") is True and run.get("functional_chain") == CHAIN, "no complete-chain QA pass")
    margin, prior = float(run["reader_margin"]), float(run["prior_penalty"])
    _require(math.isfinite(margin) and margin >= policy["min_margin"], "insufficient/nonfinite margin")
    _require(math.isfinite(prior) and 0 <= prior <= policy["max_prior_penalty"], "abnormal/missing prior proxy")
    perturb = run.get("perturbation") or {}
    count, passed = int(perturb.get("n", 0)), int(perturb.get("passed", 0))
    _require(count >= policy["min_perturbations"] and 0 <= passed <= count, "missing/incomplete perturbation evaluation")
    _require(passed / count >= policy["min_perturbation_success"], "perturbation instability")
    record = dict(run)
    for key in ("optimized_xT_path", "endpoint_z_path"):
        path = resolve_artifact(root, run[key])
        value = image_latent(load_array(path))
        digest = sha256_file(path)
        expected = run.get(key.replace("_path", "_file_sha256")) or run.get(key.replace("_path", "_sha256"))
        if expected:
            _require(digest == expected, f"Artifact hash mismatch: {key}")
        record[key] = str(path)
        record[key.replace("_path", "_sha256")] = digest
        record["_" + key] = value
    return record


def select_canonical(candidates: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Choose an observed passing endpoint; never invent a mean latent label."""
    matrix = np.stack([row["_endpoint_z_path"].reshape(-1) for row in candidates]).astype(np.float64)
    # Gram distances avoid allocating [K,K,65536] for dense multi-start banks.
    norms = np.sum(matrix * matrix, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2 * matrix @ matrix.T, 0.0)
    distances = np.sqrt(squared / matrix.shape[1])
    np.fill_diagonal(distances, 0.0)
    medoid_index = int(np.argmin(distances.mean(axis=1)))
    positive = distances[np.triu_indices(len(matrix), 1)]
    scale = max(float(np.median(positive)), 1e-12)
    scores = [float(distances[index, medoid_index] / scale
                    + policy["prior_weight"] * row["prior_penalty"]
                    - policy["margin_weight"] * row["reader_margin"])
              for index, row in enumerate(candidates)]
    winner = min(range(len(scores)), key=lambda index: (scores[index], candidates[index]["run_id"]))
    return {"selected_run_id": candidates[winner]["run_id"], "medoid_run_id": candidates[medoid_index]["run_id"],
            "score": scores[winner], "distance_scale": scale,
            "score_definition": "RMSE(z,medoid)/median_pair_RMSE + prior_weight*prior_proxy - margin_weight*margin",
            "candidate_scores": [{"run_id": row["run_id"], "score": score} for row, score in zip(candidates, scores)]}


def build_bank(manifest_path: Path, decision_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    result: dict[str, Any] = {"schema_version": 1, "status": "blocked", "records": [], "ledger": [],
                              "blocking_reasons": [], "manifest_sha256": sha256_file(manifest_path),
                              "geometry_decision_sha256": sha256_file(decision_path),
                              "controller_supervision": decision.get("controller_supervision"),
                              "checkpoint_hashes": manifest.get("checkpoint_hashes", {}),
                              "formal_success": False, "failed_tasks_replaced": False}
    try:
        _require(manifest.get("schema_version") == 1, "Unknown manifest schema")
        policy = validate_geometry_decision(decision)
        result["canonical_policy"] = dict(policy)
        validate_task_inventory(manifest["tasks"], min_train=int(manifest.get("min_train_tasks", 32)))
        hashes = result["checkpoint_hashes"]
        _require(all(isinstance(hashes.get(key), str) and len(hashes[key]) == 64
                     for key in ("dreamlite", "vae", "reader")), "Missing checkpoint SHA-256 provenance")
    except (ValueError, KeyError, TypeError) as exc:
        result["blocking_reasons"].append(str(exc))
        for task in manifest.get("tasks", []):
            result["ledger"].append({"task_id": task.get("task_id"), "status": "not_selected", "reason": str(exc)})
        result["bank_sha256"] = json_sha256(result)
        write_json(output, result)
        return result
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_ids: set[str] = set()
    inventory_ids = {task["task_id"] for task in manifest["tasks"]}
    for run in manifest.get("runs", []):
        if run.get("run_id") in run_ids or run.get("task_id") not in inventory_ids:
            result["blocking_reasons"].append("Duplicate run id or run outside preregistered inventory")
            continue
        run_ids.add(run.get("run_id"))
        by_task[run["task_id"]].append(run)
    for task in manifest["tasks"]:
        ledger = {"task_id": task["task_id"], "status": "blocked", "runs": [], "reason": None}
        result["ledger"].append(ledger)
        try:
            provenance = task.get("conditioning_provenance", {})
            _require(provenance.get("input_fields") == ["source_latent", "event"]
                     and provenance.get("query_answer_excluded") is True, "Conditioning provenance is missing/unsafe")
            source_path = resolve_artifact(root, task["source_latent_path"])
            source = image_latent(load_array(source_path))
            embedding_path = resolve_artifact(root, task["event_embedding_path"])
            embedding = load_array(embedding_path)
            mask_path = resolve_artifact(root, task["event_attention_mask_path"]) if task.get("event_attention_mask_path") else None
            pooled = event_pool(embedding, load_array(mask_path) if mask_path else None)
            candidates = []
            for run in by_task[task["task_id"]]:
                try:
                    candidate = _candidate(run, root, policy)
                    _require(candidate["_optimized_xT_path"].shape == source.shape, "xT/source shape mismatch")
                    candidates.append(candidate)
                    ledger["runs"].append({"run_id": run["run_id"], "status": "eligible"})
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    ledger["runs"].append({"run_id": run["run_id"], "status": "excluded", "reason": str(exc)})
            _require(len(candidates) >= policy["min_passing_endpoints"], "Too few verified stable passing endpoints")
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                cluster = candidate.get("cluster_id") if decision["canonicalization"] == "cluster_medoids" else "all"
                _require(cluster is not None, "Cluster medoids require Stage A cluster assignments")
                groups[str(cluster)].append(candidate)
            selections = []
            for cluster, members in sorted(groups.items()):
                _require(len(members) >= 2, "Singleton cluster cannot establish a stable canonical medoid")
                selections.append({"cluster_id": cluster, **select_canonical(members, policy)})
            selected_ids = {selection["selected_run_id"] for selection in selections}
            targets = [{key: value for key, value in candidate.items() if not key.startswith("_")}
                       for candidate in candidates if candidate["run_id"] in selected_ids]
            record = {key: task[key] for key in ("task_id", "source_episode_id", "source_event_id", "split")}
            if "target_index" in task:
                record["target_index"] = task["target_index"]
            record.update(source_latent_path=str(source_path), source_latent_sha256=sha256_file(source_path),
                          event_embedding_path=str(embedding_path), event_embedding_sha256=sha256_file(embedding_path),
                          event_attention_mask_path=str(mask_path) if mask_path else None,
                          event_attention_mask_sha256=sha256_file(mask_path) if mask_path else None,
                          source_shape=list(source.shape), event_dimension=int(pooled.size),
                          conditioning_provenance=provenance, targets=targets, canonical_selections=selections)
            result["records"].append(record)
            ledger["status"] = "selected"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            ledger["reason"] = str(exc)
            result["blocking_reasons"].append(f"{task['task_id']}: {exc}")
    if len(result["records"]) == len(manifest["tasks"]) and not result["blocking_reasons"]:
        shapes = {(tuple(row["source_shape"]), row["event_dimension"]) for row in result["records"]}
        if len(shapes) == 1:
            result["status"] = "ready"
        else:
            result["blocking_reasons"].append("Controller tensor shapes differ between tasks")
    result["independent_task_count"] = len(manifest["tasks"])
    result["selected_task_count"] = len(result["records"])
    result["bank_sha256"] = json_sha256(result)
    write_json(output, result)
    return result


class FrozenOracleController(nn.Module):
    """Shared basis controller; source state and event only, no task embedding."""

    def __init__(self, latent_shape: Sequence[int], event_dim: int, *, basis_rank: int = 32, hidden: int = 128):
        super().__init__()
        channels, height, width = map(int, latent_shape)
        self.latent_shape = (channels, height, width)
        self.source_encoder = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.SiLU(),
                                            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.event_norm = nn.LayerNorm(event_dim)
        self.coefficients = nn.Sequential(nn.Linear(16 * 4 * 4 + event_dim, hidden), nn.SiLU(),
                                          nn.Linear(hidden, basis_rank))
        self.basis = nn.Parameter(torch.randn(basis_rank, channels, height, width) / math.sqrt(basis_rank))
        self.common = nn.Parameter(torch.zeros(channels, height, width))

    def forward(self, source_latent: torch.Tensor, event_embedding: torch.Tensor) -> torch.Tensor:
        conditioning = torch.cat((self.source_encoder(source_latent), self.event_norm(event_embedding)), dim=1)
        coefficients = self.coefficients(conditioning)
        return self.common[None] + torch.einsum("br,rchw->bchw", coefficients, self.basis)


def latent_supervision(predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """targets=[B,K,C,H,W]; padded targets are excluded before minimizing."""
    _require(mode in {"single_mse", "set_mse"}, "Unsupported geometry loss; do not invent an MSE decision")
    _require(mask.dtype == torch.bool and mask.shape == targets.shape[:2] and bool(mask.any(1).all()),
             "Every training task needs at least one real target")
    errors = (predicted[:, None] - targets).square().flatten(2).mean(2)
    if mode == "single_mse":
        _require(bool((mask.sum(1) == 1).all()), "single MSE cannot discard multiple canonical basins")
    return errors.masked_fill(~mask, torch.inf).min(1).values.mean()


def load_bank_tensors(bank: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(bank.get("status") == "ready", "Cannot train from a partial/failed oracle bank")
    unsigned = {key: value for key, value in bank.items() if key != "bank_sha256"}
    _require(json_sha256(unsigned) == bank.get("bank_sha256"), "Bank JSON hash mismatch")
    sources, embeddings, target_sets = [], [], []
    for record in bank["records"]:
        for key in ("source_latent", "event_embedding", "event_attention_mask"):
            path = record.get(key + "_path")
            if path:
                _require(sha256_file(path) == record[key + "_sha256"], f"Bank input changed: {key}")
        sources.append(image_latent(load_array(record["source_latent_path"])))
        mask_path = record.get("event_attention_mask_path")
        embeddings.append(event_pool(load_array(record["event_embedding_path"]), load_array(mask_path) if mask_path else None))
        targets = []
        for target in record["targets"]:
            for name in ("optimized_xT", "endpoint_z"):
                _require(sha256_file(target[name + "_path"]) == target[name + "_sha256"], "Bank target changed")
            targets.append(image_latent(load_array(target["optimized_xT_path"])))
        target_sets.append(np.stack(targets))
    max_targets = max(len(values) for values in target_sets)
    padded = np.zeros((len(sources), max_targets, *sources[0].shape), dtype=np.float32)
    mask = np.zeros((len(sources), max_targets), dtype=bool)
    for index, values in enumerate(target_sets):
        padded[index, :len(values)] = values
        mask[index, :len(values)] = True
    return tuple(torch.from_numpy(values) for values in (np.stack(sources), np.stack(embeddings), padded, mask))


def validate_overfit_evaluation(evaluation: Mapping[str, Any], bank: Mapping[str, Any]) -> None:
    _require(evaluation.get("bank_sha256") == bank["bank_sha256"], "Overfit evaluation uses another bank")
    _require(evaluation.get("phase") == "overfit" and evaluation.get("status") == "complete", "Overfit evaluation incomplete")
    _require(evaluation.get("functional_chain") == CHAIN, "Overfit must use complete DreamLite/Reader chain")
    rows = evaluation.get("rows", [])
    _require(bool(rows) and all(row.get("split") == "train" and row.get("qa_pass") is True for row in rows),
             "Small-bank overfit has not passed functional evaluation")
    _require(evaluation.get("checkpoint_hashes") == bank["checkpoint_hashes"], "Overfit checkpoint provenance mismatch")
    spec_path = Path(evaluation.get("evaluation_spec_path", ""))
    _require(spec_path.is_file() and sha256_file(spec_path) == evaluation.get("evaluation_spec_sha256"),
             "Overfit evaluation must bind the original prediction specification")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _require(spec.get("phase") == "overfit" and spec.get("bank_sha256") == bank["bank_sha256"], "Wrong overfit prediction spec")
    expected = {case["case_id"]: case for case in spec["cases"]}
    _require(len(expected) == len(rows) and {row.get("case_id") for row in rows} == set(expected),
             "Functional evaluation must cover every overfit prediction without replacement")
    allowed_tasks = {record["task_id"] for record in bank["records"] if record["split"] == "train"}
    for row in rows:
        case = expected[row["case_id"]]
        _require(case["task_id"] in allowed_tasks and row.get("task_id") == case["task_id"], "Overfit case/task mismatch")
        _require(row.get("latent_sha256") == case["latent_sha256"] == sha256_file(case["latent_path"]),
                 "Overfit evaluation did not use the saved Controller prediction")
        _require(row.get("choice_permutations_evaluated", 0) >= 4 and row.get("all_choice_permutations_pass") is True,
                 "Overfit QA needs all preregistered choice permutations")


def train_controller(*, bank_path: Path, decision_path: Path, output: Path, phase: str = "overfit",
                     overfit_evaluation: Path | None = None, steps: int = 1000, learning_rate: float = 1e-3,
                     seed: int = 0, basis_rank: int = 32, hidden: int = 128, overfit_tasks: int = 8,
                     device: str = "cpu") -> dict[str, Any]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    validate_geometry_decision(decision)
    _require(decision.get("controller_target_space") == "xT", "Controller loss must be decided from control-space xT geometry")
    _require(sha256_file(decision_path) == bank["geometry_decision_sha256"], "Training geometry decision differs from bank")
    _require(phase in {"overfit", "heldout"} and steps > 0 and learning_rate > 0, "Invalid training configuration")
    _require(1 <= overfit_tasks <= 8, "Overfit stage uses at most eight fixed training tasks")
    if phase == "heldout":
        _require(overfit_evaluation is not None, "Run and functionally validate the overfit stage first")
        validate_overfit_evaluation(json.loads(overfit_evaluation.read_text(encoding="utf-8")), bank)
    _require(not output.exists() or not any(output.iterdir()), "Output must be empty; never overwrite prior experiment")
    sources, embeddings, targets, masks = load_bank_tensors(bank)
    records = bank["records"]
    train_indices = [index for index, record in enumerate(records) if record["split"] == "train"]
    # Deterministic task selection independent of successes, CE, seed, or heldout data.
    train_indices.sort(key=lambda index: hashlib.sha256(records[index]["task_id"].encode()).hexdigest())
    if phase == "overfit":
        train_indices = train_indices[:overfit_tasks]
    evaluation_indices = train_indices if phase == "overfit" else list(range(len(records)))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = FrozenOracleController(sources.shape[1:], embeddings.shape[1], basis_rank=basis_rank, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    source_train, event_train, target_train, mask_train = [value[train_indices].to(device)
                                                         for value in (sources, embeddings, targets, masks)]
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"phase": phase, "steps": steps, "learning_rate": learning_rate, "seed": seed,
                     "basis_rank": basis_rank, "hidden": hidden, "device": device,
                     "bank_sha256": bank["bank_sha256"], "model_inputs": ["source_latent", "event_embedding"],
                     "supervision": bank["controller_supervision"], "train_task_ids": [records[i]["task_id"] for i in train_indices],
                     "heldout_used_for_optimizer_or_selection": False, "dreamlite_vae_reader_frozen": True}
    write_json(output / "training_manifest.json", configuration)
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(source_train, event_train)
            loss = latent_supervision(predictions, target_train, mask_train, bank["controller_supervision"])
            _require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            _require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                         for parameter in model.parameters()), "Nonfinite controller gradient")
            optimizer.step()
            log.write(json.dumps({"step": step + 1, "train_latent_mse_before_update": float(loss.detach().cpu())}) + "\n")
    torch.save({"state_dict": model.cpu().state_dict(), "configuration": configuration,
                "latent_shape": list(sources.shape[1:]), "event_dimension": int(embeddings.shape[1])}, output / "controller.pt")
    model.eval()
    rows, cases = [], []
    with torch.inference_mode():
        for index in evaluation_indices:
            record = records[index]
            prediction = model(sources[index:index + 1], embeddings[index:index + 1])
            errors = (prediction[:, None] - targets[index:index + 1]).square().flatten(2).mean(2)
            nearest_mse = float(errors.masked_fill(~masks[index:index + 1], torch.inf).min())
            relative = nearest_mse / max(float(targets[index][masks[index]].square().mean()), 1e-12)
            artifact = output / "predictions" / f"task-{index:04d}.npy"
            artifact.parent.mkdir(exist_ok=True)
            np.save(artifact, prediction.numpy())
            rows.append({"task_id": record["task_id"], "split": record["split"], "xT_set_mse": nearest_mse,
                         "relative_mse": relative, "functional_status": "pending", "qa_pass": None})
            cases.append({"case_id": f"controller-{phase}-{index:04d}", "task_id": record["task_id"],
                          "target_index": record.get("target_index"), "split": record["split"], "space": "xT",
                          "latent_path": str(artifact.resolve()), "latent_sha256": sha256_file(artifact),
                          "source_episode_id": record["source_episode_id"], "source_event_id": record["source_event_id"],
                          "source_latent_path": record["source_latent_path"],
                          "event_embedding_path": record["event_embedding_path"],
                          "event_attention_mask_path": record.get("event_attention_mask_path")})
    evaluation_spec = {"schema_version": 1, "phase": phase, "bank_sha256": bank["bank_sha256"],
                       "status": "pending", "required_functional_chain": CHAIN,
                       "checkpoint_hashes": bank["checkpoint_hashes"],
                       "controller_sha256": sha256_file(output / "controller.pt"),
                       "required_evaluations": ["four_choice_permutations", "reset_source", "wrong_donor", "small_perturbations"],
                       "cases": cases}
    write_json(output / "evaluation_spec.json", evaluation_spec)
    result = {"schema_version": 1, "status": "awaiting_functional_evaluation", "phase": phase,
              "formal_success": False, "capacity_escalation_allowed": False, "bank_sha256": bank["bank_sha256"],
              "train_task_count": len(train_indices), "prediction_count": len(rows), "errors": rows,
              "evaluation_spec": str((output / "evaluation_spec.json").resolve()),
              "note": "Latent MSE measures coordinate fit only; no QA/generalization success has been established."}
    write_json(output / "result.json", result)
    return result
