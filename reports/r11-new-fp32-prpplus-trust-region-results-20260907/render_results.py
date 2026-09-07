"""Verify, summarize, and render the delivered FP32 PRP+ trust-region run."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from vision_memory.training import r11_new_fp32_prpplus_trust_region as core  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as parent_core  # noqa: E402


ARCHIVE_NAME = "r11-new-prpplus-de26e3a-20260907-round01-full.tar.gz"
ARCHIVE_BYTES = 384_333_897
ARCHIVE_SHA256 = "e6abd466e26302a052ce43df1f7a3c326c8620aa76151df4c38540bc1eb68015"
PREFLIGHT = "r11-new-prpplus-de26e3a-20260907-round01-preflight"
FORMAL = "r11-new-prpplus-de26e3a-20260907-round01-formal"
EXTERNAL_AUDIT = "r11-new-prpplus-de26e3a-20260907-round01-parent-audit.json"
EXTERNAL_AUDIT_SHA256 = "e43335d1536510674dcb402f11b330407e29cbd429128f7f27a7b4ac39a63695"
LAUNCH_LOG = "r11-new-prpplus-de26e3a-20260907-round01-formal.launch.log"
PARENT_ARCHIVE = (
    REPO
    / "reports/r11-new-fp32-trust-region-horizon128-results-20260907/raw"
    / "r11-new-trust-h128-8efeabc-20260907-round01.tar.gz"
)
PARENT_ARCHIVE_BYTES = 616_876_521
PARENT_ARCHIVE_SHA256 = "d907382f050eb7e5a452fe8632f079c746f2d6cd1f9519a667c95dae9e740483"
PARENT_PREFLIGHT = "r11-new-trust-h128-8efeabc-20260907-round01-preflight"
PARENT_FORMAL = "r11-new-trust-h128-8efeabc-20260907-round01-formal"
PARENT_EXTERNAL_AUDIT = "r11-new-trust-h128-8efeabc-20260907-parent-prefix-audit.json"

PART_SPECS = (
    ("00", 47_185_920, "6507a8324d27b6e73b96b05d9aab53f72f1a4851c8516a1168f5fd6a4caba8e4"),
    ("01", 47_185_920, "9d7a6d66ce6dbddf54e3fd071f11f1d4db753a3b8b39620a4fb2dd2b6cfcc469"),
    ("02", 47_185_920, "9c12c0d3999b33beda460f535eb1a110ee2e6425489aa47369ce948f7f0d4a4d"),
    ("03", 47_185_920, "b310b758332c305ed978b6e5be9267f631d6a53c44bcb4880fa8cbfcdcecdd2d"),
    ("04", 47_185_920, "f49256fb091fb2914779748811dec1c02bca5fbeade5860a5476394d69f4452c"),
    ("05", 47_185_920, "7dbdc7d3cc02275c50f01426f434536ee0bd0ea501a2a1d57401463e8733a35f"),
    ("06", 47_185_920, "3c89a1da54d60a0969166180df190b3d7a8414f15d5f658b5ff3e8ead6b79271"),
    ("07", 47_185_920, "2348294f4c32ec9c77b3c6bc413311e4e576b8348fe048d9c83c20dc48d5c370"),
    ("08", 6_846_537, "5d23abd0ae36c122d06828c61c885470ce21d1f51e09fc39c5da851eca3d0271"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def part_path(index: str) -> Path:
    return ROOT / "raw" / f"{ARCHIVE_NAME}.part-{index}"


def reconstruct_archive(destination: Path) -> list[dict]:
    expected_paths = [part_path(index) for index, _, _ in PART_SPECS]
    observed_paths = sorted((ROOT / "raw").glob(f"{ARCHIVE_NAME}.part-*"))
    if observed_paths != expected_paths:
        raise ValueError("Raw archive part set is incomplete or contains unexpected files.")
    records = []
    complete_hash = hashlib.sha256()
    complete_bytes = 0
    with destination.open("xb") as output:
        for (index, expected_bytes, expected_hash), path in zip(PART_SPECS, expected_paths, strict=True):
            observed_hash = sha256_file(path)
            if path.stat().st_size != expected_bytes or observed_hash != expected_hash:
                raise ValueError(f"Archive part {index} failed its bound size or SHA-256.")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(chunk)
                    complete_hash.update(chunk)
                    complete_bytes += len(chunk)
            records.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": expected_bytes,
                    "sha256": expected_hash,
                }
            )
    if complete_bytes != ARCHIVE_BYTES or complete_hash.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Parts do not reconstruct the exact remote archive.")
    return records


def validate_members(bundle: tarfile.TarFile, allowed_roots: tuple[str, ...]) -> None:
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or path.parts[0] not in allowed_roots
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise ValueError(f"Unsafe archive member: {member.name}")


def verify_inventory(root: Path) -> dict:
    inventory_path = root / "artifact_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    listed = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        path = PurePosixPath(relative)
        if relative in listed or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe inventory path: {relative}")
        artifact = root.joinpath(*path.parts)
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != item["bytes"]
            or sha256_file(artifact) != item["sha256"]
        ):
            raise ValueError(f"Inventory mismatch: {root.name}/{relative}")
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != inventory_path
    }
    if observed != listed or inventory["artifact_count"] != len(listed):
        raise ValueError(f"Incomplete inventory: {root.name}")
    return {
        "passed": True,
        "artifact_count": len(listed),
        "inventory_sha256": sha256_file(inventory_path),
    }


def read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"Missing terminal newline: {path}")
    return [json.loads(line) for line in text.splitlines()]


def compare_selected(extracted: Path, selected: Path) -> dict:
    files = sorted(path for path in selected.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Selected delivery is empty: {selected}")
    for delivered in files:
        relative = delivered.relative_to(selected)
        source = extracted / relative
        if not source.is_file() or delivered.read_bytes() != source.read_bytes():
            raise ValueError(f"Selected artifact is not byte-exact: {selected.name}/{relative}")
    return {"passed": True, "file_count": len(files)}


ITERATION_FIELDS = (
    "iteration_id",
    "iteration",
    "current_loss",
    "current_loss_ratio_to_plateau",
    "current_x_T_fp32_sha256",
    "current_endpoint_fp32_sha256",
    "gradient_fp32_sha256",
    "direction_fp32_sha256",
    "gradient_norm",
    "gradient_nonzero_fraction",
    "analytic_directional_derivative",
    "selected_candidate_id",
    "selected_radius_index",
    "selected_radius_l2",
    "selected_loss",
    "selected_loss_ratio_to_plateau",
    "selected_x_T_fp32_sha256",
    "selected_endpoint_fp32_sha256",
    "selected_candidate_residual_l2",
    "relative_improvement",
    "accepted",
)
CANDIDATE_FIELDS = (
    "candidate_id",
    "iteration",
    "radius_index",
    "radius_l2",
    "candidate_x_T_fp32_sha256",
    "endpoint_fp32_sha256",
    "loss",
    "loss_ratio_to_plateau",
    "loss_ratio_to_current",
)


def common_fields(row: dict, names: tuple[str, ...]) -> dict:
    return {name: row[name] for name in names}


def audit_first_step(formal: Path, parent: Path) -> dict:
    run_iteration = read_jsonl(formal / "iteration_metrics.jsonl")[0]
    parent_iteration = read_jsonl(parent / "iteration_metrics.jsonl")[0]
    run_candidates = read_jsonl(formal / "candidate_metrics.jsonl")[: len(core.RADII)]
    parent_candidates = read_jsonl(parent / "candidate_metrics.jsonl")[: len(core.RADII)]
    iteration_exact = common_fields(run_iteration, ITERATION_FIELDS) == common_fields(
        parent_iteration, ITERATION_FIELDS
    )
    candidates_exact = all(
        common_fields(run, CANDIDATE_FIELDS) == common_fields(parent_row, CANDIDATE_FIELDS)
        for run, parent_row in zip(run_candidates, parent_candidates, strict=True)
    )
    initial_metadata = (
        run_iteration["restart_reason"] == "initial"
        and run_iteration["beta_raw"] == 0.0
        and run_iteration["beta"] == 0.0
        and run_iteration["previous_gradient_fp32_sha256"] is None
        and run_iteration["previous_raw_direction_fp32_sha256"] is None
    )
    if not (iteration_exact and candidates_exact and initial_metadata):
        raise ValueError("Local first-step parent reproduction failed.")
    return {
        "passed": True,
        "iteration_common_fields_exact": True,
        "all_five_candidate_common_fields_exact": True,
        "initial_recurrence_metadata_exact": True,
    }


def validate_external_audit(path: Path) -> dict:
    if sha256_file(path) != EXTERNAL_AUDIT_SHA256:
        raise ValueError("External parent audit SHA-256 drift.")
    value = json.loads(path.read_text(encoding="utf-8"))
    first = value["first_step"]
    run = value["run_audit"]
    if not (
        value["passed"] is True
        and value["classification_accepted"] is True
        and value["formal_picture_memory_success"] is False
        and value["phase2_allowed"] is False
        and first["iteration_common_fields_exact"] is True
        and first["all_five_candidate_common_fields_exact"] is True
        and first["initial_beta_zero"] is True
        and first["initial_direction_is_negative_gradient"] is True
        and run["passed"] is True
        and run["classification"] == "prpplus_material_improvement_only"
        and run["fixed_target_optimization_success"] is False
        and run["formal_success"] is False
        and run["phase2_allowed"] is False
    ):
        raise ValueError("External parent audit boundary drift.")
    return {
        "passed": True,
        "sha256": EXTERNAL_AUDIT_SHA256,
        "first_step": first,
    }


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_statistics(path: Path, reference: Path) -> dict:
    pixels = image_array(path)
    reference_pixels = image_array(reference)
    difference = pixels - reference_pixels
    variation = (np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()) / 2.0
    return {
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float(variation),
        "difference_from_fixed_target_png": {
            "pixel_mse": float(np.square(difference).mean()),
            "pixel_mae": float(np.abs(difference).mean()),
            "pixel_max_abs": float(np.abs(difference).max()),
        },
    }


def render_loss(prp_rows: list[dict], parent_rows: list[dict], figures: Path) -> None:
    prp_path = [1.0] + [row["selected_loss_ratio_to_plateau"] for row in prp_rows if row["accepted"]]
    parent_path = [1.0] + [row["selected_loss_ratio_to_plateau"] for row in parent_rows]
    fig, axis = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    axis.plot(range(len(parent_path)), parent_path, color="#64748B", linewidth=1.8, label="unit negative gradient")
    axis.plot(range(len(prp_path)), prp_path, color="#7C3AED", linewidth=2.3, label="PRP+")
    axis.scatter([len(prp_path) - 1], [prp_path[-1]], color="#7C3AED", s=38, zorder=4)
    axis.axhline(0.1, linestyle="--", color="#0369A1", label="strong capture = 0.1")
    axis.axhline(0.01, linestyle=":", color="#166534", label="fixed-target success = 0.01")
    axis.axhline(parent_path[-1], linestyle="-.", color="#475569", label="parent final ratio")
    axis.annotate(
        "next attempt rejected",
        xy=(len(prp_path) - 1, prp_path[-1]),
        xytext=(74, 0.041),
        arrowprops={"arrowstyle": "->", "color": "#7C3AED"},
        color="#5B21B6",
    )
    axis.set_yscale("log")
    axis.set_xlim(0, 130)
    axis.set_xlabel("Accepted update")
    axis.set_ylabel("Endpoint MSE / FP32 plateau MSE")
    axis.set_title("PRP+ reduces the first-order long tail but does not reach the fixed target")
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(figures / "prpplus_loss_comparison.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def render_dynamics(rows: list[dict], figures: Path) -> None:
    attempts = np.arange(1, len(rows) + 1)
    radii = np.asarray([row["selected_radius_l2"] for row in rows])
    betas = np.asarray([row["beta"] for row in rows])
    cosines = np.asarray([row["negative_gradient_cosine"] for row in rows])
    improvements = np.asarray([row["relative_improvement"] for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 10.8), sharex=True, constrained_layout=True)
    axes[0].step(attempts, radii, where="mid", color="#7C3AED", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best radius")
    axes[1].plot(attempts, betas, color="#EA580C", linewidth=1.4)
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_ylabel("PRP+ beta")
    axes[2].plot(attempts, cosines, color="#0284C7", linewidth=1.5)
    axes[2].set_ylabel("cos(direction, -g)")
    axes[2].set_ylim(0, 1.04)
    axes[3].plot(attempts, improvements, color="#16A34A", linewidth=1.5)
    axes[3].axhline(0.0, color="#DC2626", linestyle="--", linewidth=1.0)
    axes[3].scatter([attempts[-1]], [improvements[-1]], color="#DC2626", s=35, zorder=4)
    axes[3].set_ylabel("Relative improvement")
    axes[3].set_xlabel("Candidate-search attempt")
    for axis in axes:
        axis.axvline(65, color="#DC2626", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.2, which="both")
    fig.suptitle("PRP+ recurrence dynamics and preregistered rejection stop")
    fig.savefig(figures / "prpplus_dynamics.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def render_matched_updates(prp_path: list[float], parent_path: list[float], figures: Path) -> None:
    updates = [8, 16, 32, 40, 64]
    parent_values = np.asarray([parent_path[index] for index in updates])
    prp_values = np.asarray([prp_path[index] for index in updates])
    x = np.arange(len(updates))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.1), constrained_layout=True)
    axes[0].bar(x - width / 2, parent_values, width, color="#94A3B8", label="unit negative gradient")
    axes[0].bar(x + width / 2, prp_values, width, color="#7C3AED", label="PRP+")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, updates)
    axes[0].set_xlabel("Accepted update")
    axes[0].set_ylabel("Loss ratio")
    axes[0].set_title("Matched-update endpoint loss")
    axes[0].grid(alpha=0.2, axis="y", which="both")
    axes[0].legend(fontsize=8)
    improvement = 100.0 * (1.0 - prp_values / parent_values)
    axes[1].bar(x, improvement, color="#16A34A")
    axes[1].axhline(0.0, color="#475569", linewidth=1.0)
    axes[1].set_xticks(x, updates)
    axes[1].set_xlabel("Accepted update")
    axes[1].set_ylabel("PRP+ lower loss than parent (%)")
    axes[1].set_title("Curvature memory advantage")
    axes[1].grid(alpha=0.2, axis="y")
    fig.suptitle("PRP+ advantage appears early and persists through the stopping point")
    fig.savefig(figures / "prpplus_matched_updates.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def render_montage(formal: Path, figures: Path) -> dict:
    selections = [
        ("Fixed BF16 target", formal / "target/fixed_parent_target.png"),
        ("FP32 teacher endpoint", formal / "target/fp32_teacher.png"),
        ("Update 1", formal / "accepted_images/iteration-00.png"),
        ("Update 8", formal / "accepted_images/iteration-07.png"),
        ("Update 16", formal / "accepted_images/iteration-15.png"),
        ("Update 24", formal / "accepted_images/iteration-23.png"),
        ("Update 32", formal / "accepted_images/iteration-31.png"),
        ("Update 40", formal / "accepted_images/iteration-39.png"),
        ("Update 48", formal / "accepted_images/iteration-47.png"),
        ("Update 56", formal / "accepted_images/iteration-55.png"),
        ("Update 64", formal / "accepted_images/iteration-63.png"),
        ("Final replay", formal / "final/final.png"),
    ]
    reference = selections[0][1]
    stats = {}
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.7), constrained_layout=True)
    for axis, (title, path) in zip(axes.flat, selections, strict=True):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        stats[title] = image_statistics(path, reference)
    fig.suptitle("Decoded image trajectory (visual similarity is not a success criterion)")
    fig.savefig(figures / "prpplus_montage.png", dpi=160, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)
    return stats


def write_csvs(rows: list[dict], prp_path: list[float], parent_path: list[float], derived: Path) -> None:
    fields = (
        "iteration",
        "accepted",
        "current_loss_ratio_to_plateau",
        "selected_loss_ratio_to_plateau",
        "selected_radius_l2",
        "relative_improvement",
        "gradient_norm",
        "beta_raw",
        "beta",
        "restart_reason",
        "negative_gradient_cosine",
    )
    with (derived / "iteration_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)
    with (derived / "matched_update_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("accepted_update", "parent_ratio", "prpplus_ratio"),
            lineterminator="\n",
        )
        writer.writeheader()
        for update in (1, 8, 16, 32, 40, 64):
            writer.writerow(
                {
                    "accepted_update": update,
                    "parent_ratio": parent_path[update],
                    "prpplus_ratio": prp_path[update],
                }
            )


def write_delivery_manifest() -> None:
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "DELIVERY_MANIFEST.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "vision_memory.r11-new-fp32-prpplus-trust-region-delivery-manifest.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise ValueError("Independent PRP+ audit requires one CPU reduction thread.")
    if (
        not PARENT_ARCHIVE.is_file()
        or PARENT_ARCHIVE.stat().st_size != PARENT_ARCHIVE_BYTES
        or sha256_file(PARENT_ARCHIVE) != PARENT_ARCHIVE_SHA256
    ):
        raise ValueError("Delivered horizon-128 parent archive is absent or has drifted.")
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vlm-r11-prpplus-delivery-") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / ARCHIVE_NAME
        archive_parts = reconstruct_archive(archive)
        if archive.stat().st_size != ARCHIVE_BYTES or sha256_file(archive) != ARCHIVE_SHA256:
            raise ValueError("Reconstructed archive identity drift.")
        run_materialized = temporary_root / "run"
        parent_materialized = temporary_root / "parent"
        run_materialized.mkdir()
        parent_materialized.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            validate_members(bundle, (PREFLIGHT, FORMAL, EXTERNAL_AUDIT, LAUNCH_LOG))
            bundle.extractall(run_materialized, filter="data")
        with tarfile.open(PARENT_ARCHIVE, "r:gz") as bundle:
            validate_members(bundle, (PARENT_PREFLIGHT, PARENT_FORMAL, PARENT_EXTERNAL_AUDIT))
            bundle.extractall(parent_materialized, filter="data")
        preflight_root = run_materialized / PREFLIGHT
        formal_root = run_materialized / FORMAL
        parent_root = parent_materialized / PARENT_FORMAL
        inventories = {
            "preflight": verify_inventory(preflight_root),
            "formal": verify_inventory(formal_root),
        }
        config = core.load_config()
        audits = {
            "preflight": core.audit_delivery(preflight_root, config),
            "formal": core.audit_delivery(formal_root, config),
        }
        parent_audit = parent_core.audit_delivery(parent_root, parent_core.load_config())
        local_first_step = audit_first_step(formal_root, parent_root)
        external_path = run_materialized / EXTERNAL_AUDIT
        external_audit = validate_external_audit(external_path)
        if sha256_file(ROOT / EXTERNAL_AUDIT) != sha256_file(external_path):
            raise ValueError("Selected external audit is not byte-exact.")
        selections = {
            "preflight": compare_selected(preflight_root, ROOT / "preflight"),
            "formal": compare_selected(formal_root, ROOT / "formal"),
        }
        result = json.loads((formal_root / "result.json").read_text(encoding="utf-8"))
        rows = read_jsonl(formal_root / "iteration_metrics.jsonl")
        candidates = read_jsonl(formal_root / "candidate_metrics.jsonl")
        parent_result = json.loads((parent_root / "result.json").read_text(encoding="utf-8"))
        parent_rows = read_jsonl(parent_root / "iteration_metrics.jsonl")
    summary = result["trace_summary"]
    if not (
        result["classification"] == "prpplus_material_improvement_only"
        and result["engineering_gate"] is True
        and result["fixed_target_optimization_success"] is False
        and result["formal_success"] is False
        and result["phase2_allowed"] is False
        and summary["iteration_count"] == 65
        and summary["accepted_update_count"] == 64
        and summary["stop_reason"] == "no_acceptable_candidate"
        and summary["all_accepted_losses_strictly_monotone"] is True
        and len(candidates) == 325
        and audits["preflight"]["passed"] is True
        and audits["formal"]["passed"] is True
        and parent_audit["classification"] == "strong_capture_reached_only"
        and parent_audit["formal_success"] is False
    ):
        raise ValueError("Unexpected PRP+ result or scientific boundary.")
    prp_path = [1.0] + [row["selected_loss_ratio_to_plateau"] for row in rows if row["accepted"]]
    parent_path = [1.0] + [row["selected_loss_ratio_to_plateau"] for row in parent_rows]
    if len(prp_path) != 65 or len(parent_path) != 129:
        raise ValueError("Accepted-path length drift.")
    final_ratio = summary["final_loss_ratio_to_plateau"]
    parent_final_ratio = parent_result["trace_summary"]["final_loss_ratio_to_plateau"]
    first_capture = next(index for index, ratio in enumerate(prp_path) if ratio <= 0.1)
    first_better_parent_final = next(index for index, ratio in enumerate(prp_path) if ratio <= parent_final_ratio)
    rejected = rows[-1]
    rejected_candidates = [row for row in candidates if row["iteration"] == rejected["iteration"]]
    if rejected["accepted"] is not False or not all(
        row["loss"] > rejected["current_loss"] for row in rejected_candidates
    ):
        raise ValueError("Final rejection evidence drift.")
    render_loss(rows, parent_rows, figures)
    render_dynamics(rows, figures)
    render_matched_updates(prp_path, parent_path, figures)
    image_stats = render_montage(ROOT / "formal", figures)
    write_csvs(rows, prp_path, parent_path, derived)
    radius_counts = Counter(str(row["selected_radius_l2"]) for row in rows)
    accepted_radius_counts = Counter(str(row["selected_radius_l2"]) for row in rows if row["accepted"])
    accepted_rows = [row for row in rows if row["accepted"]]
    payload = {
        "schema": "vision_memory.r11-new-fp32-prpplus-trust-region-derived-summary.v1",
        "archive": {
            "remote_complete_archive_bytes": ARCHIVE_BYTES,
            "remote_complete_archive_sha256": ARCHIVE_SHA256,
            "github_delivery_parts": archive_parts,
        },
        "inventories": inventories,
        "selected_artifacts": selections,
        "independent_audits": {
            "preflight": audits["preflight"],
            "formal": audits["formal"],
            "parent_horizon128": parent_audit,
            "local_first_step": local_first_step,
            "remote_external_first_step": external_audit,
        },
        "formal_result": {
            "classification": result["classification"],
            "engineering_gate": result["engineering_gate"],
            "fixed_target_optimization_success": result["fixed_target_optimization_success"],
            "formal_success": result["formal_success"],
            "phase2_allowed": result["phase2_allowed"],
            "elapsed_seconds": result["elapsed_seconds"],
            "counters": result["counters"],
            "trace_summary": summary,
        },
        "comparison_to_parent": {
            "parent_final_ratio_at_128_updates": parent_final_ratio,
            "prpplus_final_ratio_after_64_accepted_updates": final_ratio,
            "relative_loss_reduction_vs_parent_final": 1.0 - final_ratio / parent_final_ratio,
            "relative_loss_reduction_vs_parent_at_update64": 1.0 - prp_path[64] / parent_path[64],
            "first_strong_capture_update": first_capture,
            "first_update_better_than_parent_128_final": first_better_parent_final,
            "ratio_at_matched_updates": {
                str(update): {"parent": parent_path[update], "prpplus": prp_path[update]}
                for update in (1, 8, 16, 32, 40, 64)
            },
            "remaining_factor_above_fixed_target_threshold": final_ratio / 0.01,
        },
        "optimization_diagnostics": {
            "selected_radius_counts_all_attempts": dict(radius_counts),
            "selected_radius_counts_accepted": dict(accepted_radius_counts),
            "restart_counts": summary["restart_counts"],
            "positive_beta_attempts": sum(row["beta"] > 0.0 for row in rows),
            "minimum_negative_gradient_cosine": min(row["negative_gradient_cosine"] for row in rows),
            "median_negative_gradient_cosine": float(np.median([row["negative_gradient_cosine"] for row in rows])),
            "last_16_accepted_mean_relative_improvement": float(
                np.mean([row["relative_improvement"] for row in accepted_rows[-16:]])
            ),
            "rejected_attempt": rejected,
            "rejected_candidate_count": len(rejected_candidates),
            "all_rejected_candidates_worse_than_current": True,
        },
        "interpretation": {
            "supported": (
                "PRP+ history materially reduces the unit-negative-gradient long tail on this fixed oracle target: "
                "it reaches strong capture at update 16, beats the parent's 128-update endpoint by update 35, and "
                "ends 45.85% lower than that endpoint after only 64 accepted updates."
            ),
            "not_supported": (
                "PRP+ does not solve the fixed-target problem: the final ratio is 2.86 times above 0.01 and the "
                "next preregistered five-radius search has no improving candidate."
            ),
            "next_preregistered_discriminator": (
                "One limited-memory inverse-curvature (L-BFGS) direction test on the same target and unchanged "
                "precision, loss, radii, acceptance, threshold, Reader exclusion, and interpretation boundary."
            ),
        },
        "image_statistics": image_stats,
        "scientific_boundary": {
            "single_fixed_oracle_target_only": True,
            "reader_evaluated": False,
            "shared_writer_evaluated": False,
            "multi_target_evaluated": False,
            "multi_seed_evaluated": False,
            "id_ood_evaluated": False,
            "picture_memory_success": False,
            "phase2_allowed": False,
        },
    }
    (derived / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (derived / "local_first_step_audit.json").write_text(
        json.dumps(local_first_step, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (ROOT / "raw/archive_parts.json").write_text(
        json.dumps(
            {
                "schema": "vision_memory.r11-new-fp32-prpplus-trust-region-archive-parts.v1",
                "complete_archive": {"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
                "parts": archive_parts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_delivery_manifest()


if __name__ == "__main__":
    main()
