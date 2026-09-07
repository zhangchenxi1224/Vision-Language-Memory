"""Verify, summarize, and render the delivered horizon-128 trust-region run."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.experiments import audit_r11_new_fp32_trust_region_horizon128 as prefix_auditor  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as core  # noqa: E402


ARCHIVE_NAME = "r11-new-trust-h128-8efeabc-20260907-round01.tar.gz"
ARCHIVE = ROOT / "raw" / ARCHIVE_NAME
ARCHIVE_BYTES = 616_876_521
ARCHIVE_SHA256 = "d907382f050eb7e5a452fe8632f079c746f2d6cd1f9519a667c95dae9e740483"
LIGHT_ARCHIVE = ROOT / "raw/r11-new-trust-h128-8efeabc-20260907-round01-light.tar.gz"
PREFLIGHT = "r11-new-trust-h128-8efeabc-20260907-round01-preflight"
FORMAL = "r11-new-trust-h128-8efeabc-20260907-round01-formal"
EXTERNAL_AUDIT_NAME = "r11-new-trust-h128-8efeabc-20260907-parent-prefix-audit.json"
PARENT_ARCHIVE = (
    REPO / "reports/r11-new-fp32-trust-region-results-20260907/raw" / "r11-new-trust-bebeffd-20260907-round02.tar.gz"
)
PARENT_ARCHIVE_SHA256 = "e8b8630ac8177ef51ea4c812cd513ae295032db4f2b523a7805bfb7c3002397b"
PARENT_PREFLIGHT = "r11-new-trust-bebeffd-20260907-round02-preflight"
PARENT_FORMAL = "r11-new-trust-bebeffd-20260907-round02-formal"

PART_SPECS = (
    ("00", 47_185_920, "2e0c498510fd4da83893bd7dc1233652f1b62b18adec6134b4ea10268e08c562"),
    ("01", 47_185_920, "b6317e40ed5e776c60a898f86f7968930067ee0eeb08bca74fd1d84e9804a938"),
    ("02", 47_185_920, "457129db3edb628ec535425896ddf2e684a469f25d4efd4b34ce5fb0b33ec0f3"),
    ("03", 47_185_920, "3c942d78aa4b3bb20febdae1b7cc5e39a3d7a4875851b1e2739a5717ec2f54c0"),
    ("04", 47_185_920, "fb0d3062e39d6729b235740d22b162f392683f20d79caa50b7fcec2e12c0d4e7"),
    ("05", 47_185_920, "8542e230fa08abdb051d86ca0121932082b68f4a2dcf96904a648b0fcdf5b5b9"),
    ("06", 47_185_920, "52b055d46dfd7cb651ff6fb778d7ab7f7e015188a2c3c0d1d65b642ad64e380a"),
    ("07", 47_185_920, "871827481ee866290527bb7e6f895b5ba7c00f9be6ea06e280bd871f5c3e0fb9"),
    ("08", 47_185_920, "5f956770c4b859f95ed42442cf6d756195363f1ebe3e42fcbf9b0c8b7abf8f7c"),
    ("09", 47_185_920, "258a0568585db7c1d28230f5bd9f8fe90ff2e11c2f0aa922e25095dd41c3fdd2"),
    ("10", 47_185_920, "5a3f5e4b438a2fd6dcafdd73231b5ef9ac0c4903ee3d106a5cc61536f75c6c9e"),
    ("11", 47_185_920, "9c002f099ea3c34372f4fcc9a13a3ea59202cb694eb4ddada7d804dfd3fa0cc6"),
    ("12", 47_185_920, "be905bf06676055c33aa98354ddf2cac0dd0f1ac1ba8ab45fc961b03f1353b8d"),
    ("13", 3_459_561, "97ed73235ffbeedc51519accfefb9f4fedca768be4aecd0d434168091ce6533e"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def part_path(index: str) -> Path:
    return ROOT / "raw" / f"{ARCHIVE_NAME}.github-part-{index}.bin"


def verify_and_reconstruct_archive() -> list[dict]:
    expected_paths = [part_path(index) for index, _, _ in PART_SPECS]
    observed_paths = sorted((ROOT / "raw").glob(f"{ARCHIVE_NAME}.github-part-*.bin"))
    if observed_paths != expected_paths:
        raise ValueError("GitHub archive-part set is incomplete or contains unexpected files.")
    records: list[dict] = []
    complete_digest = hashlib.sha256()
    complete_bytes = 0
    temporary = ARCHIVE.with_name(f"{ARCHIVE.name}.reconstructing")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as destination:
        for (index, expected_bytes, expected_sha), path in zip(PART_SPECS, expected_paths, strict=True):
            observed_sha = sha256_file(path)
            if path.stat().st_size != expected_bytes or observed_sha != expected_sha:
                raise ValueError(f"Archive part {index} failed its bound size or SHA-256.")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    destination.write(chunk)
                    complete_digest.update(chunk)
                    complete_bytes += len(chunk)
            records.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": expected_bytes,
                    "sha256": expected_sha,
                }
            )
    if complete_bytes != ARCHIVE_BYTES or complete_digest.hexdigest() != ARCHIVE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("Archive parts do not reconstruct the exact bound raw archive.")
    temporary.replace(ARCHIVE)
    return records


def validate_members(bundle: tarfile.TarFile, allowed_roots: tuple[str, ...]) -> None:
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] not in allowed_roots
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise ValueError(f"Unsafe archive member: {member.name}")


def verify_inventory(root: Path) -> dict:
    inventory_path = root / "artifact_inventory.json"
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes.decode("utf-8"))
    listed: set[str] = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        path = PurePosixPath(relative)
        if relative in listed or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe inventory path: {relative}")
        artifact = root.joinpath(*path.parts)
        if (
            not artifact.is_file()
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
        "inventory_sha256": sha256_bytes(inventory_bytes),
    }


def materialize(bundle: tarfile.TarFile, destination: Path) -> None:
    bundle.extractall(path=destination, members=bundle.getmembers(), filter="data")


def copy_selected(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    include_names = {
        "artifact_inventory.json",
        "candidate_metrics.jsonl",
        "config.json",
        "determinism.json",
        "environment.txt",
        "iteration_metrics.jsonl",
        "launch.json",
        "manifest.json",
        "model_snapshot_verification_end.json",
        "model_snapshot_verification_start.json",
        "REPORT.md",
        "result.json",
        "runtime.json",
        "stderr.log",
        "stdout.log",
        "terminal.json",
    }
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name not in include_names and path.suffix.lower() != ".png":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def render_loss(iterations: list[dict], candidates: list[dict], figures: Path) -> None:
    steps = np.arange(len(iterations) + 1)
    selected = np.asarray([1.0] + [row["selected_loss_ratio_to_plateau"] for row in iterations])
    first_capture = next(index + 1 for index, value in enumerate(selected[1:]) if value <= 0.1)
    colors = {0.1: "#DC2626", 0.03: "#F97316", 0.01: "#CA8A04", 0.003: "#16A34A", 0.001: "#7C3AED"}
    fig, axis = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    for radius in core.RADII:
        rows = [row for row in candidates if row["radius_l2"] == radius]
        axis.scatter(
            [row["iteration"] + 1 for row in rows],
            [row["loss_ratio_to_plateau"] for row in rows],
            s=10,
            alpha=0.18,
            color=colors[radius],
            label=f"candidate r={radius:g}",
        )
    axis.plot(steps, selected, linewidth=2.1, color="#0F172A", label="accepted path")
    axis.axhline(0.1, linestyle="--", color="#0369A1", label="strong capture = 0.1")
    axis.axhline(0.01, linestyle=":", color="#166534", label="fixed-target success = 0.01")
    axis.axvline(32, linestyle="--", linewidth=1.2, color="#64748B", label="delivered parent horizon = 32")
    axis.axvline(
        first_capture, linestyle="-.", linewidth=1.2, color="#0284C7", label=f"first capture = {first_capture}"
    )
    axis.set_yscale("log")
    axis.set_xlabel("Accepted update")
    axis.set_ylabel("Endpoint MSE / FP32 plateau MSE")
    axis.set_title("Horizon-128 FP32 trust-region: strong capture reached, exact target not reached")
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=7.4, ncol=2)
    fig.savefig(figures / "trust_region_loss.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def render_dynamics(iterations: list[dict], figures: Path) -> None:
    updates = np.arange(1, len(iterations) + 1)
    radii = np.asarray([row["selected_radius_l2"] for row in iterations])
    gradient = np.asarray([row["gradient_norm"] for row in iterations])
    improvement = np.asarray([row["relative_improvement"] for row in iterations])
    residual = np.asarray([row["selected_candidate_residual_l2"] for row in iterations])
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 10.8), sharex=True, constrained_layout=True)
    axes[0].step(updates, radii, where="mid", color="#7C3AED", linewidth=1.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Selected radius")
    axes[1].plot(updates, gradient, color="#DC2626", linewidth=1.7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gradient L2")
    axes[2].plot(updates, improvement, color="#2563EB", linewidth=1.5)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Relative improvement")
    axes[3].plot(updates, residual, color="#16A34A", linewidth=1.8)
    axes[3].set_xlabel("Accepted update")
    axes[3].set_ylabel("Residual L2")
    for axis in axes:
        axis.axvline(32, linestyle="--", linewidth=1.0, color="#94A3B8")
        axis.axvline(40, linestyle=":", linewidth=1.0, color="#0284C7")
        axis.grid(alpha=0.2, which="both")
    fig.suptitle("Horizon-128 trust-region optimization dynamics")
    fig.savefig(figures / "trust_region_dynamics.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def phase_records(iterations: list[dict]) -> list[dict]:
    records = []
    previous_ratio = 1.0
    for start in range(0, 128, 32):
        rows = iterations[start : start + 32]
        end_ratio = rows[-1]["selected_loss_ratio_to_plateau"]
        counts = Counter(str(row["selected_radius_l2"]) for row in rows)
        records.append(
            {
                "updates": f"{start + 1}-{start + 32}",
                "start_ratio": previous_ratio,
                "end_ratio": end_ratio,
                "relative_reduction_within_phase": 1.0 - end_ratio / previous_ratio,
                "selected_radius_counts": dict(sorted(counts.items(), key=lambda item: float(item[0]), reverse=True)),
            }
        )
        previous_ratio = end_ratio
    return records


def render_horizon_comparison(iterations: list[dict], figures: Path) -> None:
    phases = phase_records(iterations)
    labels = ["0", "32", "64", "96", "128"]
    endpoints = [1.0] + [row["end_ratio"] for row in phases]
    colors = {0.1: "#DC2626", 0.03: "#F97316", 0.01: "#CA8A04", 0.003: "#16A34A", 0.001: "#7C3AED"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    axes[0].plot(labels, endpoints, marker="o", linewidth=2.2, color="#0F172A")
    axes[0].axhline(0.1, linestyle="--", color="#0369A1", label="strong capture")
    axes[0].axhline(0.01, linestyle=":", color="#166534", label="fixed-target success")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Accepted updates")
    axes[0].set_ylabel("Loss ratio")
    axes[0].set_title("Endpoint ratio by 32-step phase")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)
    bottoms = np.zeros(4)
    x = np.arange(4)
    for radius in core.RADII:
        values = np.asarray(
            [
                sum(row["selected_radius_l2"] == radius for row in iterations[start : start + 32])
                for start in range(0, 128, 32)
            ]
        )
        axes[1].bar(x, values, bottom=bottoms, color=colors[radius], label=f"r={radius:g}")
        bottoms += values
    axes[1].set_xticks(x, [row["updates"] for row in phases])
    axes[1].set_xlabel("Update phase")
    axes[1].set_ylabel("Selections")
    axes[1].set_title("Selected radius shifts to the smallest step")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.2, axis="y")
    fig.suptitle("Budget extension helps, then enters a first-order long tail")
    fig.savefig(
        figures / "trust_region_horizon_comparison.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_statistics(path: Path) -> dict:
    pixels = image_array(path)
    variation = (np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()) / 2.0
    return {
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float(variation),
    }


def image_difference(reference: Path, candidate: Path) -> dict:
    left = image_array(reference)
    right = image_array(candidate)
    if left.shape != right.shape:
        raise ValueError("Montage images have inconsistent shapes.")
    difference = right - left
    return {
        "pixel_mse": float(np.mean(np.square(difference))),
        "pixel_mae": float(np.mean(np.abs(difference))),
        "pixel_max_abs": float(np.max(np.abs(difference))),
    }


def render_montage(formal: Path, figures: Path) -> dict:
    selections = [
        ("Fixed BF16 target", formal / "target/fixed_parent_target.png"),
        ("FP32 teacher endpoint", formal / "target/fp32_teacher.png"),
        ("Update 1", formal / "accepted_images/iteration-00.png"),
        ("Update 32", formal / "accepted_images/iteration-31.png"),
        ("Update 48", formal / "accepted_images/iteration-47.png"),
        ("Update 64", formal / "accepted_images/iteration-63.png"),
        ("Update 96", formal / "accepted_images/iteration-95.png"),
        ("Update 128", formal / "accepted_images/iteration-127.png"),
        ("Final replay", formal / "final/final.png"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.3), constrained_layout=True)
    fixed_target = selections[0][1]
    statistics = {}
    for axis, (title, path) in zip(axes.flat, selections, strict=True):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        statistics[title] = {
            **image_statistics(path),
            "difference_from_fixed_target_png": image_difference(fixed_target, path),
        }
    fig.suptitle("Decoded images along the fixed-target latent optimization path", fontsize=12)
    fig.savefig(figures / "trust_region_montage.png", dpi=160, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)
    return statistics


def validate_remote_prefix_audit(value: dict) -> None:
    prefix = value["prefix"]
    run = value["run_audit"]
    if not (
        value["passed"] is True
        and value["classification_accepted"] is True
        and value["formal_picture_memory_success"] is False
        and value["phase2_allowed"] is False
        and prefix["iteration_rows"] == 32
        and prefix["candidate_rows"] == 160
        and prefix["iteration_rows_byte_exact"] is True
        and prefix["candidate_rows_byte_exact"] is True
        and run["classification"] == "strong_capture_reached_only"
        and run["formal_success"] is False
    ):
        raise ValueError("Remote parent-prefix audit boundary drift.")


def write_delivery_manifest() -> None:
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        relative_parts = path.relative_to(ROOT).parts
        if (
            not path.is_file()
            or path.name == "DELIVERY_MANIFEST.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or "selected" in relative_parts
            or path == ARCHIVE
            or path == LIGHT_ARCHIVE
            or path.name.endswith(".reconstructing")
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
        "schema": "vision_memory.r11-new-fp32-trust-region-horizon128-delivery-manifest.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    archive_parts = verify_and_reconstruct_archive()
    if ARCHIVE.stat().st_size != ARCHIVE_BYTES or sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Reconstructed raw archive SHA-256 mismatch.")
    if not PARENT_ARCHIVE.is_file() or sha256_file(PARENT_ARCHIVE) != PARENT_ARCHIVE_SHA256:
        raise ValueError("Delivered 32-step parent archive is absent or has drifted.")
    config = core.load_config()
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vlm-r11-h128-audit-") as temporary:
        temporary_root = Path(temporary)
        run_root = temporary_root / "run"
        parent_root = temporary_root / "parent"
        run_root.mkdir()
        parent_root.mkdir()
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            validate_members(bundle, (PREFLIGHT, FORMAL, EXTERNAL_AUDIT_NAME))
            materialize(bundle, run_root)
        with tarfile.open(PARENT_ARCHIVE, "r:gz") as parent_bundle:
            validate_members(parent_bundle, (PARENT_PREFLIGHT, PARENT_FORMAL))
            materialize(parent_bundle, parent_root)
        preflight_root = run_root / PREFLIGHT
        formal_root = run_root / FORMAL
        inventories = {
            "preflight": verify_inventory(preflight_root),
            "formal": verify_inventory(formal_root),
        }
        audits = {
            "preflight": core.audit_delivery(preflight_root, config),
            "formal": core.audit_delivery(formal_root, config),
        }
        local_prefix_audit = prefix_auditor.audit_parent_prefix(formal_root, parent_root / PARENT_FORMAL)
        external_audit_path = run_root / EXTERNAL_AUDIT_NAME
        remote_prefix_audit = json.loads(external_audit_path.read_text(encoding="utf-8"))
        validate_remote_prefix_audit(remote_prefix_audit)
        shutil.copy2(external_audit_path, ROOT / "external_parent_prefix_audit.json")
        result = json.loads((formal_root / "result.json").read_text(encoding="utf-8"))
        iterations = [
            json.loads(line)
            for line in (formal_root / "iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        candidates = [
            json.loads(line)
            for line in (formal_root / "candidate_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        copy_selected(preflight_root, ROOT / "preflight")
        copy_selected(formal_root, ROOT / "formal")
    if not (
        result["classification"] == "strong_capture_reached_only"
        and result["fixed_target_optimization_success"] is False
        and result["formal_success"] is False
        and result["phase2_allowed"] is False
        and result["trace_summary"]["accepted_update_count"] == 128
        and result["trace_summary"]["all_accepted_losses_strictly_monotone"] is True
        and audits["preflight"]["passed"] is True
        and audits["formal"]["passed"] is True
        and local_prefix_audit["passed"] is True
    ):
        raise ValueError("Unexpected horizon-128 result or audit boundary.")
    render_loss(iterations, candidates, figures)
    render_dynamics(iterations, figures)
    render_horizon_comparison(iterations, figures)
    image_stats = render_montage(ROOT / "formal", figures)
    ratios = [row["selected_loss_ratio_to_plateau"] for row in iterations]
    first_capture = next(index + 1 for index, value in enumerate(ratios) if value <= 0.1)
    late_rows = iterations[-32:]
    radius_counts = Counter(str(row["selected_radius_l2"]) for row in iterations)
    summary = {
        "schema": "vision_memory.r11-new-fp32-trust-region-horizon128-derived-summary.v1",
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "github_delivery_parts": archive_parts,
            "light_archive": (
                {
                    "path": LIGHT_ARCHIVE.relative_to(ROOT).as_posix(),
                    "bytes": LIGHT_ARCHIVE.stat().st_size,
                    "sha256": sha256_file(LIGHT_ARCHIVE),
                }
                if LIGHT_ARCHIVE.is_file()
                else None
            ),
        },
        "parent_archive": {
            "path": PARENT_ARCHIVE.relative_to(REPO).as_posix(),
            "bytes": PARENT_ARCHIVE.stat().st_size,
            "sha256": PARENT_ARCHIVE_SHA256,
        },
        "inventories": inventories,
        "local_independent_audits": audits,
        "parent_prefix_audits": {
            "remote_audit_artifact": {
                "path": "external_parent_prefix_audit.json",
                "sha256": sha256_file(ROOT / "external_parent_prefix_audit.json"),
                "passed": remote_prefix_audit["passed"],
                "prefix": remote_prefix_audit["prefix"],
            },
            "local_recomputation": local_prefix_audit,
        },
        "formal_result": {
            "classification": result["classification"],
            "engineering_gate": result["engineering_gate"],
            "fixed_target_optimization_success": result["fixed_target_optimization_success"],
            "formal_success": result["formal_success"],
            "phase2_allowed": result["phase2_allowed"],
            "counters": result["counters"],
            "elapsed_seconds": result["elapsed_seconds"],
            "trace_summary": result["trace_summary"],
        },
        "observations": {
            "relative_loss_reduction_from_plateau": 1.0 - ratios[-1],
            "loss_ratio_at_updates": {"32": ratios[31], "64": ratios[63], "96": ratios[95], "128": ratios[127]},
            "first_strong_capture_update": first_capture,
            "remaining_factor_above_fixed_target_success_threshold": ratios[-1] / 0.01,
            "post_parent_horizon_relative_loss_reduction": 1.0 - ratios[-1] / ratios[31],
            "selected_radius_counts": dict(
                sorted(radius_counts.items(), key=lambda item: float(item[0]), reverse=True)
            ),
            "phases": phase_records(iterations),
            "late_32_updates": {
                "mean_relative_improvement": float(np.mean([row["relative_improvement"] for row in late_rows])),
                "minimum_relative_improvement": min(row["relative_improvement"] for row in late_rows),
                "maximum_relative_improvement": max(row["relative_improvement"] for row in late_rows),
                "selected_radius_counts": dict(Counter(str(row["selected_radius_l2"]) for row in late_rows)),
            },
            "final_gradient_norm": iterations[-1]["gradient_norm"],
            "final_relative_improvement": iterations[-1]["relative_improvement"],
            "final_residual_l2_from_plateau": iterations[-1]["selected_candidate_residual_l2"],
            "all_gradients_dense": all(row["gradient_nonzero_fraction"] == 1.0 for row in iterations),
            "interpretation": (
                "The exact parent algorithm crosses strong capture at update 40, proving the 32-step budget was "
                "insufficient for that threshold. It remains 5.28 times above the fixed-target success threshold "
                "after 128 monotone updates, while 88/128 selections use the smallest radius and the last 32 "
                "updates improve by only about 0.28% each. The preregistered next discriminator is a curvature-aware "
                "or conjugate-direction FP32 update on the same target, not more identical first-order budget."
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
        },
    }
    (derived / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_delivery_manifest()


if __name__ == "__main__":
    main()
