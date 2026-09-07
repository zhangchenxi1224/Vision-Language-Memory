"""Verify, summarize, and render the delivered FP32 trust-region experiment."""

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
sys.path.insert(0, str(REPO / "src"))

from vision_memory.training import r11_new_fp32_trust_region as core  # noqa: E402


ARCHIVE = ROOT / "raw/r11-new-trust-bebeffd-20260907-round02.tar.gz"
ARCHIVE_SHA256 = "e8b8630ac8177ef51ea4c812cd513ae295032db4f2b523a7805bfb7c3002397b"
INTERRUPTED_ARCHIVE = ROOT / "raw/r11-new-trust-bebeffd-20260907-round01-interrupted.tar.gz"
INTERRUPTED_ARCHIVE_SHA256 = "b71aa8fc8db6c046cd8571683682d9969ac129f9ba16bb98673e9be2ab9887de"
ARCHIVE_PART_GLOB = "r11-new-trust-bebeffd-20260907-round02.tar.gz.part-*.bin"
PREFLIGHT = "r11-new-trust-bebeffd-20260907-round02-preflight"
FORMAL = "r11-new-trust-bebeffd-20260907-round02-formal"
INTERRUPTED_PREFLIGHT = "r11-new-trust-bebeffd-20260907-round01-preflight"
INTERRUPTED_FORMAL = "r11-new-trust-bebeffd-20260907-round01-formal"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive_parts() -> list[dict]:
    parts = sorted((ROOT / "raw").glob(ARCHIVE_PART_GLOB))
    if not parts:
        raise ValueError("No GitHub-safe round02 archive parts were found.")
    digest = hashlib.sha256()
    total = 0
    records = []
    for path in parts:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if total != ARCHIVE.stat().st_size or digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Round02 archive parts do not reconstruct the exact raw archive.")
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


def read_member(bundle: tarfile.TarFile, name: str) -> bytes:
    member = bundle.getmember(name)
    if not member.isfile():
        raise ValueError(f"Archive member is not a file: {name}")
    handle = bundle.extractfile(member)
    if handle is None:
        raise ValueError(f"Cannot read archive member: {name}")
    return handle.read()


def read_json(bundle: tarfile.TarFile, name: str) -> dict:
    return json.loads(read_member(bundle, name).decode("utf-8"))


def read_jsonl(bundle: tarfile.TarFile, name: str) -> list[dict]:
    return [json.loads(line) for line in read_member(bundle, name).decode("utf-8").splitlines()]


def verify_inventory(bundle: tarfile.TarFile, prefix: str) -> dict:
    inventory_bytes = read_member(bundle, f"{prefix}/artifact_inventory.json")
    inventory = json.loads(inventory_bytes.decode("utf-8"))
    listed: set[str] = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        path = PurePosixPath(relative)
        if relative in listed or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe inventory path: {relative}")
        payload = read_member(bundle, f"{prefix}/{relative}")
        if len(payload) != item["bytes"] or sha256_bytes(payload) != item["sha256"]:
            raise ValueError(f"Inventory mismatch: {prefix}/{relative}")
        listed.add(relative)
    observed = {
        member.name.removeprefix(f"{prefix}/")
        for member in bundle.getmembers()
        if member.isfile()
        and member.name.startswith(f"{prefix}/")
        and member.name != f"{prefix}/artifact_inventory.json"
    }
    if observed != listed or inventory["artifact_count"] != len(listed):
        raise ValueError(f"Incomplete inventory: {prefix}")
    return {
        "passed": True,
        "artifact_count": len(listed),
        "inventory_sha256": sha256_bytes(inventory_bytes),
    }


def materialize(bundle: tarfile.TarFile, destination: Path) -> None:
    for member in bundle.getmembers():
        if not member.isfile():
            continue
        target = destination.joinpath(*PurePosixPath(member.name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_member(bundle, member.name))


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
    fig, axis = plt.subplots(figsize=(10.8, 6.0), constrained_layout=True)
    colors = {0.1: "#DC2626", 0.03: "#F97316", 0.01: "#CA8A04", 0.003: "#16A34A", 0.001: "#7C3AED"}
    for radius in core.RADII:
        rows = [row for row in candidates if row["radius_l2"] == radius]
        axis.scatter(
            [row["iteration"] + 1 for row in rows],
            [row["loss_ratio_to_plateau"] for row in rows],
            s=14,
            alpha=0.25,
            color=colors[radius],
            label=f"candidate r={radius:g}",
        )
    axis.plot(steps, selected, marker="o", markersize=3.5, linewidth=2.2, color="#0F172A", label="accepted path")
    axis.axhline(0.1, linestyle="--", color="#0369A1", label="strong capture = 0.1")
    axis.axhline(0.01, linestyle=":", color="#166534", label="fixed-target success = 0.01")
    axis.set_yscale("log")
    axis.set_xlabel("Accepted update")
    axis.set_ylabel("Endpoint MSE / FP32 plateau MSE")
    axis.set_title("FP32 trust-region loss: monotone progress, threshold not reached")
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=7.5, ncol=2)
    fig.savefig(figures / "trust_region_loss.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def render_dynamics(iterations: list[dict], figures: Path) -> None:
    updates = np.arange(1, len(iterations) + 1)
    radii = np.asarray([row["selected_radius_l2"] for row in iterations])
    gradient = np.asarray([row["gradient_norm"] for row in iterations])
    improvement = np.asarray([row["relative_improvement"] for row in iterations])
    residual = np.asarray([row["selected_candidate_residual_l2"] for row in iterations])
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.2), sharex=True, constrained_layout=True)
    axes[0].step(updates, radii, where="mid", color="#7C3AED", linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Selected radius")
    axes[0].grid(alpha=0.2, which="both")
    axes[1].plot(updates, gradient, color="#DC2626", marker="o", markersize=2.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gradient L2 norm")
    axes[1].grid(alpha=0.2, which="both")
    axes[2].plot(updates, improvement, color="#2563EB", marker="o", markersize=2.8, label="relative improvement")
    axes[2].plot(updates, residual, color="#16A34A", linewidth=1.8, label="residual L2 from plateau")
    axes[2].set_xlabel("Accepted update")
    axes[2].set_ylabel("Value")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("Trust-region optimization dynamics")
    fig.savefig(figures / "trust_region_dynamics.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)


def image_statistics(path: Path) -> dict:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    variation = (np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()) / 2.0
    return {
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float(variation),
    }


def render_montage(formal: Path, figures: Path) -> dict:
    selections = [
        ("Fixed BF16 target", formal / "target/fixed_parent_target.png"),
        ("FP32 teacher endpoint", formal / "target/fp32_teacher.png"),
        ("Update 1", formal / "accepted_images/iteration-00.png"),
        ("Update 8", formal / "accepted_images/iteration-07.png"),
        ("Update 16", formal / "accepted_images/iteration-15.png"),
        ("Update 24", formal / "accepted_images/iteration-23.png"),
        ("Update 32", formal / "accepted_images/iteration-31.png"),
        ("Final replay", formal / "final/final.png"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.0), constrained_layout=True)
    statistics = {}
    for axis, (title, path) in zip(axes.flat, selections, strict=True):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        statistics[title] = image_statistics(path)
    fig.suptitle("Human-visible images remain nearly indistinguishable during latent optimization", fontsize=12)
    fig.savefig(figures / "trust_region_montage.png", dpi=160, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)
    return statistics


def interrupted_evidence(bundle: tarfile.TarFile, formal_rows: list[dict], formal_candidates: list[dict]) -> dict:
    names = {member.name for member in bundle.getmembers() if member.isfile()}
    rows = read_jsonl(bundle, f"{INTERRUPTED_FORMAL}/iteration_metrics.jsonl")
    candidates = read_jsonl(bundle, f"{INTERRUPTED_FORMAL}/candidate_metrics.jsonl")
    return {
        "terminal_present": f"{INTERRUPTED_FORMAL}/terminal.json" in names,
        "iteration_rows": len(rows),
        "candidate_rows": len(candidates),
        "iteration_prefix_exact_match_round02": rows == formal_rows[: len(rows)],
        "candidate_prefix_exact_match_round02": candidates == formal_candidates[: len(candidates)],
        "last_completed_iteration": rows[-1]["iteration"],
        "last_selected_loss_ratio_to_plateau": rows[-1]["selected_loss_ratio_to_plateau"],
    }


def write_delivery_manifest() -> None:
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "DELIVERY_MANIFEST.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or "selected" in path.relative_to(ROOT).parts
            or path == ARCHIVE
            or path.name == "r11-new-trust-bebeffd-20260907-round02-light.tar.gz"
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
        "schema": "vision_memory.r11-new-fp32-trust-region-delivery-manifest.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    if sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Raw round02 archive SHA-256 mismatch.")
    if sha256_file(INTERRUPTED_ARCHIVE) != INTERRUPTED_ARCHIVE_SHA256:
        raise ValueError("Interrupted round01 archive SHA-256 mismatch.")
    archive_parts = verify_archive_parts()
    config = core.load_config()
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vlm-r11-trust-audit-") as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            validate_members(bundle, (PREFLIGHT, FORMAL))
            inventories = {
                "preflight": verify_inventory(bundle, PREFLIGHT),
                "formal": verify_inventory(bundle, FORMAL),
            }
            materialize(bundle, temporary_root)
        preflight_root = temporary_root / PREFLIGHT
        formal_root = temporary_root / FORMAL
        audits = {
            "preflight": core.audit_delivery(preflight_root, config),
            "formal": core.audit_delivery(formal_root, config),
        }
        result = json.loads((formal_root / "result.json").read_text(encoding="utf-8"))
        iterations = [json.loads(line) for line in (formal_root / "iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        candidates = [json.loads(line) for line in (formal_root / "candidate_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        copy_selected(preflight_root, ROOT / "preflight")
        copy_selected(formal_root, ROOT / "formal")
    with tarfile.open(INTERRUPTED_ARCHIVE, "r:gz") as interrupted:
        validate_members(interrupted, (INTERRUPTED_PREFLIGHT, INTERRUPTED_FORMAL))
        interruption = interrupted_evidence(interrupted, iterations, candidates)
    if (
        result["classification"] != "monotone_progress_above_capture"
        or result["fixed_target_optimization_success"] is not False
        or result["formal_success"] is not False
        or result["phase2_allowed"] is not False
        or not audits["preflight"]["passed"]
        or not audits["formal"]["passed"]
    ):
        raise ValueError("Unexpected formal result boundary.")
    render_loss(iterations, candidates, figures)
    render_dynamics(iterations, figures)
    image_stats = render_montage(ROOT / "formal", figures)
    radius_counts = Counter(str(row["selected_radius_l2"]) for row in iterations)
    summary = {
        "schema": "vision_memory.r11-new-fp32-trust-region-derived-summary.v1",
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": ARCHIVE_SHA256,
            "github_delivery_parts": archive_parts,
        },
        "interrupted_archive": {
            "path": INTERRUPTED_ARCHIVE.relative_to(ROOT).as_posix(),
            "bytes": INTERRUPTED_ARCHIVE.stat().st_size,
            "sha256": INTERRUPTED_ARCHIVE_SHA256,
        },
        "inventories": inventories,
        "local_independent_audits": audits,
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
            "relative_loss_reduction": 1.0 - result["trace_summary"]["final_loss_ratio_to_plateau"],
            "selected_radius_counts": dict(sorted(radius_counts.items())),
            "final_gradient_norm": iterations[-1]["gradient_norm"],
            "final_relative_improvement": iterations[-1]["relative_improvement"],
            "final_residual_l2_from_plateau": iterations[-1]["selected_candidate_residual_l2"],
            "all_gradients_dense": all(row["gradient_nonzero_fraction"] == 1.0 for row in iterations),
            "interpretation": (
                "FP32 normalized-gradient trust-region search is stable and causally actionable: all 32 updates "
                "were accepted and endpoint MSE fell by 87.26%. The run ended only at the preregistered horizon "
                "with ratio 0.12738, above both strong-capture (0.1) and fixed-target success (0.01). The next "
                "minimal discriminator is an iteration-budget extension with the same target, map, radii, loss, "
                "and update rule; this is not yet Picture Memory or Reader success."
            ),
        },
        "interruption_evidence": interruption,
        "image_statistics": image_stats,
    }
    (derived / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_delivery_manifest()


if __name__ == "__main__":
    main()
