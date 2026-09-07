"""Verify and render the R11_new activation-precision control delivery."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from vision_memory.training import r11_new_activation_precision_control as core  # noqa: E402


ARCHIVE = ROOT / "raw/r11-new-precision-2753094-20260907-round01.tar.gz"
ARCHIVE_SHA256 = "b4a34fa3e9d3d7f40949b4b6534978e00e552f1baaf2903554dc5ee112def659"
PREFLIGHT = "r11-new-precision-2753094-20260907-round01-preflight"
FORMAL = "r11-new-precision-2753094-20260907-round01-formal"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(bundle: tarfile.TarFile) -> None:
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] not in (PREFLIGHT, FORMAL)
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


def materialize_and_audit(bundle: tarfile.TarFile, config: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="vlm-r11-precision-audit-") as temporary:
        temporary_root = Path(temporary)
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            destination = temporary_root.joinpath(*PurePosixPath(member.name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(read_member(bundle, member.name))
        return {
            "preflight": core.audit_delivery(temporary_root / PREFLIGHT, config),
            "formal": core.audit_delivery(temporary_root / FORMAL, config),
        }


def save_selected_images(bundle: tarfile.TarFile) -> None:
    image_root = ROOT / "images"
    image_root.mkdir(exist_ok=True)
    selections = {}
    for condition in core.CONDITIONS:
        for index in (0, 11, 15, 16):
            name = f"{condition}-p{index:02d}.png"
            member = f"{FORMAL}/path_images/{core.path_row_id(condition, 'teacher-outward', index)}.png"
            selections[name] = member
    for name, member in selections.items():
        (image_root / name).write_bytes(read_member(bundle, member))


def canonical_path_rows(rows: list[dict], condition: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["condition"] == condition and row["pass"] == "teacher-outward"),
        key=lambda row: core.PATH_POINT_INDICES.index(row["source_point_index"]),
    )


def render_path(rows: list[dict], figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.8, 6.0), constrained_layout=True)
    colors = {"bf16-baseline": "#DC2626", "fp32-lifted": "#2563EB"}
    labels = {"bf16-baseline": "BF16 baseline", "fp32-lifted": "FP32 lifted"}
    for condition in core.CONDITIONS:
        selected = canonical_path_rows(rows, condition)
        x = np.asarray([max(row["requested_remaining_l2"], 1e-6) for row in selected])
        y = np.asarray([max(row["loss_ratio_to_plateau"], 1e-10) for row in selected])
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            color=colors[condition],
            label=labels[condition],
        )
    axis.axhline(0.1, linestyle=":", color="#7F1D1D", label="strong capture <= 0.1")
    axis.axvline(0.001, linestyle="--", color="#475569", label="restoration width >= 1e-3")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Remaining L2 distance to teacher xT (zero displayed at 1e-6)")
    axis.set_ylabel("Endpoint MSE / condition-specific plateau MSE")
    axis.set_title("FP32 arithmetic removes the BF16 terminal cliff")
    axis.grid(alpha=0.22, which="both")
    axis.legend(fontsize=8)
    fig.savefig(
        figures / "activation_precision_path.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_gradient(rows: list[dict], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, constrained_layout=True)
    for axis, condition in zip(axes, core.CONDITIONS, strict=True):
        selected = [row for row in rows if row["condition"] == condition]
        for sign, color, label in (
            ("plus", "#2563EB", "+ unit negative gradient"),
            ("minus", "#F97316", "- unit negative gradient"),
        ):
            sign_rows = sorted(
                (row for row in selected if row["sign"] == sign),
                key=lambda row: row["radius_index"],
            )
            axis.plot(
                [row["radius_l2"] for row in sign_rows],
                [row["loss_ratio_to_plateau"] for row in sign_rows],
                marker="o",
                color=color,
                label=label,
            )
        axis.axhline(0.9, linestyle="--", color="#166534", label="actionable <= 0.9")
        axis.axhline(1.0, linestyle=":", color="#334155", label="plateau")
        axis.set_xscale("log")
        axis.set_xlabel("Signed-scan radius L2")
        axis.set_title(condition)
        axis.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Endpoint MSE / condition-specific plateau MSE")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Only FP32 yields consistent finite-difference descent")
    fig.savefig(
        figures / "activation_precision_gradient.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
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


def render_montage(figures: Path) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.0), constrained_layout=True)
    statistics = {}
    titles = {0: "teacher", 11: "delta=1e-3", 15: "delta=.05", 16: "plateau"}
    for row, condition in enumerate(core.CONDITIONS):
        for column, index in enumerate((0, 11, 15, 16)):
            name = f"{condition}-p{index:02d}.png"
            path = ROOT / "images" / name
            axes[row, column].imshow(Image.open(path).convert("RGB"))
            axes[row, column].set_title(f"{condition}\n{titles[index]}", fontsize=9)
            axes[row, column].axis("off")
            statistics[name] = image_statistics(path)
    fig.suptitle("Same xT path under BF16 and FP32 arithmetic", fontsize=13)
    fig.savefig(
        figures / "activation_precision_montage.png",
        dpi=160,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)
    return statistics


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
        "schema": "vision_memory.r11-new-activation-precision-control-delivery-manifest.v1",
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
        raise ValueError("Raw precision-control archive SHA-256 mismatch.")
    config = core.load_config()
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        validate_members(bundle)
        inventories = {
            "preflight": verify_inventory(bundle, PREFLIGHT),
            "formal": verify_inventory(bundle, FORMAL),
        }
        audits = materialize_and_audit(bundle, config)
        save_selected_images(bundle)
        result = read_json(bundle, f"{FORMAL}/result.json")
        path_rows = [
            json.loads(line)
            for line in read_member(bundle, f"{FORMAL}/path_metrics.jsonl").decode("utf-8").splitlines()
        ]
        gradient_rows = [
            json.loads(line)
            for line in read_member(bundle, f"{FORMAL}/gradient_metrics.jsonl").decode("utf-8").splitlines()
        ]
    expected_counters = {
        "full_chain_forward_calls": 54,
        "reader_forward_calls": 0,
        "backward_calls": 2,
        "optimizer_steps": 0,
    }
    if (
        result["classification"] != "fp32_restores_capture_and_gradient"
        or result["counters"] != expected_counters
        or result["formal_success"] is not False
        or result["phase2_allowed"] is not False
    ):
        raise ValueError("Unexpected precision-control formal boundary.")
    recomputed = core.summarize_experiment(path_rows, gradient_rows, config)
    if result["scan_summary"] != recomputed or not recomputed["bf16_parent_reproduced"]:
        raise ValueError("Precision summary or BF16 positive control drift.")
    render_path(path_rows, figures)
    render_gradient(gradient_rows, figures)
    image_stats = render_montage(figures)
    fp32_gradient = recomputed["conditions"]["fp32-lifted"]["gradient"]
    summary = {
        "schema": "vision_memory.r11-new-activation-precision-control-derived-summary.v1",
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": ARCHIVE_SHA256,
        },
        "inventories": inventories,
        "local_independent_audits": audits,
        "formal_result": {
            "engineering_gate": result["engineering_gate"],
            "classification": result["classification"],
            "counters": result["counters"],
            "elapsed_seconds": result["elapsed_seconds"],
            "formal_success": result["formal_success"],
            "phase2_allowed": result["phase2_allowed"],
        },
        "scan_summary": recomputed,
        "precision_lift_audit": result["conditions"]["fp32-lifted"]["precision_audit"],
        "observations": {
            "bf16_capture_width": recomputed["bf16_capture_width"],
            "fp32_capture_width": recomputed["fp32_capture_width"],
            "capture_widening_factor": recomputed["fp32_to_bf16_capture_widening_factor"],
            "fp32_best_negative_gradient_loss_ratio": fp32_gradient["best_positive_loss_ratio"],
            "fp32_best_negative_gradient_radius_l2": fp32_gradient["best_positive_radius_l2"],
            "fp32_finite_difference_descent_sign_count": fp32_gradient["finite_difference_descent_sign_count"],
            "interpretation": (
                "Losslessly lifting the same BF16-valued frozen map into FP32 arithmetic "
                "widens strong oracle capture by 1666.7x and makes the plateau negative "
                "gradient actionable. BF16 input/activation arithmetic is therefore a major "
                "causal blocker, but this oracle diagnostic is not deployable memory training."
            ),
        },
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
