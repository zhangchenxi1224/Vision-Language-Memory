"""Verify and render the R11_new oracle terminal-capture delivery."""

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

from vision_memory.training import r11_new_oracle_terminal_capture as core  # noqa: E402


ARCHIVE = ROOT / "raw" / "r11-new-terminal-0262545-20260907-round01.tar.gz"
ARCHIVE_SHA256 = "9c94ff86eb2649c049a763e93660fa6393e53117ad88e0d753ec588efbc75892"
PREFLIGHT = "r11-new-terminal-0262545-20260907-round01-preflight"
FORMAL = "r11-new-terminal-0262545-20260907-round01-formal"


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
    with tempfile.TemporaryDirectory(prefix="vlm-r11-terminal-audit-") as temporary:
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


def save_selected_evidence(bundle: tarfile.TarFile) -> None:
    common = (
        "artifact_inventory.json",
        "config.json",
        "determinism.json",
        "environment.txt",
        "launch.json",
        "manifest.json",
        "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
        "REPORT.md",
        "result.json",
        "runtime.json",
        "stderr.log",
        "stdout.log",
        "terminal.json",
    )
    for label, prefix in (("preflight", PREFLIGHT), ("formal", FORMAL)):
        for relative in common:
            destination = ROOT / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(read_member(bundle, f"{prefix}/{relative}"))
    (ROOT / "formal/scan_metrics.jsonl").write_bytes(
        read_member(bundle, f"{FORMAL}/scan_metrics.jsonl")
    )
    image_members = {
        "delta-0-exact-teacher.png": f"{FORMAL}/scan_images/teacher-outward__p00.png",
        "delta-3e-5-exact-endpoint.png": f"{FORMAL}/scan_images/teacher-outward__p08.png",
        "delta-1e-4-cliff.png": f"{FORMAL}/scan_images/teacher-outward__p09.png",
        "delta-1e-3.png": f"{FORMAL}/scan_images/teacher-outward__p11.png",
        "delta-5e-2.png": f"{FORMAL}/scan_images/teacher-outward__p15.png",
        "plateau.png": f"{FORMAL}/scan_images/teacher-outward__p16.png",
    }
    image_root = ROOT / "images"
    image_root.mkdir(exist_ok=True)
    for name, member in image_members.items():
        (image_root / name).write_bytes(read_member(bundle, member))


def image_statistics(path: Path) -> dict:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    variation = (
        np.abs(np.diff(pixels, axis=0)).mean()
        + np.abs(np.diff(pixels, axis=1)).mean()
    ) / 2.0
    return {
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float(variation),
    }


def canonical_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        (row for row in rows if row["pass"] == "teacher-outward"),
        key=lambda row: row["point_index"],
    )


def render_full_path(rows: list[dict], figures: Path) -> None:
    selected = canonical_rows(rows)
    x = np.asarray([row["requested_remaining_l2"] for row in selected])
    ratio = np.asarray([row["loss_ratio_to_plateau"] for row in selected])
    equality = np.asarray([row["bf16_equal_fraction_to_teacher"] for row in selected])
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    axis.plot(x, ratio, marker="o", color="#DC2626", label="endpoint MSE / plateau MSE")
    axis.axhline(0.1, color="#991B1B", linestyle=":", label="strong capture gate")
    axis.axhline(0.9, color="#B91C1C", linestyle="--", label="meaningful capture gate")
    axis.set_xscale("symlog", linthresh=1e-8)
    axis.set_xlabel("Requested remaining L2 distance to teacher (symlog)")
    axis.set_ylabel("Endpoint loss ratio")
    axis.grid(alpha=0.2)
    second = axis.twinx()
    second.plot(x, equality, marker="s", color="#2563EB", label="BF16 code equality")
    second.set_ylabel("Fraction of BF16 coordinates equal to teacher")
    second.set_ylim(0.2, 1.02)
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, fontsize=8, loc="lower left")
    axis.set_title("Terminal capture is a microscopic quantized cell, not a smooth basin")
    fig.savefig(
        figures / "terminal_capture_full_path.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_boundary_zoom(rows: list[dict], figures: Path) -> None:
    selected = [row for row in canonical_rows(rows) if row["point_index"] <= 11]
    x = np.asarray([max(row["requested_remaining_l2"], 1e-9) for row in selected])
    ratio = np.asarray([row["loss_ratio_to_plateau"] for row in selected])
    differing = np.asarray(
        [65536 * (1.0 - row["bf16_equal_fraction_to_teacher"]) for row in selected]
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    axis.plot(x, ratio, marker="o", color="#DC2626", linewidth=2, label="endpoint loss ratio")
    axis.fill_between(x, 0, ratio, color="#FCA5A5", alpha=0.25)
    axis.axhline(0.1, color="#991B1B", linestyle=":", label="strong capture gate")
    axis.set_xscale("log")
    axis.set_xlabel("Requested remaining L2 (delta=0 displayed at 1e-9)")
    axis.set_ylabel("Endpoint MSE / plateau MSE")
    axis.grid(alpha=0.2)
    second = axis.twinx()
    second.plot(x, differing, marker="s", color="#2563EB", label="changed BF16 coordinates")
    second.set_yscale("symlog", linthresh=1.0)
    second.set_ylabel("Number of BF16 coordinates different from teacher")
    axis.annotate(
        "delta=3e-5: 3 codes differ, endpoint still bitwise exact",
        xy=(3e-5, 0.0),
        xytext=(2e-8, 0.45),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=8,
    )
    axis.annotate(
        "delta=1e-4: ~17 codes differ, loss jumps to 1.5267x",
        xy=(1e-4, 1.52669160588039),
        xytext=(4e-7, 1.15),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=8,
    )
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, fontsize=8, loc="upper left")
    axis.set_title("Resolved BF16 terminal cliff: exact through 3e-5, failed by 1e-4")
    fig.savefig(
        figures / "terminal_capture_boundary_zoom.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_montage(figures: Path) -> dict:
    panels = (
        ("delta-0-exact-teacher.png", "delta=0 / exact teacher"),
        ("delta-3e-5-exact-endpoint.png", "delta=3e-5 / exact endpoint"),
        ("delta-1e-4-cliff.png", "delta=1e-4 / 1.5267x loss"),
        ("delta-1e-3.png", "delta=1e-3 / 1.6023x loss"),
        ("delta-5e-2.png", "delta=.05 / 1.5978x loss"),
        ("plateau.png", "plateau / 1.0x loss"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    statistics = {}
    for axis, (name, title) in zip(axes.flat, panels, strict=True):
        path = ROOT / "images" / name
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        statistics[name] = image_statistics(path)
    fig.suptitle("Terminal-capture changes are not human-readable", fontsize=13)
    fig.savefig(
        figures / "terminal_capture_montage.png",
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
        "schema": "vision_memory.r11-new-oracle-terminal-capture-delivery-manifest.v1",
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
        raise ValueError("Raw terminal archive SHA-256 mismatch.")
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
        save_selected_evidence(bundle)
        result = read_json(bundle, f"{FORMAL}/result.json")
        rows = [
            json.loads(line)
            for line in read_member(bundle, f"{FORMAL}/scan_metrics.jsonl")
            .decode("utf-8")
            .splitlines()
        ]
    if (
        result["classification"] != "microscopic_partial_code_capture"
        or result["counters"]
        != {
            "full_chain_forward_calls": 35,
            "reader_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        or result["formal_success"] is not False
        or result["phase2_allowed"] is not False
    ):
        raise ValueError("Unexpected terminal formal boundary.")
    selected = canonical_rows(rows)
    if not all(
        left["endpoint_fp32_sha256"] == right["endpoint_fp32_sha256"]
        for left, right in zip(
            selected,
            sorted(
                (row for row in rows if row["pass"] == "plateau-inward"),
                key=lambda row: row["point_index"],
            ),
            strict=True,
        )
    ):
        raise ValueError("Duplicate terminal path mismatch.")
    render_full_path(rows, figures)
    render_boundary_zoom(rows, figures)
    image_stats = render_montage(figures)
    boundary = {row["point_index"]: row for row in selected}
    summary = {
        "schema": "vision_memory.r11-new-oracle-terminal-capture-derived-summary.v1",
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
        "scan_summary": result["scan_summary"],
        "boundary_points": {str(index): boundary[index] for index in (0, 1, 2, 8, 9, 10, 11, 16)},
        "observations": {
            "largest_exact_endpoint_requested_remaining_l2": 3e-5,
            "largest_strong_capture_requested_remaining_l2": result["scan_summary"][
                "strong_capture_prefix_max_requested_remaining_l2"
            ],
            "first_failed_requested_remaining_l2": 1e-4,
            "first_failed_loss_ratio": boundary[9]["loss_ratio_to_plateau"],
            "first_failed_bf16_changed_coordinate_count": round(
                65536 * (1.0 - boundary[9]["bf16_equal_fraction_to_teacher"])
            ),
            "last_exact_bf16_changed_coordinate_count": round(
                65536 * (1.0 - boundary[8]["bf16_equal_fraction_to_teacher"])
            ),
            "interpretation": (
                "The exact endpoint tolerates three changed BF16 coordinates through requested "
                "delta=3e-5, then fails abruptly by delta=1e-4. The terminal capture region is "
                "microscopic and unsuitable for blind FP32-to-BF16 surrogate-gradient search."
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
