"""Verify and render the R11_new local direction-fidelity delivery."""

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

from vision_memory.training import r11_new_direction_fidelity as core  # noqa: E402


ARCHIVE = ROOT / "raw" / "r11-new-direction-8f696fa-20260907-round01.tar.gz"
ARCHIVE_SHA256 = "02cf6befb4171168d87c4b6512820b73904a05ef20712185177666b1613ea6f3"
PREFLIGHT = "r11-new-direction-8f696fa-20260907-round01-preflight"
FORMAL = "r11-new-direction-8f696fa-20260907-round01-formal"
COLORS = {
    "negative-autograd": "#2563EB",
    "negative-adam-preconditioned": "#D97706",
    "teacher-residual": "#059669",
    "deterministic-orthogonal-control": "#7C3AED",
}
LABELS = {
    "negative-autograd": "negative autograd",
    "negative-adam-preconditioned": "negative Adam proposal",
    "teacher-residual": "oracle teacher residual",
    "deterministic-orthogonal-control": "orthogonal control",
}


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
    inventory = read_json(bundle, f"{prefix}/artifact_inventory.json")
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
        "inventory_sha256": sha256_bytes(
            read_member(bundle, f"{prefix}/artifact_inventory.json")
        ),
    }


def materialize_and_audit(bundle: tarfile.TarFile, config: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="vlm-r11-direction-audit-") as temporary:
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
        "fixed-target.png": f"{FORMAL}/target/fixed_parent_target.png",
        "alpha099-start.png": f"{FORMAL}/anchors/alpha099-start.png",
        "lr001-raw256.png": f"{FORMAL}/anchors/lr001-raw256.png",
        "alpha-neggrad-r0.3.png": (
            f"{FORMAL}/scan_images/alpha099-start__negative-autograd__r04__plus.png"
        ),
        "alpha-adam-r1.0.png": (
            f"{FORMAL}/scan_images/alpha099-start__negative-adam-preconditioned__r05__plus.png"
        ),
        "alpha-oracle-r2.0.png": (
            f"{FORMAL}/scan_images/alpha099-start__teacher-residual__r06__plus.png"
        ),
        "plateau-neggrad-r0.001.png": (
            f"{FORMAL}/scan_images/lr001-raw256__negative-autograd__r01__plus.png"
        ),
        "plateau-adam-r0.01.png": (
            f"{FORMAL}/scan_images/lr001-raw256__negative-adam-preconditioned__r02__plus.png"
        ),
        "plateau-oracle-r0.0001.png": (
            f"{FORMAL}/scan_images/lr001-raw256__teacher-residual__r00__plus.png"
        ),
        "plateau-oracle-r2.0.png": (
            f"{FORMAL}/scan_images/lr001-raw256__teacher-residual__r06__plus.png"
        ),
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


def render_loss_grid(rows: list[dict], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    for axis, anchor in zip(axes, core.ANCHORS, strict=True):
        for direction in core.DIRECTIONS:
            selected = [
                row for row in rows if row["anchor"] == anchor and row["direction"] == direction
            ]
            for sign, style in (("plus", "-"), ("minus", ":")):
                signed = sorted(
                    (row for row in selected if row["sign"] == sign),
                    key=lambda row: row["radius_index"],
                )
                axis.plot(
                    [row["radius_l2"] for row in signed],
                    [row["loss_ratio_to_anchor"] for row in signed],
                    color=COLORS[direction],
                    linestyle=style,
                    marker="o" if sign == "plus" else None,
                    markersize=3,
                    label=LABELS[direction] if sign == "plus" else None,
                    alpha=0.95 if sign == "plus" else 0.55,
                )
        axis.axhline(1.0, color="#111827", linewidth=1, linestyle="--", label="anchor")
        axis.axhline(0.9, color="#DC2626", linewidth=1, linestyle="--", label="descent gate")
        axis.axhline(0.1, color="#991B1B", linewidth=1, linestyle=":", label="oracle strong gate")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(anchor)
        axis.set_xlabel("Signed-scan L2 radius (solid: +, dotted: −)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Endpoint MSE / anchor MSE (log)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Local direction scan: descent at start, no actionable descent at plateau")
    fig.savefig(
        figures / "direction_scan_loss_ratio.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_derivative_fidelity(result: dict, config: dict, figures: Path) -> None:
    radii = np.asarray(config["direction_scan_contract"]["signed_radii_l2"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for axis, anchor in zip(axes, core.ANCHORS, strict=True):
        for direction in core.DIRECTIONS:
            record = result["scan_summary"][anchor][direction]
            axis.plot(
                radii,
                record["central_finite_difference"],
                marker="o",
                markersize=3,
                color=COLORS[direction],
                label=f"finite: {LABELS[direction]}",
            )
            axis.axhline(
                record["analytic_directional_derivative"],
                color=COLORS[direction],
                linestyle=":",
                linewidth=1,
                alpha=0.8,
            )
        axis.axhline(0.0, color="#111827", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_title(anchor)
        axis.set_xlabel("L2 radius")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Directional derivative (symlog); dotted = autograd analytic")
    axes[0].legend(fontsize=6.5, loc="best")
    fig.suptitle("BF16 map: analytic gradients poorly predict measured finite differences")
    fig.savefig(
        figures / "finite_difference_fidelity.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_plateau_oracle_path(rows: list[dict], result: dict, figures: Path) -> None:
    selected = sorted(
        (
            row
            for row in rows
            if row["anchor"] == "lr001-raw256"
            and row["direction"] == "teacher-residual"
            and row["sign"] == "plus"
        ),
        key=lambda row: row["radius_index"],
    )
    exact_radius = result["anchors"]["lr001-raw256"]["quantization"]["fp32_l2_to_teacher"]
    radii = [0.0, *[row["radius_l2"] for row in selected], exact_radius]
    ratios = [1.0, *[row["loss_ratio_to_anchor"] for row in selected], 0.0]
    equal = [
        result["anchors"]["lr001-raw256"]["quantization"]["bf16_equal_fraction_to_teacher"],
        *[row["bf16_equal_fraction_to_teacher"] for row in selected],
        1.0,
    ]
    fig, axis = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    axis.plot(radii, ratios, marker="o", color="#DC2626", label="endpoint MSE / plateau MSE")
    axis.axhline(0.9, color="#991B1B", linestyle="--", linewidth=1, label="meaningful descent gate")
    axis.axhline(0.1, color="#7F1D1D", linestyle=":", linewidth=1, label="strong oracle gate")
    axis.set_xlabel("Movement from plateau toward exact teacher xT (L2)")
    axis.set_ylabel("Endpoint loss ratio")
    axis.set_ylim(bottom=-0.04)
    axis.grid(alpha=0.2)
    second = axis.twinx()
    second.plot(radii, equal, marker="s", color="#2563EB", label="BF16 coordinates equal to teacher")
    second.set_ylabel("BF16 coordinate equality fraction")
    second.set_ylim(0.0, 1.05)
    radius_two = next(row for row in selected if row["radius_l2"] == 2.0)
    axis.annotate(
        "r=2.0; 7.9% BF16 codes still differ",
        xy=(2.0, radius_two["loss_ratio_to_anchor"]),
        xytext=(1.05, 1.50),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=8,
    )
    axis.annotate(
        f"exact teacher replay\nr={exact_radius:.6f}",
        xy=(exact_radius, 0.0),
        xytext=(1.38, 0.16),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=8,
    )
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, fontsize=8, loc="upper left")
    axis.set_title("Coarse oracle scan leaves an unresolved transition near exact teacher")
    fig.savefig(
        figures / "plateau_teacher_residual_path.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)


def render_montage(figures: Path) -> dict:
    panels = (
        ("fixed-target.png", "Exact teacher target"),
        ("alpha099-start.png", "alpha=.99 start"),
        ("alpha-neggrad-r0.3.png", "start + neg-grad r=.3"),
        ("alpha-adam-r1.0.png", "start + Adam r=1"),
        ("lr001-raw256.png", "lr=.001 raw step 256"),
        ("plateau-neggrad-r0.001.png", "plateau + neg-grad r=.001"),
        ("plateau-adam-r0.01.png", "plateau + Adam r=.01"),
        ("plateau-oracle-r2.0.png", "plateau + oracle r=2"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.7), constrained_layout=True)
    statistics = {}
    for axis, (name, title) in zip(axes.flat, panels, strict=True):
        path = ROOT / "images" / name
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        statistics[name] = image_statistics(path)
    fig.suptitle("Direction-scan endpoints remain non-semantic to human inspection", fontsize=13)
    fig.savefig(
        figures / "direction_endpoint_montage.png",
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
            or "source" in path.parts
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
        "schema": "vision_memory.r11-new-local-direction-fidelity-delivery-manifest.v1",
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
        raise ValueError("Raw archive SHA-256 mismatch.")
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
        result["classification"] != "plateau_oracle_not_strongly_descending"
        or result["counters"]
        != {
            "full_chain_forward_calls": 115,
            "reader_forward_calls": 0,
            "backward_calls": 2,
            "optimizer_steps": 0,
        }
        or result["formal_success"] is not False
        or result["phase2_allowed"] is not False
    ):
        raise ValueError("Unexpected formal result boundary.")
    render_loss_grid(rows, figures)
    render_derivative_fidelity(result, config, figures)
    render_plateau_oracle_path(rows, result, figures)
    image_stats = render_montage(figures)
    plateau_oracle = [
        row
        for row in rows
        if row["anchor"] == "lr001-raw256"
        and row["direction"] == "teacher-residual"
        and row["sign"] == "plus"
    ]
    summary = {
        "schema": "vision_memory.r11-new-local-direction-fidelity-derived-summary.v1",
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
        "anchors": result["anchors"],
        "scan_summary": result["scan_summary"],
        "plateau_teacher_residual_positive_rows": sorted(
            plateau_oracle, key=lambda row: row["radius_index"]
        ),
        "observations": {
            "alpha_start_negative_autograd_best_ratio": result["scan_summary"]["alpha099-start"][
                "negative-autograd"
            ]["best_positive_loss_ratio"],
            "alpha_start_adam_best_ratio": result["scan_summary"]["alpha099-start"][
                "negative-adam-preconditioned"
            ]["best_positive_loss_ratio"],
            "plateau_negative_autograd_best_ratio": result["scan_summary"]["lr001-raw256"][
                "negative-autograd"
            ]["best_positive_loss_ratio"],
            "plateau_adam_best_ratio": result["scan_summary"]["lr001-raw256"][
                "negative-adam-preconditioned"
            ]["best_positive_loss_ratio"],
            "plateau_teacher_residual_best_ratio": result["scan_summary"]["lr001-raw256"][
                "teacher-residual"
            ]["best_positive_loss_ratio"],
            "plateau_oracle_radius_2_ratio": next(
                row["loss_ratio_to_anchor"]
                for row in plateau_oracle
                if row["radius_l2"] == 2.0
            ),
            "plateau_oracle_radius_2_fp32_l2_remaining": next(
                row["fp32_l2_to_teacher"]
                for row in plateau_oracle
                if row["radius_l2"] == 2.0
            ),
            "plateau_oracle_radius_2_bf16_equal_fraction": next(
                row["bf16_equal_fraction_to_teacher"]
                for row in plateau_oracle
                if row["radius_l2"] == 2.0
            ),
            "exact_teacher_replay_loss_ratio": 0.0,
            "interpretation": (
                "At the lr=.001 plateau, neither autograd nor Adam has preregistered actionable "
                "descent. The oracle straight path is non-monotone and only becomes exact at the "
                "teacher xT; resolve the terminal BF16/Jacobian cliff before any optimizer rerun."
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
