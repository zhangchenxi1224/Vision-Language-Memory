"""Verify, extract selected evidence, and render the reachable-basin delivery."""

from __future__ import annotations

import hashlib
import io
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
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from vision_memory.training import r11_new_reachable_basin_control as core  # noqa: E402


ARCHIVE = ROOT / "raw" / "r11-new-basin-7d4a845-20260907-round01.tar.gz"
ARCHIVE_SHA256 = "5134e63f4cc2574a8d553341cc536052038727dd1b117cec64506b7165a37b4e"
PREFLIGHT = "r11-new-basin-7d4a845-20260907-round01-preflight"
FORMAL = "r11-new-basin-7d4a845-20260907-round01-formal"
CONDITIONS = ("alpha-050", "alpha-099")
STEPS = (0, 64, 128, 192, 256)
COLORS = {"alpha-050": "#1D4ED8", "alpha-099": "#D97706"}


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
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.parts[0] not in (PREFLIGHT, FORMAL)
                or member.issym() or member.islnk() or member.isdev()):
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


def read_tensor(bundle: tarfile.TarFile, name: str) -> dict:
    return torch.load(io.BytesIO(read_member(bundle, name)), map_location="cpu", weights_only=True)


def verify_inventory(bundle: tarfile.TarFile, prefix: str) -> dict:
    inventory = read_json(bundle, f"{prefix}/artifact_inventory.json")
    listed = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        if relative in listed or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
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
    return {"passed": True, "artifact_count": len(listed)}


def materialize_and_audit(bundle: tarfile.TarFile, config: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="vlm-r11-basin-audit-") as temporary:
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
        "artifact_inventory.json", "config.json", "determinism.json", "environment.txt",
        "launch.json", "manifest.json", "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json", "REPORT.md", "result.json", "runtime.json",
        "stderr.log", "stdout.log", "terminal.json",
    )
    for label, prefix in (("preflight", PREFLIGHT), ("formal", FORMAL)):
        for relative in common:
            destination = ROOT / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(read_member(bundle, f"{prefix}/{relative}"))
    for condition in CONDITIONS:
        relative = f"conditions/{condition}/metrics.jsonl"
        destination = ROOT / "formal" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_member(bundle, f"{FORMAL}/{relative}"))
    image_members = {"fixed-target.png": f"{FORMAL}/target/fixed_parent_target.png"}
    for condition in CONDITIONS:
        for step in STEPS:
            image_members[f"{condition}-step-{step:03d}.png"] = (
                f"{FORMAL}/conditions/{condition}/images/step-{step:03d}.png"
            )
    image_root = ROOT / "images"
    image_root.mkdir(exist_ok=True)
    for name, member in image_members.items():
        (image_root / name).write_bytes(read_member(bundle, member))


def image_statistics(path: Path) -> dict:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    total_variation = (
        np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()
    ) / 2.0
    return {
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float(total_variation),
    }


def main() -> None:
    if sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Raw archive SHA-256 mismatch.")
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    config = core.load_config()

    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        validate_members(bundle)
        inventories = {
            "preflight": verify_inventory(bundle, PREFLIGHT),
            "formal": verify_inventory(bundle, FORMAL),
        }
        local_audits = materialize_and_audit(bundle, config)
        save_selected_evidence(bundle)
        result = read_json(bundle, f"{FORMAL}/result.json")
        target = read_tensor(bundle, f"{FORMAL}/target/fixed_parent_target.pt")
        metrics = {
            condition: [
                json.loads(line)
                for line in read_member(bundle, f"{FORMAL}/conditions/{condition}/metrics.jsonl")
                .decode("utf-8").splitlines()
            ]
            for condition in CONDITIONS
        }
        checkpoints = {
            condition: {
                step: read_tensor(bundle, f"{FORMAL}/conditions/{condition}/checkpoints/step-{step:03d}.pt")
                for step in STEPS
            }
            for condition in CONDITIONS
        }

    teacher_x_t = target["teacher_x_T_fp32"]
    source_x_t = target["source_only_x_T_init_fp32"]
    condition_geometry = {}
    for condition in CONDITIONS:
        alpha = float(result["conditions"][condition]["alpha"])
        start = core.warm_start(source_x_t, teacher_x_t, alpha)
        records = []
        for step in STEPS:
            payload = checkpoints[condition][step]
            student = payload["student_x_T_fp32"]
            records.append({
                "optimizer_step": step,
                "endpoint_distance": payload["distance"],
                "x_T_l2_to_teacher": float(torch.linalg.vector_norm((student - teacher_x_t).double())),
                "x_T_mse_to_teacher": float((student - teacher_x_t).square().mean()),
                "x_T_l2_from_condition_start": float(torch.linalg.vector_norm((student - start).double())),
            })
        rows = metrics[condition]
        initial_distance = records[0]["x_T_l2_to_teacher"]
        first_update = float(rows[0]["x_T_update_norm"])
        condition_geometry[condition] = {
            "alpha": alpha,
            "checkpoints": records,
            "first_step": {
                "loss_before": float(rows[0]["loss_before_step"]),
                "loss_after_first_step": float(rows[1]["loss_before_step"]),
                "loss_multiplier": float(rows[1]["loss_before_step"] / rows[0]["loss_before_step"]),
                "x_T_update_norm": first_update,
                "initial_x_T_l2_to_teacher": initial_distance,
                "update_to_remaining_distance_ratio": first_update / initial_distance,
            },
            "minimum_observed_pre_step_loss": {
                "value": float(min(row["loss_before_step"] for row in rows)),
                "optimizer_step": int(min(rows, key=lambda row: row["loss_before_step"])["optimizer_step"]),
            },
        }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    for condition in CONDITIONS:
        rows = metrics[condition]
        x = np.asarray([row["optimizer_step"] for row in rows])
        y = np.asarray([row["loss_before_step"] for row in rows])
        axes[0, 0].plot(x, y, color=COLORS[condition], label=condition)
        m0 = float(result["conditions"][condition]["m0_mse"])
        axes[0, 1].plot(x, y / m0, color=COLORS[condition], label=condition)
        axes[1, 1].plot(
            STEPS,
            [item["x_T_l2_to_teacher"] for item in condition_geometry[condition]["checkpoints"]],
            marker="o",
            color=COLORS[condition],
            label=condition,
        )
    axes[0, 0].axhline(0.001, color="#B91C1C", linestyle="--", label="absolute gate")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Near solution, LR=0.05 immediately overshoots")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Endpoint MSE (log)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(0.01, color="#B91C1C", linestyle="--", label="ratio gate")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Neither condition achieves 99% reduction")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Endpoint MSE / M0 (log)")
    axes[0, 1].legend(fontsize=8)

    names = list(CONDITIONS)
    remaining = [condition_geometry[name]["first_step"]["initial_x_T_l2_to_teacher"] for name in names]
    updates = [condition_geometry[name]["first_step"]["x_T_update_norm"] for name in names]
    positions = np.arange(len(names))
    width = 0.34
    axes[1, 0].bar(positions - width / 2, remaining, width, label="distance to exact xT", color="#0F766E")
    axes[1, 0].bar(positions + width / 2, updates, width, label="first Adam update", color="#D97706")
    axes[1, 0].set_xticks(positions, names)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Adam update is 4.8× the alpha=.99 distance")
    axes[1, 0].set_ylabel("xT-space L2 norm (log)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Training moves both students away from exact xT")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("xT L2 distance to teacher (log)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(figures / "basin_diagnostics.png", dpi=180,
                metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)

    montage = [
        (ROOT / "images/fixed-target.png", "Fixed reachable target"),
        (ROOT / "images/alpha-050-step-000.png", "alpha=.50 / step 0"),
        (ROOT / "images/alpha-050-step-256.png", "alpha=.50 / step 256"),
        (ROOT / "images/alpha-099-step-000.png", "alpha=.99 / step 0"),
        (ROOT / "images/alpha-099-step-256.png", "alpha=.99 / step 256"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    image_stats = {}
    for axis, (path, title) in zip(axes.flat, montage, strict=False):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title)
        axis.axis("off")
        image_stats[path.name] = image_statistics(path)
    axes.flat[-1].axis("off")
    fig.suptitle("Fixed-target basin control remains non-semantic", fontsize=13)
    fig.savefig(figures / "basin_endpoint_montage.png", dpi=150,
                metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)

    summary = {
        "schema": "vision_memory.r11-new-reachable-basin-derived-summary.v1",
        "archive": {"path": ARCHIVE.relative_to(ROOT).as_posix(), "sha256": ARCHIVE_SHA256},
        "inventories": inventories,
        "local_independent_audits": local_audits,
        "formal_result": {
            "engineering_gate": result["engineering_gate"],
            "condition_gates": result["condition_gates"],
            "classification": result["classification"],
            "counters": result["counters"],
            "elapsed_seconds": result["elapsed_seconds"],
            "formal_success": result["formal_success"],
            "phase2_allowed": result["phase2_allowed"],
        },
        "conditions": {
            name: {
                "m0_mse": result["conditions"][name]["m0_mse"],
                "endpoint_distance": result["conditions"][name]["endpoint_distance"],
                "technical_audit": result["conditions"][name]["technical_audit"],
                **condition_geometry[name],
            }
            for name in CONDITIONS
        },
        "image_statistics": image_stats,
    }
    (derived / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema": "vision_memory.r11-new-reachable-basin-delivery-manifest.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
