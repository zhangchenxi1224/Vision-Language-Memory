"""Verify and render the R11_new reachable-endpoint control delivery."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_reachable_control as core  # noqa: E402


ARCHIVE = ROOT / "raw" / "r11-new-reachable-986a181-20260907-round01.tar.gz"
ARCHIVE_SHA256 = "41741aaf7e85022d4a1440a7a0b07ed580151201fec01e10db82205427dc74fb"
PREFLIGHT = "r11-new-reachable-986a181-20260907-round01-preflight"
FORMAL = "r11-new-reachable-986a181-20260907-round01-formal"
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_tensor_payload(bundle: tarfile.TarFile, name: str) -> dict:
    return torch.load(io.BytesIO(read_member(bundle, name)), map_location="cpu", weights_only=True)


def verify_inventory(bundle: tarfile.TarFile, prefix: str) -> dict:
    inventory = read_json(bundle, f"{prefix}/artifact_inventory.json")
    listed = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        if relative in listed or relative.startswith("/") or "\\" in relative or ".." in relative.split("/"):
            raise ValueError(f"Unsafe inventory path: {relative}")
        payload = read_member(bundle, f"{prefix}/{relative}")
        if len(payload) != item["bytes"] or sha256_bytes(payload) != item["sha256"]:
            raise ValueError(f"Inventory mismatch: {relative}")
        listed.add(relative)
    observed = {
        member.name.removeprefix(f"{prefix}/")
        for member in bundle.getmembers()
        if member.isfile()
        and member.name.startswith(f"{prefix}/")
        and member.name != f"{prefix}/artifact_inventory.json"
    }
    if observed != listed or inventory["artifact_count"] != len(listed):
        raise ValueError(f"Incomplete inventory for {prefix}")
    return {"passed": True, "artifact_count": len(listed)}


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

    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        unsafe = [
            member.name
            for member in bundle.getmembers()
            if member.name.startswith("/")
            or ".." in Path(member.name).parts
            or not member.name.startswith((PREFLIGHT, FORMAL))
        ]
        if unsafe:
            raise ValueError(f"Unsafe archive members: {unsafe[:3]}")
        inventories = {
            "preflight": verify_inventory(bundle, PREFLIGHT),
            "formal": verify_inventory(bundle, FORMAL),
        }
        preflight = read_json(bundle, f"{PREFLIGHT}/result.json")
        result = read_json(bundle, f"{FORMAL}/result.json")
        runtime = read_json(bundle, f"{FORMAL}/runtime.json")
        metrics = [
            json.loads(line)
            for line in read_member(bundle, f"{FORMAL}/metrics.jsonl").decode("utf-8").splitlines()
        ]
        target = read_tensor_payload(bundle, f"{FORMAL}/target/reachable_target.pt")
        checkpoints = {
            step: read_tensor_payload(bundle, f"{FORMAL}/checkpoints/step-{step:03d}.pt")
            for step in CHECKPOINT_STEPS
        }

    for name in (
        "source_only_x_T_init_fp32",
        "teacher_x_T_fp32",
        "teacher_endpoint_fp32",
        "normalized_noise_fp32",
    ):
        if canonical_tensor_sha256(target[name]) != target["tensor_sha256"][name]:
            raise ValueError(f"Target tensor hash mismatch: {name}")
    source = target["source_only_x_T_init_fp32"]
    teacher_x_t = target["teacher_x_T_fp32"]
    teacher_endpoint = target["teacher_endpoint_fp32"]
    stored_noise = target["normalized_noise_fp32"]
    formula_exact = torch.equal(teacher_x_t, source + stored_noise)
    if not formula_exact:
        raise ValueError("Stored reachable-target construction formula failed.")

    config = core.load_config()
    regenerated_noise = core.normalized_teacher_noise(stored_noise.shape, config)
    seed_replay_equal = torch.equal(regenerated_noise, stored_noise)
    runtime_match = (
        runtime["packages"]["torch"] == torch.__version__
        and runtime["platform"].split("-", 1)[0] == platform.system()
    )
    if runtime_match and not seed_replay_equal:
        raise ValueError("Seed replay drifted under the recorded runtime family.")

    checkpoint_geometry = []
    for step, payload in checkpoints.items():
        student = payload["student_x_T_fp32"]
        endpoint = payload["endpoint_fp32"]
        for name, tensor in (("student_x_T_fp32", student), ("endpoint_fp32", endpoint)):
            if canonical_tensor_sha256(tensor) != payload["tensor_sha256"][name]:
                raise ValueError(f"Checkpoint tensor hash mismatch at {step}: {name}")
        displacement = student - source
        teacher_direction = teacher_x_t - source
        cosine = None if step == 0 else float(
            torch.nn.functional.cosine_similarity(
                displacement.flatten().double(), teacher_direction.flatten().double(), dim=0
            )
        )
        checkpoint_geometry.append(
            {
                "optimizer_step": step,
                "endpoint_mse": float((endpoint - teacher_endpoint).square().mean()),
                "x_T_mse_to_teacher": float((student - teacher_x_t).square().mean()),
                "x_T_mse_from_source": float(displacement.square().mean()),
                "displacement_cosine_to_teacher_direction": cosine,
            }
        )

    final_endpoint_mse = checkpoint_geometry[-1]["endpoint_mse"]
    if not np.isclose(final_endpoint_mse, result["endpoint_distance"]["mse"], rtol=1e-6, atol=1e-8):
        raise ValueError("Final endpoint MSE does not reproduce.")
    if len(metrics) != 256 or [row["optimizer_step"] for row in metrics] != list(range(1, 257)):
        raise ValueError("Optimizer metrics are incomplete or non-contiguous.")
    technical = core.validate_metrics(metrics, config)
    if technical != result["technical_audit"]:
        raise ValueError("Technical metric audit does not reproduce.")

    steps = np.asarray([row["optimizer_step"] for row in metrics])
    losses = np.asarray([row["loss_before_step"] for row in metrics])
    gradients = np.asarray([row["gradient_norm"] for row in metrics])
    updates = np.asarray([row["x_T_update_norm"] for row in metrics])
    checkpoint_x = np.asarray(CHECKPOINT_STEPS)
    endpoint_ratios = np.asarray(
        [item["endpoint_mse"] / checkpoint_geometry[0]["endpoint_mse"] for item in checkpoint_geometry]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    axes[0, 0].plot(steps, losses, color="#1D4ED8", linewidth=1.5)
    axes[0, 0].axhline(0.001, color="#B91C1C", linestyle="--", label="absolute gate = 0.001")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Endpoint loss plateaus far above gate")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("FP32 endpoint MSE (log)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, gradients, color="#7C3AED", label="gradient norm")
    axes[0, 1].plot(steps, updates, color="#D97706", alpha=0.8, label="xT update norm")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Dense gradients persist; cosine LR reaches zero")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Norm (log)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        checkpoint_x,
        [item["x_T_mse_to_teacher"] for item in checkpoint_geometry],
        marker="o",
        label="MSE to exact teacher xT",
        color="#B91C1C",
    )
    axes[1, 0].plot(
        checkpoint_x,
        [item["x_T_mse_from_source"] for item in checkpoint_geometry],
        marker="o",
        label="MSE from source init",
        color="#0F766E",
    )
    axes[1, 0].set_title("Student moves away from the known exact solution")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("xT-space MSE")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(checkpoint_x, endpoint_ratios, marker="o", color="#1D4ED8")
    axes[1, 1].axhline(0.01, color="#B91C1C", linestyle="--", label="ratio gate = 0.01")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Only 20.6% MSE reduction; 99% required")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Endpoint MSE / M0 (log)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(
        figures / "optimization_diagnostics.png",
        dpi=180,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)

    montage_files = [ROOT / "images" / "reachable-target.png"] + [
        ROOT / "images" / f"student-step-{step:03d}.png" for step in CHECKPOINT_STEPS
    ]
    montage_titles = ["Reachable target"] + [f"Student step {step}" for step in CHECKPOINT_STEPS]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    image_stats = {}
    for axis, path, title in zip(axes.flat, montage_files, montage_titles, strict=True):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(title)
        axis.axis("off")
        image_stats[path.name] = image_statistics(path)
    fig.suptitle("Diagnostic endpoints are near-uniform and non-semantic", fontsize=13)
    fig.savefig(
        figures / "endpoint_montage.png",
        dpi=150,
        metadata={"Software": "Vision-Language-Memory"},
    )
    plt.close(fig)

    summary = {
        "schema": "vision_memory.r11-new-reachable-control-derived-summary.v1",
        "archive": {"path": ARCHIVE.relative_to(ROOT).as_posix(), "sha256": ARCHIVE_SHA256},
        "inventories": inventories,
        "preflight": {
            "engineering_gate": preflight["engineering_gate"],
            "preflight_gate": preflight["preflight_gate"],
            "m0_mse": preflight["m0_mse"],
            "gradient_probe": preflight["gradient_probe"],
            "counters": preflight["counters"],
        },
        "formal": {
            "engineering_gate": result["engineering_gate"],
            "reachable_control_gate": result["reachable_control_gate"],
            "formal_success": result["formal_success"],
            "phase2_allowed": result["phase2_allowed"],
            "decision": result["decision"],
            "m0_mse": result["m0_mse"],
            "endpoint_distance": result["endpoint_distance"],
            "counters": result["counters"],
            "technical_audit": result["technical_audit"],
            "loss_minimum": {"value": float(losses.min()), "optimizer_step": int(steps[losses.argmin()])},
        },
        "checkpoint_geometry": checkpoint_geometry,
        "target_construction_verification": {
            "stored_formula_bitwise_equal": formula_exact,
            "stored_noise_population_rms": float(stored_noise.double().square().mean().sqrt()),
            "stored_noise_sha256": canonical_tensor_sha256(stored_noise),
            "local_runtime_matches_training": runtime_match,
            "local_seed_replay_bitwise_equal": seed_replay_equal,
            "local_seed_replay_max_abs_delta": float((regenerated_noise - stored_noise).abs().max()),
            "training_torch": runtime["packages"]["torch"],
            "local_torch": torch.__version__,
            "training_platform": runtime["platform"],
            "local_platform": platform.platform(),
            "interpretation": (
                "Exact seed replay is only an asserted gate under the recorded runtime family; "
                "cross-runtime mismatch is reported, never relabeled as exact reproducibility."
            ),
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
        artifacts.append(
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema": "vision_memory.r11-new-reachable-control-delivery-manifest.v1",
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
