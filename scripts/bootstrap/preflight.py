from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTED_PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "huggingface-hub",
    "safetensors",
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def marker_value(path: Path, name: str) -> str | None:
    marker = path / name
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any, *, required: bool = True) -> None:
    checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the reproducible local or cluster runtime")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local")
    parser.add_argument("--model-root", type=Path, default=None, help="Override VLM_MODEL_ROOT")
    parser.add_argument(
        "--expected-torch",
        default=os.environ.get("VLM_EXPECTED_TORCH"),
        help="Exact cluster torch version; required in cluster mode.",
    )
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    runtime = lock["canonical_runtime"]
    expected_packages = runtime["packages"]
    packages = {name: package_version(name) for name in REPORTED_PACKAGES}
    model_root = args.model_root or Path(os.environ.get("VLM_MODEL_ROOT", ROOT / "models"))
    model_root = model_root.expanduser().resolve()

    dreamlite_source = ROOT / lock["sources"]["dreamlite_reference"]["local_dir"]
    source_report = {
        "path": str(dreamlite_source),
        "expected_revision": lock["sources"]["dreamlite_reference"]["revision"],
        "actual_revision": git_value(dreamlite_source, "rev-parse", "HEAD"),
        "status": git_value(dreamlite_source, "status", "--short"),
    }

    models: dict[str, dict[str, Any]] = {}
    for name, spec in lock["models"].items():
        path = model_root / Path(spec["local_dir"]).name
        weight_files = [*path.rglob("*.safetensors"), *path.rglob("*.bin")] if path.exists() else []
        models[name] = {
            "path": str(path),
            "exists": path.exists(),
            "has_model_index": (path / "model_index.json").exists(),
            "has_config": (path / "config.json").exists(),
            "locked_revision": marker_value(path, ".locked_revision"),
            "metadata_complete": marker_value(path, ".metadata_complete"),
            "snapshot_complete": marker_value(path, ".snapshot_complete"),
            "weight_file_count": len(weight_files),
            "weight_bytes": sum(item.stat().st_size for item in weight_files),
        }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpus = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        ] if cuda_available else []
        torch_runtime: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        torch_runtime = {"error": repr(exc), "cuda_available": False, "gpu_count": 0, "gpus": []}

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "dreamlite_source_revision",
        source_report["actual_revision"] == source_report["expected_revision"],
        source_report,
    )
    add_check(checks, "dreamlite_source_clean", source_report["status"] == "", source_report["status"])
    for name, expected in expected_packages.items():
        add_check(checks, f"package:{name}", packages.get(name) == expected, {"expected": expected, "actual": packages.get(name)})

    if args.mode == "cluster":
        add_check(
            checks,
            "python_target",
            sys.version_info[:2] == tuple(int(part) for part in runtime["python"].split(".")),
            {"expected": runtime["python"], "actual": f"{sys.version_info.major}.{sys.version_info.minor}"},
        )
        add_check(
            checks,
            "torch_pin_provided",
            bool(args.expected_torch) and not str(args.expected_torch).startswith("REPLACE_"),
            args.expected_torch,
        )
        if args.expected_torch and not str(args.expected_torch).startswith("REPLACE_"):
            add_check(
                checks,
                "torch_version",
                packages.get("torch") == args.expected_torch,
                {"expected": args.expected_torch, "actual": packages.get("torch")},
            )
        add_check(checks, "cuda_available", bool(torch_runtime.get("cuda_available")), torch_runtime)
        add_check(
            checks,
            "gpu_count",
            int(torch_runtime.get("gpu_count", 0)) >= args.min_gpus,
            {"minimum": args.min_gpus, "actual": torch_runtime.get("gpu_count", 0)},
        )
        if args.min_gpu_memory_gib > 0:
            qualifying = sum(
                gpu["memory_gib"] >= args.min_gpu_memory_gib for gpu in torch_runtime.get("gpus", [])
            )
            add_check(
                checks,
                "gpu_memory",
                qualifying >= args.min_gpus,
                {"minimum_gib": args.min_gpu_memory_gib, "qualifying_gpus": qualifying},
            )

        for name, spec in lock["models"].items():
            state = models[name]
            expected_revision = spec["revision"]
            expected_bytes = int(spec.get("weight_bytes", spec.get("approx_bytes", 0)))
            add_check(
                checks,
                f"model_snapshot:{name}",
                state["locked_revision"] == expected_revision
                and state["snapshot_complete"] == expected_revision
                and state["weight_file_count"] > 0
                and (not expected_bytes or state["weight_bytes"] >= int(expected_bytes * 0.85)),
                {"expected_revision": expected_revision, **state},
            )

    report = {
        "mode": args.mode,
        "project_root": str(ROOT),
        "model_root": str(model_root),
        "platform": platform.platform(),
        "python": sys.version,
        "python_target": runtime["python"],
        "packages": packages,
        "dreamlite_source": source_report,
        "models": models,
        "torch_runtime": torch_runtime,
        "checks": checks,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    failed = [check for check in checks if check["required"] and not check["ok"]]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
