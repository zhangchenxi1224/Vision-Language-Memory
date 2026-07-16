from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGES = {
    "accelerate": "1.12.0",
    "diffusers": "0.39.0",
    "einops": "0.8.1",
    "huggingface-hub": "0.36.0",
    "numpy": "1.26.4",
    "peft": "0.18.1",
    "Pillow": "12.3.0",
    "pytest": "9.1.1",
    "PyYAML": "6.0.2",
    "ruff": "0.14.10",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.1",
    "tqdm": "4.67.1",
    "transformers": "4.57.3",
}
STRICT_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
SAFE_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TORCH_HOME",
    "VLM_INSPIRE_IMAGE",
    "VLM_INSPIRE_INSTANCE",
    "VLM_INSPIRE_NODE",
    "VLM_INSPIRE_PROJECT",
    "VLM_INSPIRE_WORKSPACE",
    "VLM_MODEL_ROOT",
    "VLM_RUN_ROOT",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_git(repo: Path, *args: str) -> str | None:
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


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def collect_inventory(repo: Path, model_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    model_root = model_root.expanduser().resolve()
    status = run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    packages = {
        name: package_version(name)
        for name in sorted(set(EXPECTED_PACKAGES) | {"torch", "torchvision"})
    }

    try:
        import torch

        torch_file = Path(torch.__file__).resolve()
        venv_purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        gpus = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpus.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_mib": properties.total_memory // 2**20,
                    }
                )
        torch_runtime: dict[str, Any] = {
            "runtime_version": torch.__version__,
            "distribution_version": package_version("torch"),
            "file": str(torch_file),
            "venv_purelib": str(venv_purelib),
            "installed_inside_venv": path_is_within(torch_file, venv_purelib),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
    except Exception as exc:  # pragma: no cover - live diagnostic path
        torch_runtime = {
            "error": repr(exc),
            "installed_inside_venv": None,
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
        }

    driver_versions: list[str] = []
    driver_probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if driver_probe.returncode == 0:
        driver_versions = [line.strip() for line in driver_probe.stdout.splitlines() if line.strip()]
    torch_runtime["driver_versions"] = driver_versions

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - Windows development fallback
        cpu_count = os.cpu_count() or 0
    memory_gib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_gib = int(line.split()[1]) / 2**20
                break
    shm_gib = None
    shm_path = Path("/dev/shm")
    if shm_path.is_dir():
        stat = os.statvfs(shm_path)
        shm_gib = stat.f_frsize * stat.f_blocks / 2**30

    lock = json.loads((repo / "models.lock.json").read_text(encoding="utf-8"))
    source_spec = lock["sources"]["dreamlite_reference"]
    source_path = repo / source_spec["local_dir"]
    source = {
        "path": str(source_path),
        "expected_revision": source_spec["revision"],
        "actual_revision": run_git(source_path, "rev-parse", "HEAD") if source_path.exists() else None,
        "status": run_git(source_path, "status", "--short") if source_path.exists() else None,
    }

    models: dict[str, Any] = {}
    for name, specification in lock["models"].items():
        path = model_root / Path(specification["local_dir"]).name
        weight_files = sorted([*path.rglob("*.safetensors"), *path.rglob("*.bin")]) if path.exists() else []
        models[name] = {
            "path": str(path),
            "expected_revision": specification["revision"],
            "locked_revision": marker_value(path, ".locked_revision"),
            "snapshot_complete": marker_value(path, ".snapshot_complete"),
            "weight_file_count": len(weight_files),
            "weight_bytes": sum(item.stat().st_size for item in weight_files),
            "minimum_weight_bytes": int(int(specification.get("weight_bytes", specification.get("approx_bytes", 0))) * 0.85),
        }

    paths = {}
    for variable in ("VLM_MODEL_ROOT", "VLM_RUN_ROOT", "HF_HOME", "TORCH_HOME"):
        raw = os.environ.get(variable)
        path = Path(raw).expanduser() if raw else None
        paths[variable] = {
            "value": raw,
            "absolute": bool(path and path.is_absolute()),
            "exists": bool(path and path.exists()),
            "writable": bool(path and path.exists() and os.access(path, os.W_OK)),
        }

    return {
        "schema_version": 1,
        "profile": "inspire-r3-h200-ngc2502",
        "collected_at": utc_now(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "resources": {
            "cpu_affinity_count": cpu_count,
            "memory_gib": memory_gib,
            "shm_gib": shm_gib,
        },
        "git": {
            "root": str(repo),
            "commit": run_git(repo, "rev-parse", "HEAD"),
            "status": status,
            "clean": status == "",
        },
        "python": {
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_venv": sys.prefix != sys.base_prefix,
        },
        "packages": packages,
        "torch": torch_runtime,
        "environment": {key: os.environ.get(key) for key in SAFE_ENVIRONMENT_KEYS},
        "paths": paths,
        "dreamlite_source": source,
        "models": models,
    }


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def evaluate_inventory(
    inventory: dict[str, Any],
    *,
    expected_commit: str,
    expected_python: str,
    expected_torch: str,
    expected_cuda: str,
    expected_gpu_count: int,
    expected_gpu_name: str,
    min_gpu_memory_mib: int,
    expected_instance: str,
    expected_image: str,
    expected_node: str,
    expected_workspace: str,
    expected_project: str,
    expected_driver: str,
    min_cpus: int,
    min_memory_gib: float,
    min_shm_gib: float,
    require_models: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    git = inventory["git"]
    python = inventory["python"]
    torch_runtime = inventory["torch"]
    environment = inventory["environment"]

    add_check(checks, "git_commit", git.get("commit") == expected_commit, {"expected": expected_commit, "actual": git.get("commit")})
    add_check(checks, "git_clean", git.get("clean") is True, git.get("status"))
    add_check(checks, "python", python.get("major_minor") == expected_python, {"expected": expected_python, "actual": python.get("major_minor")})
    add_check(checks, "overlay_venv", python.get("in_venv") is True, python)
    add_check(
        checks,
        "system_torch_not_overlaid",
        torch_runtime.get("installed_inside_venv") is False,
        {"torch_file": torch_runtime.get("file"), "venv_purelib": torch_runtime.get("venv_purelib")},
    )
    add_check(
        checks,
        "torch_version",
        torch_runtime.get("runtime_version") == expected_torch and torch_runtime.get("distribution_version") == expected_torch,
        {
            "expected": expected_torch,
            "runtime": torch_runtime.get("runtime_version"),
            "distribution": torch_runtime.get("distribution_version"),
        },
    )
    add_check(checks, "cuda_available", torch_runtime.get("cuda_available") is True, torch_runtime)
    add_check(checks, "cuda_runtime", torch_runtime.get("cuda_runtime") == expected_cuda, {"expected": expected_cuda, "actual": torch_runtime.get("cuda_runtime")})
    add_check(
        checks,
        "driver_version",
        torch_runtime.get("driver_versions") == [expected_driver] * expected_gpu_count,
        {"expected": [expected_driver] * expected_gpu_count, "actual": torch_runtime.get("driver_versions")},
    )
    gpus = torch_runtime.get("gpus", [])
    add_check(checks, "gpu_count", torch_runtime.get("gpu_count") == expected_gpu_count, {"expected": expected_gpu_count, "actual": torch_runtime.get("gpu_count")})
    add_check(
        checks,
        "gpu_type",
        len(gpus) == expected_gpu_count and all(expected_gpu_name.casefold() in str(gpu.get("name", "")).casefold() for gpu in gpus),
        {"expected_substring": expected_gpu_name, "gpus": gpus},
    )
    add_check(
        checks,
        "gpu_memory",
        len(gpus) == expected_gpu_count and all(int(gpu.get("total_memory_mib", 0)) >= min_gpu_memory_mib for gpu in gpus),
        {"minimum_mib": min_gpu_memory_mib, "gpus": gpus},
    )

    for name, expected in EXPECTED_PACKAGES.items():
        actual = inventory["packages"].get(name)
        add_check(checks, f"package:{name}", actual == expected, {"expected": expected, "actual": actual})

    for key, expected in STRICT_ENVIRONMENT.items():
        add_check(checks, f"determinism:{key}", environment.get(key) == expected, {"expected": expected, "actual": environment.get(key)})
    for key, expected in (
        ("VLM_INSPIRE_INSTANCE", expected_instance),
        ("VLM_INSPIRE_IMAGE", expected_image),
        ("VLM_INSPIRE_NODE", expected_node),
        ("VLM_INSPIRE_WORKSPACE", expected_workspace),
        ("VLM_INSPIRE_PROJECT", expected_project),
    ):
        add_check(checks, f"runtime:{key}", environment.get(key) == expected, {"expected": expected, "actual": environment.get(key)})

    resources = inventory["resources"]
    add_check(checks, "cpu_allocation", int(resources.get("cpu_affinity_count") or 0) >= min_cpus, {"minimum": min_cpus, "actual": resources.get("cpu_affinity_count")})
    add_check(checks, "memory_allocation", float(resources.get("memory_gib") or 0.0) >= min_memory_gib, {"minimum_gib": min_memory_gib, "actual_gib": resources.get("memory_gib")})
    add_check(checks, "shm_allocation", float(resources.get("shm_gib") or 0.0) >= min_shm_gib, {"minimum_gib": min_shm_gib, "actual_gib": resources.get("shm_gib")})

    for variable, state in inventory["paths"].items():
        add_check(checks, f"path:{variable}", state["absolute"] and state["exists"] and state["writable"], state)

    source = inventory["dreamlite_source"]
    add_check(checks, "dreamlite_source_revision", source["actual_revision"] == source["expected_revision"], source)
    add_check(checks, "dreamlite_source_clean", source["status"] == "", source["status"])

    if require_models:
        for name, model in inventory["models"].items():
            add_check(
                checks,
                f"model:{name}",
                model["locked_revision"] == model["expected_revision"]
                and model["snapshot_complete"] == model["expected_revision"]
                and model["weight_file_count"] > 0
                and model["weight_bytes"] >= model["minimum_weight_bytes"],
                model,
            )

    passed = all(check["ok"] for check in checks)
    return {
        **inventory,
        "contract": {
            "expected_commit": expected_commit,
            "expected_python": expected_python,
            "expected_torch": expected_torch,
            "expected_cuda": expected_cuda,
            "expected_gpu_count": expected_gpu_count,
            "expected_gpu_name": expected_gpu_name,
            "min_gpu_memory_mib": min_gpu_memory_mib,
            "expected_instance": expected_instance,
            "expected_image": expected_image,
            "expected_node": expected_node,
            "expected_workspace": expected_workspace,
            "expected_project": expected_project,
            "expected_driver": expected_driver,
            "min_cpus": min_cpus,
            "min_memory_gib": min_memory_gib,
            "min_shm_gib": min_shm_gib,
            "models_required": require_models,
        },
        "checks": checks,
        "passed": passed,
        "formal_ready": passed and require_models,
    }


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for the locked Inspire R3 2xH200 runtime")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path, default=os.environ.get("VLM_MODEL_ROOT"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-python", default=os.environ.get("VLM_EXPECTED_PYTHON", "3.12"))
    parser.add_argument("--expected-torch", default=os.environ.get("VLM_EXPECTED_TORCH", "2.7.0a0+ecf3bae40a.nv25.02"))
    parser.add_argument("--expected-cuda", default=os.environ.get("VLM_EXPECTED_CUDA", "12.8"))
    parser.add_argument("--expected-gpu-count", type=int, default=int(os.environ.get("VLM_EXPECTED_GPU_COUNT", "2")))
    parser.add_argument("--expected-gpu-name", default=os.environ.get("VLM_EXPECTED_GPU_NAME", "H200"))
    parser.add_argument("--min-gpu-memory-mib", type=int, default=int(os.environ.get("VLM_MIN_GPU_MEMORY_MIB", "140000")))
    parser.add_argument("--expected-instance", default="vlm-r3-h200x2-live-20260717")
    parser.add_argument("--expected-image", default="ngc-pytorch:25.02-cuda12.8.0-py3")
    parser.add_argument("--expected-node", default="qb-prod-gpu2007")
    parser.add_argument("--expected-workspace", default="分布式训练空间")
    parser.add_argument("--expected-project", default="前沿课题探索")
    parser.add_argument("--expected-driver", default=os.environ.get("VLM_EXPECTED_DRIVER", "570.124.06"))
    parser.add_argument("--min-cpus", type=int, default=int(os.environ.get("VLM_MIN_CPUS", "40")))
    parser.add_argument("--min-memory-gib", type=float, default=float(os.environ.get("VLM_MIN_MEMORY_GIB", "390")))
    parser.add_argument("--min-shm-gib", type=float, default=float(os.environ.get("VLM_MIN_SHM_GIB", "120")))
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_root:
        parser.error("--model-root or VLM_MODEL_ROOT is required")
    if len(args.expected_commit) != 40 or any(character not in "0123456789abcdef" for character in args.expected_commit.lower()):
        parser.error("--expected-commit must be a full 40-character hexadecimal commit")

    inventory = collect_inventory(args.repo, Path(args.model_root))
    report = evaluate_inventory(
        inventory,
        expected_commit=args.expected_commit.lower(),
        expected_python=args.expected_python,
        expected_torch=args.expected_torch,
        expected_cuda=args.expected_cuda,
        expected_gpu_count=args.expected_gpu_count,
        expected_gpu_name=args.expected_gpu_name,
        min_gpu_memory_mib=args.min_gpu_memory_mib,
        expected_instance=args.expected_instance,
        expected_image=args.expected_image,
        expected_node=args.expected_node,
        expected_workspace=args.expected_workspace,
        expected_project=args.expected_project,
        expected_driver=args.expected_driver,
        min_cpus=args.min_cpus,
        min_memory_gib=args.min_memory_gib,
        min_shm_gib=args.min_shm_gib,
        require_models=args.require_models,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write(args.output, payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    atomic_write(args.output.with_suffix(args.output.suffix + ".sha256"), f"{digest}  {args.output.name}\n")
    print(payload, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
