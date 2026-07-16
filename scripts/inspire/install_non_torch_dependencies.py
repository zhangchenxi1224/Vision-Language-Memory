from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


DEFAULT_ROOTS = (
    "accelerate",
    "diffusers",
    "einops",
    "huggingface-hub",
    "numpy",
    "peft",
    "pillow",
    "pytest",
    "pyyaml",
    "ruff",
    "safetensors",
    "tokenizers",
    "tqdm",
    "transformers",
)
PROTECTED_EXACT = {"torch", "torchaudio", "torchvision", "triton"}


def is_protected(name: str) -> bool:
    canonical = canonicalize_name(name)
    return canonical in PROTECTED_EXACT or canonical.startswith("nvidia-")


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def requirement_applies(requirement: Requirement) -> bool:
    if requirement.marker is None:
        return True
    environment = default_environment()
    environment["extra"] = ""
    return requirement.marker.evaluate(environment)


def requirement_satisfied(requirement: Requirement, actual: str | None) -> bool:
    if actual is None:
        return False
    if requirement.url:
        return True
    if not requirement.specifier:
        return True
    return requirement.specifier.contains(Version(actual), prereleases=True)


def dependency_requirements(name: str) -> list[Requirement]:
    distribution = importlib.metadata.distribution(name)
    requirements = []
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement_applies(requirement):
            requirements.append(requirement)
    return requirements


def install_requirement(requirement: Requirement, *, index_url: str, trusted_host: str) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--index-url",
        index_url,
        "--trusted-host",
        trusted_host,
        str(requirement),
    ]
    subprocess.run(command, check=True)


def close_dependencies(
    roots: tuple[str, ...],
    *,
    index_url: str,
    trusted_host: str,
    max_installs: int,
) -> dict:
    pending = deque(canonicalize_name(name) for name in roots)
    visited: set[str] = set()
    installed: list[dict[str, str | None]] = []
    protected_overrides: list[dict[str, str]] = []

    while pending:
        name = pending.popleft()
        if name in visited:
            continue
        if distribution_version(name) is None:
            raise RuntimeError(f"Root or queued distribution is missing: {name}")
        visited.add(name)
        for requirement in dependency_requirements(name):
            dependency = canonicalize_name(requirement.name)
            actual = distribution_version(dependency)
            if is_protected(dependency):
                if actual is None:
                    raise RuntimeError(f"Protected NGC distribution is missing: {dependency}")
                if not requirement_satisfied(requirement, actual):
                    protected_overrides.append(
                        {"parent": name, "requirement": str(requirement), "actual": actual}
                    )
                continue
            if not requirement_satisfied(requirement, actual):
                if requirement.url:
                    raise RuntimeError(f"Refusing an unpinned direct dependency URL: {requirement}")
                if len(installed) >= max_installs:
                    raise RuntimeError(f"Dependency closure exceeded --max-installs={max_installs}")
                before = actual
                install_requirement(requirement, index_url=index_url, trusted_host=trusted_host)
                actual = distribution_version(dependency)
                if not requirement_satisfied(requirement, actual):
                    raise RuntimeError(
                        f"Dependency remains unsatisfied after installation: {requirement}; actual={actual}"
                    )
                installed.append(
                    {
                        "parent": name,
                        "requirement": str(requirement),
                        "before": before,
                        "after": actual,
                    }
                )
                visited.discard(dependency)
            if dependency not in visited:
                pending.append(dependency)

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "roots": list(roots),
        "visited": sorted(visited),
        "installed": installed,
        "protected_version_overrides": protected_overrides,
        "protected_distributions": sorted(PROTECTED_EXACT),
        "passed": True,
    }


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close non-Torch dependencies using only --no-deps while preserving the NGC CUDA stack"
    )
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument(
        "--index-url",
        default=os.environ.get(
            "VLM_PIP_INDEX_URL",
            "http://nexus.sii.shaipower.online/repository/pypi/simple/",
        ),
    )
    parser.add_argument(
        "--trusted-host",
        default=os.environ.get("VLM_PIP_TRUSTED_HOST", "nexus.sii.shaipower.online"),
    )
    parser.add_argument("--max-installs", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = tuple(args.roots or DEFAULT_ROOTS)
    if any(is_protected(name) for name in roots):
        parser.error("Protected NGC distributions cannot be dependency-closure roots")
    report = close_dependencies(
        roots,
        index_url=args.index_url,
        trusted_host=args.trusted_host,
        max_installs=args.max_installs,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write(args.output, payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
