from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(INSPIRE))

from launch_background import (  # noqa: E402
    validate_command,
    verify_preflight,
)
from install_non_torch_dependencies import (  # noqa: E402
    is_protected,
    requirement_applies,
    requirement_satisfied,
)
from packaging.requirements import Requirement  # noqa: E402
from poll_stage import stage_status  # noqa: E402
from preflight_r3_h200 import (  # noqa: E402
    EXPECTED_PACKAGES,
    STRICT_ENVIRONMENT,
    evaluate_inventory,
)


COMMIT = "a" * 40


def passing_inventory() -> dict:
    return {
        "schema_version": 1,
        "profile": "inspire-r3-h200-ngc2502",
        "git": {"commit": COMMIT, "clean": True, "status": ""},
        "python": {
            "major_minor": "3.12",
            "in_venv": True,
            "version": "3.12.3",
            "executable": "/env/bin/python",
            "prefix": "/env",
            "base_prefix": "/usr/local",
        },
        "packages": {**EXPECTED_PACKAGES, "torch": "2.7.0a0+ecf3bae40a.nv25.02"},
        "torch": {
            "runtime_version": "2.7.0a0+ecf3bae40a.nv25.02",
            "distribution_version": "2.7.0a0+ecf3bae40a.nv25.2",
            "file": "/usr/local/lib/python3.12/site-packages/torch/__init__.py",
            "venv_purelib": "/env/lib/python3.12/site-packages",
            "installed_inside_venv": False,
            "cuda_available": True,
            "cuda_runtime": "12.8",
            "gpu_count": 2,
            "driver_versions": ["570.124.06", "570.124.06"],
            "gpus": [
                {"index": index, "name": "NVIDIA H200", "capability": [9, 0], "total_memory_mib": 143771}
                for index in range(2)
            ],
        },
        "environment": {
            **STRICT_ENVIRONMENT,
            "VLM_INSPIRE_INSTANCE": "vlm-r3-h200x2-live-20260717",
            "VLM_INSPIRE_IMAGE": "ngc-pytorch:25.02-cuda12.8.0-py3",
            "VLM_INSPIRE_NODE": "qb-prod-gpu2007",
            "VLM_INSPIRE_WORKSPACE": "分布式训练空间",
            "VLM_INSPIRE_PROJECT": "前沿课题探索",
        },
        "resources": {"cpu_affinity_count": 40, "memory_gib": 400.0, "shm_gib": 128.0},
        "paths": {
            name: {"value": f"/project/{name}", "absolute": True, "exists": True, "writable": True}
            for name in ("VLM_MODEL_ROOT", "VLM_RUN_ROOT", "HF_HOME", "TORCH_HOME")
        },
        "dreamlite_source": {
            "expected_revision": "d" * 40,
            "actual_revision": "d" * 40,
            "status": "",
        },
        "models": {
            name: {
                "expected_revision": revision,
                "locked_revision": revision,
                "snapshot_complete": revision,
                "weight_file_count": 2,
                "weight_bytes": 100,
                "minimum_weight_bytes": 85,
            }
            for name, revision in (("dreamlite_mobile", "e" * 40), ("qwen_reader", "f" * 40))
        },
    }


def evaluate(inventory: dict, *, require_models: bool = True) -> dict:
    return evaluate_inventory(
        inventory,
        expected_commit=COMMIT,
        expected_python="3.12",
        expected_torch="2.7.0a0+ecf3bae40a.nv25.02",
        expected_cuda="12.8",
        expected_gpu_count=2,
        expected_gpu_name="H200",
        min_gpu_memory_mib=140000,
        expected_instance="vlm-r3-h200x2-live-20260717",
        expected_image="ngc-pytorch:25.02-cuda12.8.0-py3",
        expected_node="qb-prod-gpu2007",
        expected_workspace="分布式训练空间",
        expected_project="前沿课题探索",
        expected_driver="570.124.06",
        min_cpus=40,
        min_memory_gib=390.0,
        min_shm_gib=120.0,
        require_models=require_models,
    )


class InspirePreflightTest(unittest.TestCase):
    def test_complete_h200_inventory_is_formal_ready(self):
        report = evaluate(passing_inventory())
        self.assertTrue(report["passed"])
        self.assertTrue(report["formal_ready"])
        self.assertTrue(all(check["ok"] for check in report["checks"]))

    def test_ngc_torch_overlay_or_incomplete_gpu_fails_closed(self):
        inventory = passing_inventory()
        inventory["torch"]["installed_inside_venv"] = True
        inventory["torch"]["gpu_count"] = 1
        inventory["torch"]["gpus"] = inventory["torch"]["gpus"][:1]
        inventory["resources"]["shm_gib"] = 64.0
        report = evaluate(inventory)
        self.assertFalse(report["passed"])
        failures = {check["name"] for check in report["checks"] if not check["ok"]}
        self.assertIn("system_torch_not_overlaid", failures)
        self.assertIn("gpu_count", failures)
        self.assertIn("gpu_type", failures)
        self.assertIn("shm_allocation", failures)

    def test_environment_preflight_cannot_authorize_science(self):
        report = evaluate(passing_inventory(), require_models=False)
        self.assertTrue(report["passed"])
        self.assertFalse(report["formal_ready"])

    def test_model_marker_or_strict_environment_drift_fails(self):
        inventory = passing_inventory()
        inventory["models"]["qwen_reader"]["snapshot_complete"] = None
        inventory["environment"]["CUBLAS_WORKSPACE_CONFIG"] = None
        report = evaluate(inventory)
        self.assertFalse(report["passed"])
        failures = {check["name"] for check in report["checks"] if not check["ok"]}
        self.assertIn("model:qwen_reader", failures)
        self.assertIn("determinism:CUBLAS_WORKSPACE_CONFIG", failures)


class InspireBackgroundStageTest(unittest.TestCase):
    def make_repo(self, root: Path) -> tuple[Path, str]:
        repo = root / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return repo, commit

    def write_preflight(self, path: Path, commit: str, *, formal_ready: bool) -> None:
        payload = json.dumps(
            {"passed": True, "formal_ready": formal_ready, "git": {"commit": commit}},
            indent=2,
            sort_keys=True,
        ) + "\n"
        path.write_bytes(payload.encode("utf-8"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_bytes(f"{digest}  {path.name}\n".encode("utf-8"))

    def test_preflight_hash_and_formal_readiness_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            self.write_preflight(preflight, COMMIT, formal_ready=False)
            with self.assertRaisesRegex(ValueError, "model-complete"):
                verify_preflight(preflight, expected_commit=COMMIT, infrastructure_stage=False)
            verify_preflight(preflight, expected_commit=COMMIT, infrastructure_stage=True)
            preflight.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_preflight(preflight, expected_commit=COMMIT, infrastructure_stage=True)

    def test_recorded_command_rejects_secrets(self):
        with self.assertRaisesRegex(ValueError, "Secrets"):
            validate_command(["python", "fetch.py", "--token=hf_example"])
        with self.assertRaisesRegex(ValueError, "Secrets"):
            validate_command(["python", "fetch.py", "hf_example"])
        validate_command(["python", "train.py", "--target-token-count", "8"])

    def test_detached_stage_writes_terminal_sentinel_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, commit = self.make_repo(root)
            preflight = root / "preflight.json"
            self.write_preflight(preflight, commit, formal_ready=True)
            run_root = root / "runs"
            run_dir = run_root / "r3-s0"
            result = subprocess.run(
                [
                    sys.executable,
                    str(INSPIRE / "launch_background.py"),
                    "--repo",
                    str(repo),
                    "--run-root",
                    str(run_root),
                    "--run-dir",
                    str(run_dir),
                    "--stage",
                    "r3-s0",
                    "--expected-commit",
                    commit,
                    "--preflight",
                    str(preflight),
                    "--",
                    sys.executable,
                    "-c",
                    "print('stage-ok')",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            launch = json.loads(result.stdout)
            self.assertEqual(launch["status"], "started")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not (run_dir / "terminal.json").is_file():
                time.sleep(0.05)
            report, exit_code = stage_status(run_dir)
            self.assertEqual(exit_code, 0, report)
            self.assertEqual(report["status"], "succeeded")
            self.assertEqual((run_dir / "stdout.log").read_text(encoding="utf-8").strip(), "stage-ok")
            terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["stdout_sha256"],
                hashlib.sha256((run_dir / "stdout.log").read_bytes()).hexdigest(),
            )

            reused = subprocess.run(
                [
                    sys.executable,
                    str(INSPIRE / "launch_background.py"),
                    "--repo",
                    str(repo),
                    "--run-root",
                    str(run_root),
                    "--run-dir",
                    str(run_dir),
                    "--stage",
                    "r3-s0",
                    "--expected-commit",
                    commit,
                    "--preflight",
                    str(preflight),
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(reused.returncode, 0)


class InspireStaticContractTest(unittest.TestCase):
    def test_dependency_closure_protects_ngc_cuda_stack(self):
        self.assertTrue(is_protected("torch"))
        self.assertTrue(is_protected("nvidia-cublas-cu12"))
        self.assertTrue(requirement_satisfied(Requirement("torch>=1.13"), "2.7.0a0+ecf.nv25.02"))
        self.assertFalse(requirement_applies(Requirement("requests; extra == 'http'")))

    def test_bootstrap_cannot_install_torch_and_has_no_fudan_submission(self):
        bootstrap = (INSPIRE / "bootstrap_r3_h200.sh").read_text(encoding="utf-8")
        dependency_installer = (INSPIRE / "install_non_torch_dependencies.py").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements" / "inspire-ngc2502-pinned.txt").read_text(encoding="utf-8")
        self.assertIn("--system-site-packages", bootstrap)
        self.assertIn("--no-deps", bootstrap)
        self.assertIn("PIP_CONFIG_FILE=/dev/null", bootstrap)
        self.assertIn("PIP_EXTRA_INDEX_URL=", bootstrap)
        self.assertIn('"PIP_CONFIG_FILE": "/dev/null"', dependency_installer)
        self.assertIn('"PIP_EXTRA_INDEX_URL": ""', dependency_installer)
        self.assertIn("install_non_torch_dependencies.py", bootstrap)
        self.assertIn("Torch fingerprint changed", bootstrap)
        self.assertNotIn("sbatch", bootstrap)
        package_lines = [line.strip().lower() for line in requirements.splitlines() if line.strip() and not line.startswith("#")]
        self.assertFalse(any(line.startswith(("torch==", "torchvision==", "torchaudio==")) for line in package_lines))


if __name__ == "__main__":
    unittest.main()
