"""Migrate the zero-update warm-start experiment to four idle H200s."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= 0:
        raise ValueError("A finite positive deadline is required")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
        raise ValueError("Require the exact clean parallel implementation")
    from scripts.experiments.transition_warm_start_plan import plan
    from vision_memory.dreamlite.writer_package import inspect_package, file_sha
    project = Path("/inspire/ssd/project/exploration-topic/czxs26210936")
    runs = project / "runs/dreamlite-official-alignment"
    models = Path("/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory")
    original = ROOT / "reports/official-transition-warm-start-plan-20260913.json"
    design = plan()
    if json.loads(original.read_bytes()) != design:
        raise ValueError("Original fixed experiment plan changed")
    old_run = runs / "d9a1a11-transition-warm2880-seed20260914"
    old_terminal = json.loads((old_run / "train/terminal.json").read_bytes())
    if old_terminal.get("status") != "paused" or old_terminal.get("optimizer_steps") != 0 or old_terminal.get("checkpoint") is not None:
        raise ValueError("Migration expects the safely paused zero-update serial run")
    if list((old_run / "train").glob("checkpoint-*.pt")):
        raise ValueError("Unexpected old optimizer checkpoint: preserve and investigate before migration")
    package = runs / "281b653-transition-writer-package"
    manifest = inspect_package(package)
    for key, expected in {"parent_checkpoint_sha256": design["parent_checkpoint_sha256"],
        "parent_result_sha256": design["parent_result_sha256"], "optimizer_steps": 2880,
        "training_bank_sha256": design["bank_sha256"], "conditional_group_count": 45,
        "semantic_question_count": 1, "guidance_scale": 1.}.items():
        if manifest.get(key) != expected:
            raise ValueError("Initial package differs: " + key)
    manifest_sha = file_sha(package / "manifest.json")
    parity_path = runs / "281b653-transition-package-parity/parity-result.json"
    parity = json.loads(parity_path.read_bytes())
    if (parity.get("parity_pass") is not True or parity.get("package_manifest_sha256") != manifest_sha
            or parity.get("writes_compared") != 6 or parity.get("reads_compared") != 30):
        raise ValueError("Initial package lacks actual six-write/thirty-read parity")
    if file_sha(runs / "9628d71-transition-wording-bank/manifest.json") != design["bank_sha256"]:
        raise ValueError("Training bank changed")
    output = runs / (a.expected_commit[:7] + "-transition-warm2880-h200x4-seed20260914")
    amendment = {"original_plan_sha256": file_sha(original), "original_training_commit": design["training_commit"],
        "training_commit": a.expected_commit, "instance": "vlm-r11-trust-h200x4-20260907-r3",
        "world_size": 4, "global_microbatches": 4, "microbatches_per_rank": 1,
        "same_global_draw_indices": True, "optimizer": design["optimizer"],
        "serial_numerical_equivalence": "Same draws/objective; FP32 collective reduction order can change rounding",
        "prerequisite": "actual full-U-Net serial/parallel gradient check before any optimizer update",
        "old_run": str(old_run), "old_terminal_sha256": file_sha(old_run / "train/terminal.json"),
        "old_optimizer_steps": 0, "old_partial_baseline": "retained; complete baseline independently rerun on four GPUs",
        "data_and_endpoint_change": "none", "additional_updates": 2880, "new_seed": 20260914,
        "initial_package_manifest_sha256": manifest_sha, "initial_package_parity_sha256": file_sha(parity_path),
        "validation": "original warm-start preregistration unchanged; no validation examples enter training"}
    command = [sys.executable, "-u", str(ROOT / "scripts/inspire/run_official_alignment_pilot.py"),
        "--bank-manifest", str(runs / "9628d71-transition-wording-bank/manifest.json"), "--bank-sha256", design["bank_sha256"],
        "--dreamlite", str(models / "DreamLite-base-a9a0f15-20260907"), "--model-variant", "base",
        "--teacher-dreamlite", str(models / "DreamLite-mobile"), "--official-source", str(project / "Vision-Language-Memory/third_party/DreamLite"),
        "--base-manifest", str(runs / "base-complete-snapshot-seal.json"), "--base-guidance-scale", "1",
        "--reader", str(models / "Qwen3-VL-4B-Instruct"), "--output-dir", str(output), "--expected-commit", a.expected_commit,
        "--target-mode", "single", "--steps", "2880", "--eval-seeds", "4", "--trainable-scope", "full_unet",
        "--checkpoint-interval", "16", "--colocate-models", "--data-parallel-world-size", "4",
        "--seed", "20260914", "--deadline-unix", str(a.deadline_unix),
        "--initial-writer-package", str(package), "--initial-writer-package-sha256", manifest_sha]
    if a.resume:
        command.append("--resume")
    print(json.dumps({"amendment": amendment, "command": command}), flush=True)
    if a.dry_run:
        return 0
    if a.deadline_unix - time.time() < 90 * 60:
        raise ValueError("Require at least ninety minutes for the fixed four-GPU experiment")
    if a.resume:
        if json.loads((output / "migration-amendment.json").read_bytes()) != amendment:
            raise ValueError("Resume amendment changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "preregistered-experiment.json").write_bytes(original.read_bytes())
        (output / "migration-amendment.json").write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
