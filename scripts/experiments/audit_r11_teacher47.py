"""Read canonical teachers on the pinned GPU, or independently audit downloaded receipts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_teacher_swap as core  # noqa: E402


def validate_teacher(config: dict, target: dict) -> tuple[dict, dict]:
    folder = Path(config["canonical_root"]) / f"target-{target['target_index']:02d}" / "run"
    endpoint_path, manifest_path = folder / "endpoint_raw.pt", folder / "manifest.json"
    core.require(core.sha256_file(endpoint_path) == target["endpoint_sha256"], "Teacher checkpoint hash drift.")
    core.require(core.sha256_file(manifest_path) == target["manifest_sha256"], "Teacher manifest hash drift.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core.require(manifest["schema"] == "vision_memory.r11-vae-latent-oracle-manifest.v1"
                 and manifest["target_index"] == target["target_index"]
                 and manifest["target_segment"] == target["target_segment"]
                 and manifest["selected_segments_sha256"] == config["selected_segments_sha256"],
                 "Teacher target provenance drift.")
    core.require(manifest["model_snapshot_payloads_start"] == config["model_snapshots"], "Teacher model drift.")
    payload = torch.load(endpoint_path, map_location="cpu", weights_only=True)
    core.require(payload["schema"] == "vision_memory.r11-vae-latent-endpoint.v1"
                 and payload["manifest_sha256"] == target["manifest_sha256"], "Wrong endpoint schema/manifest.")
    latent, image = payload["latent_fp32"], payload["image"]
    core.require(latent.dtype == torch.float32 and list(latent.shape) == [1, 4, 128, 128]
                 and bool(torch.isfinite(latent).all()), "Invalid teacher latent.")
    core.require(image.dtype == torch.bfloat16 and list(image.shape) == [1, 3, 1024, 1024]
                 and bool(torch.isfinite(image).all()) and float(image.min()) >= 0 and float(image.max()) <= 1,
                 "Invalid archived teacher RGB.")
    return payload, {"endpoint": str(endpoint_path), "endpoint_sha256": target["endpoint_sha256"],
                     "manifest": str(manifest_path), "manifest_sha256": target["manifest_sha256"]}


def validate_environment(args: argparse.Namespace, config: dict) -> dict:
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    audit = safety._deployment_audit(args.output_root)
    core.require(bool(safety._COMMIT_RE.fullmatch(args.expected_commit or "")), "A full expected Git SHA is required.")
    core.require(safety._git("rev-parse", "HEAD") == args.expected_commit
                 and not safety._git("status", "--porcelain") and not safety._git("branch", "--show-current"),
                 "Expected clean detached checkout required.")
    observed = {key: os.environ.get(key) for key in config["environment"]}
    core.require(observed == config["environment"], "Offline/deterministic environment drift.")
    core.require(torch.__version__.startswith("2.7.0a0+ecf3bae40a") and torch.version.cuda == "12.8"
                 and sys.version_info[:2] == (3, 12), "Pinned NGC runtime drift; do not replace Torch.")
    core.require(torch.cuda.is_available() and torch.cuda.device_count() == 2
                 and all("H200" in torch.cuda.get_device_name(i).upper() for i in range(2))
                 and torch.cuda.is_bf16_supported(), "Exactly two native BF16 H200 GPUs required.")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                     text=True).splitlines()
    core.require(all(line.strip() in ("", str(os.getpid())) for line in active), "GPU is occupied; do not interrupt it.")
    locks = list(Path("/tmp").glob("*r11*lock*"))
    core.require(not locks, f"Existing R11 lock; never remove it automatically: {locks}")
    core.require(core.sha256_file(Path(config["train"]["path"])) == config["train"]["sha256"], "Training data drift.")
    return {"deployment": audit, "environment": observed, "git_commit": args.expected_commit,
            "git_clean": True, "git_detached": True, "other_compute_pids": []}


def run_gpu(args: argparse.Namespace, config: dict, validation: dict, root: Path) -> dict:
    from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism

    atomic = phase1a._atomic_json
    bindings = {}
    teachers = {}
    for target in config["targets"]:
        payload, binding = validate_teacher(config, target)
        key = str(target["target_index"])
        teachers[key], bindings[key] = payload, binding
    atomic(root / "config.json", config)
    atomic(root / "input-bindings.json", bindings)
    phase1a._write_environment(root / "environment.txt")
    atomic(root / "runtime.json", phase1a._runtime_versions())
    atomic(root / "determinism.json", configure_strict_cuda_determinism(0))
    snapshots = {name: phase1a.verify_snapshot_binding(value) for name, value in config["model_snapshots"].items()}
    atomic(root / "snapshots-start.json", snapshots)
    counters = {"unet_forward_calls": 0, "reader_forward_calls": 0, "optimizer_steps": 0}

    def forbidden_unet(_module: Any, _inputs: Any) -> None:
        counters["unet_forward_calls"] += 1
        raise RuntimeError("This is a VAE/Reader audit, not a U-Net experiment.")

    def count_reader(_module: Any, _inputs: Any) -> None:
        counters["reader_forward_calls"] += 1

    load_args = argparse.Namespace(dreamlite=config["models"]["dreamlite"], reader=config["models"]["reader"])
    started = time.monotonic()
    try:
        pipe = phase1a._load_pipeline(load_args, torch.device("cuda:0"), torch.bfloat16)
        processor, reader = phase1a._load_reader(load_args, torch.device("cuda:1"), torch.bfloat16)
        modules = [v for v in pipe.components.values() if isinstance(v, torch.nn.Module)] + [reader]

        def frozen() -> bool:
            return all(not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters())
                       for module in modules)

        core.require(frozen(), "All pipeline and Reader parameters must be frozen.")
        unet_hook = pipe.unet.register_forward_pre_hook(forbidden_unet)
        # The scorer intentionally calls ``reader.model(...)`` and then
        # ``reader.lm_head(...)`` rather than ``reader(...)``.  Count the
        # module that actually executes each candidate forward.
        reader_hook = reader.model.register_forward_pre_hook(count_reader)
        try:
            with torch.no_grad():
                images = {key: decode_model_latents_unit_interval(pipe.vae,
                          teachers[key]["latent_fp32"].to(device="cuda:0", dtype=torch.bfloat16), clamp=True)
                          for key in ("4", "7")}
                replay_equal = {key: torch.equal(images[key].cpu(), teachers[key]["image"]) for key in images}
                atomic(root / "teacher-rgb-replay.json", replay_equal)
                core.require(all(replay_equal.values()), "Decoded RGB differs from canonical artifact; technical stop.")
                images["reset"] = phase1a.blank_source_rgb(device=torch.device("cuda:0"), dtype=torch.bfloat16)
                phase1a._atomic_torch_save(root / "images.pt", {k: v.cpu() for k, v in images.items()})
                phase1a._atomic_torch_save(root / "teacher-latents.pt",
                                           {k: v["latent_fp32"] for k, v in teachers.items()})
                hashes = {k: canonical_tensor_sha256(v) for k, v in images.items()}
                for key, value in images.items():
                    phase1a._save_image(root / f"image-{key}.png", value)
                manifest = {"schema": f"{core.PREFIX}-manifest.v1", "protocol": core.PROTOCOL,
                    "config_sha256": core.canonical_sha(config), "git_commit": args.expected_commit,
                    "script_sha256": core.sha256_file(Path(__file__)), "validation": validation,
                    "inputs": bindings, "image_hashes": hashes, "teacher_rgb_bitwise_replay": replay_equal,
                    "model_snapshots": snapshots, "reader_metric": "teacher-forced mean-token-NLL listwise CE",
                    "unet_used": False, "trainable_parameters": [], "formal_success": False, "phase2_allowed": False}
                atomic(root / "manifest.json", manifest)
                reader_fn = phase1a.r8.choice_reader_callable(reader=reader, processor=processor,
                              reader_device=torch.device("cuda:1"), require_grad=False, deterministic_ce=True)
                rows = []
                with (root / "receipts.jsonl").open("x", encoding="utf-8") as receipts:
                    for target in config["targets"]:
                        index, segment = target["target_index"], target["target_segment"]
                        query = segment["query"]
                        for condition in config["conditions"]:
                            image_key = core.expected_image(index, condition)
                            image = images[image_key].to("cuda:1")
                            actual_hash = canonical_tensor_sha256(image)
                            core.require(actual_hash == hashes[image_key], "Image transfer changed the actual input.")
                            for view, permutation in enumerate(config["permutations"]):
                                ordered = tuple(query["choices"][i] for i in permutation)
                                output = reader_fn(image, phase1a.r5.format_mcq_query(query["text"], ordered),
                                                   ordered, permutation.index(query["target_index"]))
                                logits = output.choice_logits.detach().float().cpu().tolist()
                                derived = core.recompute(logits, permutation.index(query["target_index"]), permutation)
                                core.require(abs(float(output.loss) - derived["ce"]) <= 1e-6, "Reader CE mismatch.")
                                row = {"schema": f"{core.PREFIX}-row.v1", "target_index": index,
                                    "segment_id": segment["segment_id"], "query_sha256": core.canonical_sha(query),
                                    "condition": condition, "view_index": view, "permutation": permutation,
                                    "original_answer_index": query["target_index"], "image_key": image_key,
                                    "image_sha256": actual_hash, "choice_logits_ordered": logits, **derived}
                                receipts.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                                receipts.flush()
                                os.fsync(receipts.fileno())
                                rows.append(row)
                                print(f"receipt {len(rows)}/24 target={index} condition={condition} view={view}", flush=True)
                core.require(frozen(), "Frozen-parameter contract changed.")
        finally:
            unet_hook.remove()
            reader_hook.remove()
        comparison = core.aggregate(rows, config, hashes)
        expected_counters = {"unet_forward_calls": config["guardrails"]["unet_forward_calls"],
            "reader_forward_calls": config["expected_reader_forward_calls"],
            "optimizer_steps": config["guardrails"]["optimizer_steps"]}
        atomic(root / "execution-counts.json", {"observed": counters, "expected": expected_counters})
        core.require(counters == expected_counters, f"Unexpected forward counts: {counters} != {expected_counters}.")
        # Re-hash parents after scoring; originals must not be rewritten.
        for target in config["targets"]:
            validate_teacher(config, target)
    finally:
        end = {name: phase1a.verify_snapshot_binding(value) for name, value in config["model_snapshots"].items()}
        atomic(root / "snapshots-end.json", end)
    core.require(end == snapshots, "Model snapshots changed.")
    result = {"schema": f"{core.PREFIX}-result.v1", "protocol": core.PROTOCOL,
        "engineering_gate": True, "git_commit": args.expected_commit, "counters": counters,
        "snapshots_unchanged": True, "all_parameters_frozen": True, "comparison": comparison,
        "formal_success": False, "phase2_allowed": False, "elapsed_seconds": time.monotonic() - started}
    atomic(root / "result.json", result)
    lines = ["# Canonical teacher 4 ↔ 7 读取审计", "", "工程通过不等于训练成功。本轮没有参数更新。", "",
             "| Target | Own CE / acc | Donor CE / acc | Reset CE / acc | 区分力 |",
             "| --- | --- | --- | --- | --- |"]
    for target in comparison["targets"]:
        cells = [f"{target['scores'][c]['mean_ce']:.8g} / {target['scores'][c]['accuracy']:.0%}"
                 for c in config["conditions"]]
        lines.append(f"| {target['target_index']} | " + " | ".join(cells) + f" | {target['distinguishability_gate']} |")
    lines += ["", f"决策：`{comparison['decision']}`。", "", "不开放 Phase 2，不证明共享训练、事件因果性或长期记忆。",
              "", f"代码：`{args.expected_commit}`。24 行原始 logits 必须经独立 audit 模式复算后才允许启用下游。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    config = core.load_config()
    if args.mode == "audit":
        print(json.dumps(core.audit_delivery(args.output_root, config), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety
    from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a

    validation = validate_environment(args, config)
    lock_path = Path("/tmp/vision-memory-r11-new-teacher47-swap.lock")
    lock_path.mkdir(exist_ok=False)
    owner = {"owner_token": uuid.uuid4().hex, "pid": os.getpid(), "git_commit": args.expected_commit,
             "output_root": str(args.output_root), "started_at_utc": safety._utc_now()}
    lock = {"path": str(lock_path), "owner_path": str(lock_path / "owner.json"), "owner": owner}
    claimed = False
    terminal = {"schema": f"{core.PREFIX}-terminal.v1", "protocol": core.PROTOCOL, "status": "technical_failed",
                "engineering_gate": False, "exit_code": 2, "git_commit": args.expected_commit,
                "formal_success": False, "phase2_allowed": False, "started_at_utc": safety._utc_now()}
    try:
        phase1a._atomic_json(lock_path / "owner.json", owner)
        lock["owner_sha256"] = core.sha256_file(lock_path / "owner.json")
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        phase1a._atomic_json(args.output_root / "launch.json", {"owner": owner, "validation": validation})
        with (args.output_root / "stdout.log").open("x", encoding="utf-8") as out, (
            args.output_root / "stderr.log").open("x", encoding="utf-8") as err, (
            contextlib.redirect_stdout(out)), contextlib.redirect_stderr(err):
            try:
                run_gpu(args, config, validation, args.output_root)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(status="completed", engineering_gate=True, exit_code=0,
                        manifest_sha256=core.sha256_file(args.output_root / "manifest.json"),
                        result_sha256=core.sha256_file(args.output_root / "result.json"))
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            terminal["lock_release"] = safety._release_lock(lock)
        except Exception as error:
            terminal.update(status="technical_failed", engineering_gate=False, exit_code=2,
                            lock_release_error=f"{type(error).__name__}: {error}")
        if claimed:
            terminal["completed_at_utc"] = safety._utc_now()
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            artifacts = [{"path": p.relative_to(args.output_root).as_posix(), "bytes": p.stat().st_size,
                          "sha256": core.sha256_file(p)} for p in sorted(args.output_root.rglob("*")) if p.is_file()]
            phase1a._atomic_json(args.output_root / "artifact_inventory.json",
                {"schema": f"{core.PREFIX}-inventory.v1", "artifact_count": len(artifacts), "artifacts": artifacts})
            if terminal["exit_code"] == 0:
                # This check does not change the inventory; a separate process must repeat it before D1.
                core.audit_delivery(args.output_root, config)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
