"""Fixed CFG=1 control on an immutable completed Base checkpoint, without training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256
    command = json.loads((a.run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    if args.model_variant != "base" or args.flow_protocol != "official":
        raise ValueError("This control requires the official Base protocol")
    bank, teachers = load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, str(Path(__file__).resolve()),
            "--run", str(a.run), "--output", str(a.output), "--worker"], env=env)
    terminal = json.loads((a.run / "terminal.json").read_text())
    if terminal["state"] != "completed" or file_sha256(a.run / "train/result.json") != terminal["training_result_sha256"]:
        raise RuntimeError("A verified completed parent run is required")
    result = json.loads((a.run / "train/result.json").read_text())
    checkpoint = a.run / "train/checkpoint-final.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Probe source must be immutable")
    a.output.mkdir(parents=True, exist_ok=False)
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    configure_strict_cuda_determinism(args.seed)
    runtime = train.load_runtime(args, bank)
    parent_runtime = json.loads((a.run / "train/runtime.json").read_text())
    if runtime["snapshots"] != parent_runtime["snapshots"] or runtime.get("protocol_binding", {}) != parent_runtime["additional_protocol_binding"]:
        raise RuntimeError("Control model or source identity differs from parent")
    if {k: canonical_tensor_sha256(v["condition"].prompt_embeds) for k,v in runtime["contexts"].items()} != parent_runtime["condition_sha256"]:
        raise RuntimeError("Training condition differs from parent")
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime["pipe"].unet)
    if loaded["optimizer_step"] != args.steps:
        raise RuntimeError("Parent budget does not match checkpoint")
    for context in runtime["contexts"].values():
        context["inference_sampler"].guidance_scale = 1.0
    args.output_dir = a.output
    identity = {"probe_commit": commit, "parent_result_sha256": terminal["training_result_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"], "guidance_scale": 1.,
        "image_guidance_scale": 1., "inference_steps": 28, "seed": args.seed,
        "scope": "Same checkpoint, noise, native prompt, image and Reader; only CFG changes 7.5 to 1"}
    train.write_json(a.output / "identity.json", identity)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    summary = train.evaluate(args, runtime, bank, teachers, "guidance1")
    train.frozen_audit(runtime["pipe"], runtime["reader"], frozen)
    runtime["verify_additional_bindings"]()
    train.write_json(a.output / "complete.json", {"identity": identity, "summary": summary,
        "phase_complete_sha256": file_sha256(a.output / "guidance1/complete.json")})
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
