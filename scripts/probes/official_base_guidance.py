"""Frozen Base checkpoint controls for native CFG and cached training conditions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


class TrainingConditionNativeSampler:
    """Use the exact cached raw-event condition inside the upstream CFG=1 loop.

    All three condition rows receive the same cached tensors. The official
    source branches and CFG arithmetic remain untouched; at both scales=1,
    the conditional text/source prediction is selected algebraically.
    """
    def __init__(self, native_sampler, condition):
        self.native = native_sampler
        self.condition = condition
        if self.native.guidance_scale != 1.0:
            raise ValueError("Training-condition control requires guidance_scale=1")
        if condition.prompt_embeds.ndim != 3 or condition.prompt_embeds.shape[0] != 1:
            raise ValueError("Expected one cached training condition")
        if condition.attention_mask.shape != condition.prompt_embeds.shape[:2]:
            raise ValueError("Cached condition mask shape differs")

    def __call__(self, **kwargs):
        pipe = self.native.pipeline
        original = pipe.encode_prompt
        calls = []
        def cached_condition(*args, **options):
            if args or options.get("mode") != "edit" or len(options.get("prompts", [])) != 3:
                raise RuntimeError("Unexpected native conditioning call")
            calls.append(True)
            return (self.condition.prompt_embeds.repeat(3, 1, 1),
                    self.condition.attention_mask.repeat(3, 1))
        pipe.encode_prompt = cached_condition
        try:
            result = self.native(**kwargs)
        finally:
            pipe.encode_prompt = original
        if len(calls) != 1:
            raise RuntimeError("Native sampler did not consume the cached condition exactly once")
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--condition-style", choices=("native", "training_raw"), default="native")
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
            "--run", str(a.run), "--output", str(a.output),
            "--condition-style", a.condition_style, "--worker"], env=env)
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
        if a.condition_style == "training_raw":
            context["inference_sampler"] = TrainingConditionNativeSampler(context["inference_sampler"], context["condition"])
    # Load the bound trainable tensors first, then freeze every model for this
    # zero-update probe. This applies to both LoRA and full-U-Net checkpoints.
    for module in (runtime["pipe"].unet, runtime["pipe"].vae, runtime["pipe"].text_encoder, runtime["reader"]):
        module.eval().requires_grad_(False)
    args.output_dir = a.output
    identity = {"probe_commit": commit, "parent_result_sha256": terminal["training_result_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"], "guidance_scale": 1.,
        "condition_style": a.condition_style,
        "cached_condition_sha256": parent_runtime["condition_sha256"] if a.condition_style == "training_raw" else None,
        "trainable_scope": getattr(args, "trainable_scope", "lora"), "optimizer_updates": 0,
        "image_guidance_scale": 1., "inference_steps": 28, "seed": args.seed,
        "scope": ("Same checkpoint/noise/source/Reader and native28 loop; CFG1 and cached raw training condition"
                  if a.condition_style == "training_raw" else
                  "Same checkpoint, noise, native prompt, image and Reader; only CFG changes 7.5 to 1")}
    train.write_json(a.output / "identity.json", identity)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    phase = "guidance1" if a.condition_style == "native" else "training_raw_guidance1"
    summary = train.evaluate(args, runtime, bank, teachers, phase)
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    train.write_json(a.output / "complete.json", {"identity": identity, "summary": summary,
        "phase_complete_sha256": file_sha256(a.output / phase / "complete.json")})
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
