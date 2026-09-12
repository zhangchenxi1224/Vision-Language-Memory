"""Fixed image-only chains with writes, overwrites, clear, and source-dependent no-ops."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiments.official_transition_bank import NOOP_EVENT
from scripts.experiments.official_state_oracles import STATES


def chain_plan(seed=20260913):
    from vision_memory.training.latent_bank_unet import stable_seed
    events = {state: (event, gold) for state, event, gold in STATES}
    orders = [("ambient", "jazz", "clear"), ("jazz", "ambient", "clear"),
              ("clear", "ambient", "jazz"), ("clear", "jazz", "ambient")]
    plan = []
    for sequence, order in enumerate(orders):
        for repetition in range(4):
            steps = []
            for state in order:
                event, gold = events[state]
                for operation, prompt in (("write", event), ("noop", NOOP_EVENT)):
                    index = len(steps)
                    steps.append({"step": index, "operation": operation, "event_text": prompt,
                        "expected_state": state, "gold": gold,
                        "noise_seed": stable_seed(seed, "official-rgb-memory-chain-noise-v1", repetition * 6 + index)})
            plan.append({"sequence": sequence, "repetition": repetition, "steps": steps})
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--parent-result-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import file_sha256, load_teacher_bank, stable_seed
    command = json.loads((a.parent_run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    bank, _ = load_teacher_bank(args.bank_manifest)
    bank_sha = file_sha256(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--parent-run", str(a.parent_run),
            "--parent-result-sha256", a.parent_result_sha256, "--output", str(a.output),
            "--deadline-unix", str(a.deadline_unix), "--worker"], env=env)
    terminal = json.loads((a.parent_run / "terminal.json").read_text())
    if (terminal.get("state") != "completed" or terminal["training_result_sha256"] != a.parent_result_sha256
            or file_sha256(a.parent_run / "train/result.json") != a.parent_result_sha256):
        raise ValueError("An exact completed parent endpoint is required")
    if args.model_variant != "base" or args.flow_protocol != "official" or args.trainable_scope != "full_unet":
        raise ValueError("This chain protocol uses the official Base full-U-Net Writer")
    groups = {}
    for state, event, gold in STATES:
        matching = [g for g in bank["groups"] if g.get("state") == state and g["event_text"] == event and g["answer"] == gold]
        if not matching:
            raise ValueError("Missing fixed state event from the parent")
        groups[state] = matching[0]
    variants = groups["ambient"]["question_variants"]
    if any(g["question_variants"] != variants for g in bank["groups"]) or len(variants) != 5:
        raise ValueError("This registered chain covers one semantic question and five fixed query variants")
    plan = chain_plan(args.seed)
    used = {step["noise_seed"] for sequence in plan for step in sequence["steps"]}
    forbidden = {stable_seed(args.seed, "training-noise", i) for i in range(max(14000, args.steps * args.gradient_accumulation_steps))}
    for namespace, n in (("heldout-evaluation-noise", 8), ("official-prompt-control-noise", 4),
                         ("official-full-confirmation-noise-v1", 16), ("official-three-state-confirmation-noise-v1", 16),
                         ("official-source-image-parity-noise-v1", 3)):
        forbidden.update(stable_seed(args.seed, namespace, i) for i in range(n))
    if len(used) != 24 or used.intersection(forbidden):
        raise ValueError("Chain noise plan must be fresh, with intentional pairing only across sequence orders")
    result = json.loads((a.parent_run / "train/result.json").read_text())
    checkpoint = a.parent_run / "train/checkpoint-final.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Chain probe source must be immutable")
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.mem_get_info(torch.device(args.dreamlite_device))[0] < 70 * 1024**3:
        raise RuntimeError("Chain probe requires70GiB free; do not evict existing work")
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = train.load_runtime(args, bank)
    parent_runtime = json.loads((a.parent_run / "train/runtime.json").read_text())
    parent_identity = json.loads((a.parent_run / "train/identity.json").read_text())
    if (runtime["snapshots"] != parent_runtime["snapshots"] or runtime["protocol_binding"] != parent_runtime["additional_protocol_binding"]
            or {k: canonical_tensor_sha256(v["condition"].prompt_embeds) for k, v in runtime["contexts"].items()} != parent_runtime["condition_sha256"]):
        raise RuntimeError("Chain runtime differs from its parent")
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime["pipe"].unet)
    if loaded["optimizer_step"] != args.steps or loaded["manifest"] != {**parent_identity, **parent_runtime}:
        raise RuntimeError("Parent checkpoint cursor or manifest differs")
    del loaded
    for module in (runtime["pipe"].unet, runtime["pipe"].vae, runtime["pipe"].text_encoder, runtime["reader"]):
        module.eval().requires_grad_(False)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    identity = {"probe_commit": commit, "source_hashes": train.source_hashes(), "parent_result_sha256": a.parent_result_sha256,
        "checkpoint_sha256": result["checkpoint_sha256"], "bank_sha256": bank_sha, "plan": plan,
        "optimizer_updates": 0, "guidance_scale": 7.5, "image_guidance_scale": 1., "native_steps": 28,
        "state": "only the previous generated RGB uint8 image; official VAE re-encoding at every event",
        "reader_input": "actual quantized PNG pixels, including when previous steps failed",
        "scope": "one entity and three states;16 six-event chains; no unseen-entity or multi-fact claim"}
    train.write_json(a.output / "identity.json", identity)
    cells = {}
    with torch.no_grad():
        for sequence in plan:
            memory = OfficialRGBMemory(runtime["pipe"], guidance_scale=7.5)
            previous_artifact = None
            for step in sequence["steps"]:
                if time.time() >= a.deadline_unix:
                    raise TimeoutError("Chain lease expired; partial chains retained")
                output = memory.write(step["event_text"], seed=step["noise_seed"])
                label = f"sequence-{sequence['sequence']}-rep-{sequence['repetition']}-step-{step['step']}"
                output.image.save(a.output / (label + ".png"))
                image_sha = file_sha256(a.output / (label + ".png"))
                train.atomic_tensor(a.output / (label + ".pt"), {"source": output.source_latent, "noise": output.noise,
                    "latent": output.latent, "image": output.pixels, "trajectory": output.trajectory})
                memory_before_queries = memory.image.tobytes()
                for prompt_id, query in variants.items():
                    if time.time() >= a.deadline_unix:
                        raise TimeoutError("Chain lease expired")
                    ce = train.qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                        image=output.pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"], query=query,
                        target=step["gold"], termination=runtime["termination"], lambda_eos=1., require_image_grad=False,
                        deterministic_ce=True, reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
                    generation = train.generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                        image=output.pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
                    score = train.generation_diagnostics(generation, step["gold"], ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
                    row = {"sequence": sequence["sequence"], "repetition": sequence["repetition"], **step,
                        "source_artifact": previous_artifact, "image_artifact": label + ".png", "image_file_sha256": image_sha,
                        "image_sha256": canonical_tensor_sha256(output.pixels), "prompt_id": prompt_id, "query": query,
                        "answer_ce": float(ce.answer_loss), "eos_ce": float(ce.eos_loss), **generation, "scorer": score}
                    train.append_jsonl(a.output / "generations.jsonl", row)
                    key = f"sequence-{sequence['sequence']}/step-{step['step']}/{prompt_id}"
                    cell = cells.setdefault(key, {"n": 0, "correct_eos": 0})
                    cell["n"] += 1
                    cell["correct_eos"] += int(score["strict_correct"] and score["answer_followed_immediately_by_eos"])
                if memory.image.tobytes() != memory_before_queries:
                    raise RuntimeError("Reader queries mutated memory")
                previous_artifact = label + ".png"
                print(label, flush=True)
    if len(cells) != 120 or any(c["n"] != 4 for c in cells.values()):
        raise RuntimeError("Incomplete chain coverage")
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    if (file_sha256(args.bank_manifest) != bank_sha or file_sha256(checkpoint) != result["checkpoint_sha256"]
            or train.source_hashes() != identity["source_hashes"]):
        raise RuntimeError("Chain source, model, or bank changed")
    train.write_json(a.output / "complete.json", {"identity": identity, "cells": cells,
        "all_correct_eos": all(c["correct_eos"] == c["n"] for c in cells.values()),
        "artifact_hashes": {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
