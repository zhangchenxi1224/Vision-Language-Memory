"""Fresh-noise confirmation and fixed same-entity event interventions."""
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
PARENT_RESULT_SHA = "51332a99686e4de865bdfc344e68f371b132e0d9a741c1f0410c5852563f485b"
EVENT = "R3 Train Standard Templates 03: For the indigo desk train 001123, remember that the preferred music is ambient."


def confirmation_plan(master_seed):
    from vision_memory.training.latent_bank_unet import stable_seed
    seeds = [stable_seed(master_seed, "official-full-confirmation-noise-v1", i) for i in range(16)]
    forbidden = {stable_seed(master_seed, "training-noise", i) for i in range(14000)}
    forbidden.update(stable_seed(master_seed, "heldout-evaluation-noise", i) for i in range(8))
    forbidden.update(stable_seed(master_seed, "official-prompt-control-noise", i) for i in range(4))
    if len(set(seeds)) != 16 or forbidden.intersection(seeds):
        raise RuntimeError("Confirmation noises overlap previous training/evaluation")
    return [
        {"case": "trained_event", "event": EVENT, "gold": "ambient", "seeds": seeds},
        {"case": "replace_jazz", "event": EVENT.replace("ambient.", "jazz."), "gold": "jazz", "seeds": seeds[:4]},
        {"case": "clear", "event": "R3 Train Standard Templates 03: Clear the saved music preference for the indigo desk train 001123.",
         "gold": "no active preference", "seeds": seeds[:4]},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256
    command = json.loads((a.parent_run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    bank, _ = load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--parent-run", str(a.parent_run),
            "--output", str(a.output), "--deadline-unix", str(a.deadline_unix), "--worker"], env=env)
    terminal = json.loads((a.parent_run / "terminal.json").read_text())
    if (terminal.get("state") != "completed" or terminal["training_result_sha256"] != PARENT_RESULT_SHA
            or file_sha256(a.parent_run / "train/result.json") != PARENT_RESULT_SHA):
        raise RuntimeError("Confirmation must use the fixed verified successful full512 endpoint")
    result = json.loads((a.parent_run / "train/result.json").read_text())
    checkpoint = a.parent_run / "train/checkpoint-final.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint hash changed")
    if len(bank["groups"]) != 1 or bank["groups"][0]["event_text"] != EVENT:
        raise ValueError("The fixed intervention requires the original ambient event")
    if args.seed != 20260913 or args.trainable_scope != "full_unet" or args.steps != 512:
        raise ValueError("Unexpected parent training design")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Confirmation source must be immutable")
    import torch
    from PIL import Image
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.mem_get_info(torch.device(args.dreamlite_device))[0] < 70 * 1024**3:
        raise RuntimeError("Confirmation requires 70GiB free; do not evict existing work")
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = train.load_runtime(args, bank)
    parent_runtime = json.loads((a.parent_run / "train/runtime.json").read_text())
    parent_identity = json.loads((a.parent_run / "train/identity.json").read_text())
    if (runtime["snapshots"] != parent_runtime["snapshots"]
            or runtime["protocol_binding"] != parent_runtime["additional_protocol_binding"]
            or {k: canonical_tensor_sha256(v["condition"].prompt_embeds) for k, v in runtime["contexts"].items()}
               != parent_runtime["condition_sha256"]):
        raise RuntimeError("Confirmation model or training condition differs from parent")
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime["pipe"].unet)
    if loaded["optimizer_step"] != 512 or loaded["manifest"] != {**parent_identity, **parent_runtime}:
        raise RuntimeError("Parent checkpoint manifest or cursor differs")
    del loaded
    for module in (runtime["pipe"].unet, runtime["pipe"].vae, runtime["pipe"].text_encoder, runtime["reader"]):
        module.eval().requires_grad_(False)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    plan = confirmation_plan(args.seed)
    identity = {"probe_commit": commit, "parent_result_sha256": PARENT_RESULT_SHA,
        "checkpoint_sha256": result["checkpoint_sha256"], "bank_sha256": file_sha256(args.bank_manifest),
        "plan": plan, "inference_steps": 28, "guidance_scale": 7.5, "image_guidance_scale": 1.,
        "optimizer_updates": 0, "snapshots": runtime["snapshots"], "protocol_binding": runtime["protocol_binding"],
        "deadline_unix": a.deadline_unix, "placement": [args.dreamlite_device, args.reader_device],
        "scope": "16 fresh noises for the trained event; 4 paired noises for unseen same-entity jazz/clear interventions; no training"}
    train.write_json(a.output / "identity.json", identity)
    group = bank["groups"][0]
    context = runtime["contexts"][group["question_id"]]
    if set(group["question_variants"]) != {"original_open", "paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4"}:
        raise ValueError("Five fixed question variants required")
    cells = {}
    def read_image(case, event, gold, seed, pixels, artifact):
        for prompt_id, query in group["question_variants"].items():
            if time.time() >= a.deadline_unix:
                raise TimeoutError("Confirmation lease expired; partial evidence retained")
            ce = train.qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"], query=query, target=gold,
                termination=runtime["termination"], lambda_eos=1., require_image_grad=False, deterministic_ce=True,
                reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
            generation = train.generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                image=pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
            score = train.generation_diagnostics(generation, gold, ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
            row = {"case": case, "event_text": event, "gold": gold, "noise_seed": seed,
                "prompt_id": prompt_id, "query": query, "image_artifact": artifact,
                "image_sha256": canonical_tensor_sha256(pixels.cpu()), "answer_ce": float(ce.answer_loss),
                "eos_ce": float(ce.eos_loss), **generation, "scorer": score}
            train.append_jsonl(a.output / "generations.jsonl", row)
            cell = cells.setdefault(case + "/" + prompt_id, {"n": 0, "exact_match": 0, "answer_eos": 0})
            cell["n"] += 1
            cell["exact_match"] += int(score["strict_correct"])
            cell["answer_eos"] += int(score["strict_correct"] and score["answer_followed_immediately_by_eos"])
    with torch.no_grad():
        for case in plan:
            sampler = NativeBaseEditSampler(runtime["pipe"], source_image=Image.new("RGB", (1024, 1024), (128, 128, 128)),
                event_text=case["event"], guidance_scale=7.5)
            for i, seed in enumerate(case["seeds"]):
                if time.time() >= a.deadline_unix:
                    raise TimeoutError("Confirmation lease expired")
                noise = torch.randn(context["source"].shape, generator=torch.Generator().manual_seed(seed)).to(runtime["vae_device"])
                out = sampler(source_latents=context["source"], noise_latents=noise, num_steps=28)
                pixels = train.decode_model_latents_unit_interval(runtime["pipe"].vae, out.latents, clamp=True).cpu()
                label = f"{case['case']}-seed-{i:02d}"
                train.atomic_tensor(a.output / (label + ".pt"), {"noise_seed": seed, "latent": out.latents.cpu(),
                    "image": pixels, "trajectory": [x.cpu() for x in out.trajectory]})
                rgb = (pixels[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
                Image.fromarray(rgb).save(a.output / (label + ".png"))
                read_image(case["case"], case["event"], case["gold"], seed, pixels, label + ".pt")
                print(label, flush=True)
        for kind in ("blank", "donor"):
            pixels = context[kind].cpu()
            train.atomic_tensor(a.output / (kind + ".pt"), {"image": pixels})
            read_image(kind, None, "ambient", None, pixels, kind + ".pt")
    if sum(c["n"] for c in cells.values()) != 130:
        raise RuntimeError("Missing confirmation evaluation coverage")
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    if file_sha256(a.parent_run / "train/result.json") != PARENT_RESULT_SHA or file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent evidence changed during confirmation")
    train.write_json(a.output / "complete.json", {"identity": identity, "cells": cells,
        "artifact_hashes": {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    print(json.dumps(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
