"""Fixed fresh-noise and withheld-event-wording confirmation of one shared Writer."""
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
BANK_SHA = "5165c059a0a4c0760cd4b0df1663c642132e52bf55a13031cd99207fd200bb86"
PARENT_COMMIT = "c90896c55c7b6cd6f489ccb7848ade2cff587daa"
PROMPTS = {"original_open", "paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4"}
GOLDS = {"ambient": "ambient", "jazz": "jazz", "clear": "no active preference"}


def confirmation_plan(bank, seed=20260913):
    from vision_memory.training.latent_bank_unet import stable_seed
    groups = {g["state"]: g for g in bank["groups"]}
    if len(bank["groups"]) != 3 or set(groups) != set(GOLDS):
        raise ValueError("Exactly three fixed conditional states required")
    seeds = [stable_seed(seed, "official-three-state-confirmation-noise-v1", i) for i in range(16)]
    forbidden = {stable_seed(seed, "training-noise", i) for i in range(14000)}
    for namespace, count in (("heldout-evaluation-noise", 8), ("official-prompt-control-noise", 4),
                             ("official-full-confirmation-noise-v1", 16)):
        forbidden.update(stable_seed(seed, namespace, i) for i in range(count))
    if len(set(seeds)) != 16 or forbidden.intersection(seeds):
        raise ValueError("Confirmation noises must be new and unique")
    plan = []
    for state in ("ambient", "jazz", "clear"):
        group = groups[state]
        if group["answer"] != GOLDS[state] or set(group["question_variants"]) != PROMPTS:
            raise ValueError("Unexpected state answer or question variants")
        if state == "clear":
            rewordings = [
                "R3 Train Standard Templates 03: Remove the music preference stored for the indigo desk train 001123; it now has no active music preference.",
                "For the indigo desk train 001123, forget its saved music preference. There is no current music preference.",
            ]
        else:
            rewordings = [
                f"R3 Train Standard Templates 03: Set the current music preference of the indigo desk train 001123 to {state}.",
                f"The indigo desk train 001123 now prefers {state} music. Store this as its current music preference.",
            ]
        for style, event, selected in [("trained", group["event_text"], seeds),
                                      ("reworded_1", rewordings[0], seeds[:4]),
                                      ("reworded_2", rewordings[1], seeds[:4])]:
            plan.append({"case": state + "_" + style, "state": state, "style": style,
                         "question_id": group["question_id"], "event": event,
                         "gold": group["answer"], "seeds": selected})
    return plan


def validate_development_rows(rows, bank, seed=20260913):
    """Reject missing/duplicate coverage and any failed state, even if aggregate looks good."""
    from vision_memory.training.latent_bank_unet import stable_seed
    groups = {g["question_id"]: g for g in bank["groups"]}
    expected = {(qid, stable_seed(seed, "heldout-evaluation-noise", i), prompt)
                for qid in groups for i in range(8) for prompt in PROMPTS}
    seen = set()
    for row in rows:
        if row["condition"] != "matched":
            continue
        key = (row["question_id"], row["noise_seed"], row["prompt_id"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate development cell")
        seen.add(key)
        group = groups[row["question_id"]]
        tokens = row["generated_token_ids"]
        answer_tokens = row["scorer"]["gold_token_ids"]
        if (row["gold"] != group["answer"] or row["query"] != group["question_variants"][row["prompt_id"]]
                or not row["scorer"]["strict_correct"] or not row["scorer"]["answer_followed_immediately_by_eos"]
                or tokens[:-1] != answer_tokens or not tokens
                or tokens[-1] != group["termination_contract"]["assistant_end_token_id"]):
            raise ValueError("Development endpoint did not pass every state with immediate EOS")
    if seen != expected:
        raise ValueError("Incomplete development coverage")


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
    if file_sha256(args.bank_manifest) != BANK_SHA:
        raise ValueError("Parent must use the fixed complete three-state bank")
    bank, _ = load_teacher_bank(args.bank_manifest)
    plan = confirmation_plan(bank, args.seed)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--parent-run", str(a.parent_run),
            "--output", str(a.output), "--deadline-unix", str(a.deadline_unix), "--worker"], env=env)
    terminal = json.loads((a.parent_run / "terminal.json").read_text())
    result_sha = file_sha256(a.parent_run / "train/result.json")
    if terminal.get("state") != "completed" or terminal["training_result_sha256"] != result_sha:
        raise RuntimeError("Parent must have a sealed completed endpoint")
    if args.expected_commit != PARENT_COMMIT or args.seed != 20260913 or args.trainable_scope != "full_unet" or args.steps != 1536:
        raise ValueError("Unexpected parent design")
    phase = a.parent_run / "train/trained"
    phase_complete = json.loads((phase / "complete.json").read_text())
    for name, digest in phase_complete["artifact_hashes"].items():
        if file_sha256(phase / name) != digest:
            raise ValueError("Parent development artifact changed")
    validate_development_rows([json.loads(line) for line in (phase / "generations.jsonl").read_text().splitlines()], bank)
    result = json.loads((a.parent_run / "train/result.json").read_text())
    checkpoint = a.parent_run / "train/checkpoint-final.pt"
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint changed")
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
        raise RuntimeError("Runtime differs from parent bindings")
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime["pipe"].unet)
    if loaded["optimizer_step"] != 1536 or loaded["manifest"] != {**parent_identity, **parent_runtime}:
        raise RuntimeError("Parent checkpoint manifest or cursor differs")
    del loaded
    for module in (runtime["pipe"].unet, runtime["pipe"].vae, runtime["pipe"].text_encoder, runtime["reader"]):
        module.eval().requires_grad_(False)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    identity = {"probe_commit": commit, "parent_result_sha256": result_sha, "parent_commit": PARENT_COMMIT,
        "checkpoint_sha256": result["checkpoint_sha256"], "bank_sha256": BANK_SHA, "plan": plan,
        "inference_steps": 28, "guidance_scale": 7.5, "image_guidance_scale": 1., "optimizer_updates": 0,
        "snapshots": runtime["snapshots"], "protocol_binding": runtime["protocol_binding"],
        "deadline_unix": a.deadline_unix, "semantic_question_count": 1, "conditional_state_count": 3,
        "scope": "Fixed three states; 16 paired fresh noises per trained event; two unseen wordings with four paired noises each; no sequential memory update test"}
    train.write_json(a.output / "identity.json", identity)
    groups = {g["question_id"]: g for g in bank["groups"]}
    cells = {}
    def read_image(case, group, event, seed, pixels, artifact):
        for prompt_id, query in group["question_variants"].items():
            if time.time() >= a.deadline_unix:
                raise TimeoutError("Confirmation lease expired; partial evidence retained")
            gold = group["answer"]
            ce = train.qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"], query=query, target=gold,
                termination=runtime["termination"], lambda_eos=1., require_image_grad=False, deterministic_ce=True,
                reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
            generation = train.generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                image=pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
            score = train.generation_diagnostics(generation, gold, ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
            train.append_jsonl(a.output / "generations.jsonl", {"case": case, "state": group["state"],
                "question_id": group["question_id"], "event_text": event, "gold": gold, "noise_seed": seed,
                "prompt_id": prompt_id, "query": query, "image_artifact": artifact,
                "image_sha256": canonical_tensor_sha256(pixels.cpu()), "answer_ce": float(ce.answer_loss),
                "eos_ce": float(ce.eos_loss), **generation, "scorer": score})
            cell = cells.setdefault(case + "/" + prompt_id, {"n": 0, "exact_match": 0, "answer_eos": 0})
            cell["n"] += 1
            cell["exact_match"] += int(score["strict_correct"])
            cell["answer_eos"] += int(score["strict_correct"] and score["answer_followed_immediately_by_eos"])
    with torch.no_grad():
        for case in plan:
            group = groups[case["question_id"]]
            context = runtime["contexts"][group["question_id"]]
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
                Image.fromarray((pixels[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")).save(a.output / (label + ".png"))
                read_image(case["case"], group, case["event"], seed, pixels, label + ".pt")
                print(label, flush=True)
        for group in groups.values():
            context = runtime["contexts"][group["question_id"]]
            for kind in ("blank", "donor"):
                label = group["state"] + "_" + kind
                pixels = context[kind].cpu()
                train.atomic_tensor(a.output / (label + ".pt"), {"image": pixels})
                read_image(label, group, None, None, pixels, label + ".pt")
    expected = {case["case"] + "/" + prompt: len(case["seeds"]) for case in plan for prompt in PROMPTS}
    expected.update({state + "_" + kind + "/" + prompt: 1 for state in GOLDS for kind in ("blank", "donor") for prompt in PROMPTS})
    if {k: v["n"] for k, v in cells.items()} != expected:
        raise RuntimeError("Missing confirmation coverage")
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    if file_sha256(a.parent_run / "train/result.json") != result_sha or file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Parent changed during confirmation")
    train.write_json(a.output / "complete.json", {"identity": identity, "cells": cells,
        "all_generated_answer_eos": all(cells[k]["answer_eos"] == n for k, n in expected.items()
            if not any("_" + control + "/" in k for control in ("blank", "donor"))),
        "artifact_hashes": {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    print(json.dumps(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
