"""Build explicit source-conditioned transitions from three verified oracle states."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
BANK_SHA = "5165c059a0a4c0760cd4b0df1663c642132e52bf55a13031cd99207fd200bb86"
STATES = ("ambient", "jazz", "clear")
NOOP_EVENT = "R3 Train Standard Templates 03: A delivery arrived for the indigo desk train 001123. Keep its saved music preference unchanged."


def transition_plan():
    plan = [{"source_state": "gray", "target_state": target, "operation": "clear" if target == "clear" else "set"}
            for target in STATES]
    for source in STATES:
        plan.extend({"source_state": source, "target_state": target,
                     "operation": "clear" if target == "clear" else "set"} for target in STATES)
        plan.append({"source_state": source, "target_state": source, "operation": "noop"})
    return plan


def assemble_bank(parent, sources):
    groups = {g["state"]: g for g in parent["groups"]}
    teachers = {t["teacher_id"]: t for t in parent["teachers"]}
    if len(parent["groups"]) != 3 or set(groups) != set(STATES) or set(sources) != set(STATES):
        raise ValueError("Exactly the three fixed states and source images are required")
    new_groups, new_teachers = [], []
    for cell in transition_plan():
        original = groups[cell["target_state"]]
        if len(original["teacher_ids"]) != 1:
            raise ValueError("Each state must have one fixed target")
        group = copy.deepcopy(original)
        qid = original["semantic_question_id"] + "-from-" + cell["source_state"] + "-" + cell["operation"] + "-to-" + cell["target_state"]
        if cell["source_state"] != "gray":
            group.update(sources[cell["source_state"]])
            group["source_answer"] = groups[cell["source_state"]]["answer"]
        if cell["operation"] == "noop":
            group["event_text"] = NOOP_EVENT
        teacher = copy.deepcopy(teachers[original["teacher_ids"][0]])
        tid = qid + "-" + teacher["latent_sha256"][:16]
        teacher.update(teacher_id=tid, question_id=qid, reused_parent_teacher=original["teacher_ids"][0])
        group.update(question_id=qid, teacher_ids=[tid], planned_count=1, successful_run_count=1, **cell)
        new_groups.append(group)
        new_teachers.append(teacher)
    return {"schema": "latent-teacher-bank/v1", "bank_status": "sealed", "route": "direct",
            "models": copy.deepcopy(parent["models"]), "snapshots": copy.deepcopy(parent["snapshots"]),
            "groups": new_groups, "teachers": new_teachers, "semantic_question_count": 1,
            "conditional_group_count": 15, "unique_target_state_count": 3,
            "provenance": {"parent_bank_sha256": BANK_SHA, "optimizer_updates": 0,
                "target_latents": "unchanged parent endpoints; copied records identify conditional groups, not new independent targets"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256, stable_seed
    command = json.loads((a.parent_run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    parent_manifest = args.bank_manifest
    if file_sha256(args.bank_manifest) != BANK_SHA:
        raise ValueError("The fixed three-state parent bank changed")
    parent, tensors = load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(parent), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--parent-run", str(a.parent_run),
            "--output", str(a.output), "--expected-commit", a.expected_commit,
            "--deadline-unix", str(a.deadline_unix), "--worker"], env=env)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != a.expected_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Source preparation requires the exact clean commit")
    import numpy as np
    import torch
    from PIL import Image
    from scripts.train.official_base_runtime import load_base_runtime, audit_inference_only_runtime
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info(0)[0] < 95 * 1024**3:
        raise RuntimeError("A separate available full-memory GPU is required")
    def deadline():
        if time.time() >= a.deadline_unix:
            raise TimeoutError("Source preparation deadline reached")
    args.colocate_models = True
    args.dreamlite_device = args.reader_device = "cuda:0"
    a.output.mkdir(parents=True, exist_ok=False)
    identity = {"commit": commit, "source_hashes": train.source_hashes(), "parent_bank_sha256": BANK_SHA,
                "plan": transition_plan(), "optimizer_updates": 0, "writer_checkpoint_used": False,
                "source_image_construction": "VAE decode of original oracle, clamp, nearest RGB uint8, PNG; no text rendering",
                "deadline_unix": a.deadline_unix}
    train.write_json(a.output / "identity.json", identity)
    runtime = load_base_runtime(args, parent, inference_only=True)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    sources, readbacks = {}, []
    with torch.no_grad():
        for group in parent["groups"]:
            deadline()
            state = group["state"]
            latent = tensors[group["teacher_ids"][0]].to(runtime["vae_device"])
            decoded = train.decode_model_latents_unit_interval(runtime["pipe"].vae, latent, clamp=True).cpu()
            image = Image.fromarray((decoded[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            image_path = a.output / (state + "-source.png")
            image.save(image_path)
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
            source = runtime["pipe"].prepare_image_latents(runtime["pipe"].image_processor.preprocess(image),
                dtype=torch.float32, device=runtime["vae_device"])
            source_path = a.output / (state + "-source.pt")
            train.atomic_tensor(source_path, source.cpu())
            sources[state] = {"source_kind": "sealed_rgb_1024", "source_image_path": str(image_path),
                "source_image_file_sha256": file_sha256(image_path), "source_latent_path": str(source_path),
                "source_latent_file_sha256": file_sha256(source_path), "source_latent_sha256": canonical_tensor_sha256(source.cpu())}
            for prompt_id, query in group["question_variants"].items():
                deadline()
                ce = train.qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                    image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"], query=query,
                    target=group["answer"], termination=runtime["termination"], lambda_eos=1., require_image_grad=False,
                    deterministic_ce=True, reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
                generation = train.generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                    image=pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
                score = train.generation_diagnostics(generation, group["answer"], ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
                row = {"state": state, "question_id": group["question_id"], "prompt_id": prompt_id, "query": query,
                       "gold": group["answer"], "image_sha256": canonical_tensor_sha256(pixels),
                       "image_file_sha256": file_sha256(image_path), **generation, "scorer": score}
                train.append_jsonl(a.output / "source-generations.jsonl", row)
                readbacks.append(row)
            print("source_readback " + state, flush=True)
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    if len(readbacks) != 15 or not all(r["scorer"]["strict_correct"] and r["scorer"]["answer_followed_immediately_by_eos"] for r in readbacks):
        train.write_json(a.output / "complete.json", {"bank_sealed": False, "reason": "PNG source readback failed",
            "generations_sha256": file_sha256(a.output / "source-generations.jsonl"), "identity": identity})
        return 2
    # Release the first frozen runtime before loading all fifteen actual contexts.
    del runtime
    torch.cuda.empty_cache()
    new_bank = assemble_bank(parent, sources)
    candidate = a.output / "candidate-manifest.json"
    train.write_json(candidate, {**new_bank, "release_status": "candidate_runtime_verification_pending"})
    load_teacher_bank(candidate)
    args.bank_manifest = candidate
    runtime = load_base_runtime(args, new_bank, inference_only=True)
    frozen = train.frozen_versions(runtime["pipe"], runtime["reader"])
    source_checks = {}
    with torch.no_grad():
        for i, group in enumerate(g for g in new_bank["groups"] if g["operation"] == "noop"):
            deadline()
            context = runtime["contexts"][group["question_id"]]
            noise_seed = stable_seed(args.seed, "official-source-image-parity-noise-v1", i)
            noise = torch.randn(context["source"].shape, generator=torch.Generator().manual_seed(noise_seed)).to(runtime["vae_device"])
            # The native sampler asserts that upstream inference encodes exactly
            # the source used by training, and captures all28 denoising steps.
            output = context["inference_sampler"](source_latents=context["source"], noise_latents=noise, num_steps=28)
            name = group["source_state"] + "-native-source-check.pt"
            train.atomic_tensor(a.output / name, {"noise_seed": noise_seed, "source": context["source"].cpu(),
                "latent": output.latents.cpu(), "trajectory": [x.cpu() for x in output.trajectory]})
            source_checks[group["source_state"]] = {"native_source_equal": True, "artifact": name,
                "condition_sha256": canonical_tensor_sha256(context["condition"].prompt_embeds.cpu()),
                "source_sha256": canonical_tensor_sha256(context["source"].cpu()), "event_text": group["event_text"]}
            print("native_source_check " + group["source_state"], flush=True)
    if len(source_checks) != 3 or len({v["condition_sha256"] for v in source_checks.values()}) != 3:
        raise RuntimeError("Distinct source images did not produce distinct image-aware conditions")
    audit_inference_only_runtime(runtime, frozen)
    runtime["verify_additional_bindings"]()
    if train.source_hashes() != identity["source_hashes"] or file_sha256(parent_manifest) != BANK_SHA:
        raise RuntimeError("Preparation source or parent bank changed")
    new_bank["provenance"].update(preparation_identity_sha256=file_sha256(a.output / "identity.json"),
        source_generations_sha256=file_sha256(a.output / "source-generations.jsonl"))
    final_manifest = a.output / "manifest.json"
    train.write_json(final_manifest, new_bank)
    load_teacher_bank(final_manifest)
    train.write_json(a.output / "complete.json", {"bank_sealed": True, "identity": identity,
        "bank_manifest_sha256": file_sha256(final_manifest), "source_readback_answer_eos": "15/15",
        "native_source_checks": source_checks, "runtime_protocol_binding": runtime["protocol_binding"],
        "scope": "source encoding/conditioning parity and readable incoming states; untrained Writer outputs are not functional success",
        "artifact_hashes": {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
