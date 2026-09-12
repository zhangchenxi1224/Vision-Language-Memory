"""Train DreamLite U-Net LoRA from a sealed successful-latent set, then evaluate.

The two oracle routes use this identical entry point and budget independently.
The default is upstream target/noise flow matching, without Reader QA loss.
Only new-noise four-step Writer outputs determine final QA performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler
from vision_memory.dreamlite.conditioning import encode_image_edit_condition, encode_latent_path_condition
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import assistant_termination_contract, generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism, lora_trainable_parameters
from vision_memory.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from vision_memory.training.latent_bank_unet import (EFFECTIVE_SIGMAS, START_SIGMA, anchored_flow_bridge,
    OFFICIAL_REFERENCE_COMMIT, OFFICIAL_RAW_SIGMAS, official_flow_bridge,
    balanced_draw, bank_geometry, file_sha256, load_teacher_bank, member_split, predict_velocity, stable_seed)


def is_official_flow(args) -> bool:
    # Older diagnostic entry points construct their own Namespace. Preserve
    # their explicitly historical behavior; the public CLI defaults to official.
    return getattr(args, "flow_protocol", "legacy_anchored") == "official"


def training_groups(bank, target_mode):
    if target_mode == "bank":
        return bank["groups"]
    if target_mode != "single":
        raise ValueError("Unknown target mode")
    # Select a training member by hash only, never by test-prompt performance.
    return [{**g, "teacher_ids": [member_split(g["teacher_ids"])[0][0]]} for g in bank["groups"]]


class TrainingPaused(Exception):
    """An interruption is resumable, not a successful completed training run."""


def pause_if_requested(runtime) -> None:
    if runtime.get("should_pause", lambda: False)():
        raise TrainingPaused("Signal or configured lease deadline; deterministic evaluation/optimizer recovery is available")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_tensor(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): file_sha256(p)
            for tree in ("scripts", "src") for p in sorted((ROOT / tree).rglob("*.py"))}


def resolve_payload(record: dict, prefix: str, bank_path: Path) -> torch.Tensor:
    path = Path(record[prefix + "_path"])
    if not path.is_absolute():
        path = bank_path.parent / path
    if file_sha256(path) != record[prefix + "_file_sha256"]:
        raise ValueError(f"Bound {prefix} file changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    key = record.get(prefix + "_tensor_key")
    if key:
        value = value[key]
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"Invalid {prefix} tensor")
    tensor_hash = record.get(prefix + "_sha256")
    if tensor_hash is not None and canonical_tensor_sha256(value) != tensor_hash:
        raise ValueError(f"Bound {prefix} tensor changed")
    return value


def frozen_versions(pipe, reader) -> dict[str, int]:
    modules = {"vae": pipe.vae, "condition_encoder": pipe.text_encoder, "reader": reader, "unet": pipe.unet}
    return {f"{label}.{name}": int(p._version) for label, module in modules.items()
            for name, p in module.named_parameters() if not p.requires_grad}


def frozen_audit(pipe, reader, versions: dict[str, int]) -> None:
    if frozen_versions(pipe, reader) != versions:
        raise RuntimeError("A frozen model parameter was modified in memory")
    for module in (pipe.vae, pipe.text_encoder, reader):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError("VAE, Reader and condition encoder must stay frozen")
    lora_trainable_parameters(pipe.unet)
    if any(p.grad is not None for p in pipe.unet.parameters() if not p.requires_grad):
        raise RuntimeError("A frozen U-Net base weight received a gradient")


@torch.no_grad()
def load_runtime(args, bank) -> dict:
    if getattr(args,"model_variant","mobile") == "base":
        from scripts.train.official_base_runtime import load_base_runtime
        return load_base_runtime(args,bank)
    from scripts.train import r11_new_frozen_dreamlite_oracle as legacy
    from scripts.experiments.run_r11_open_answer_replay import snapshot_bindings
    from peft import LoraConfig, get_peft_model
    args.reader = args.reader_model
    snapshots = snapshot_bindings(args)
    def snapshot_identities(values):
        return {(v["repo_id"], v["revision"], v["snapshot_payload_sha256"]) for v in values.values()}
    if snapshot_identities(bank["snapshots"]) != snapshot_identities(snapshots):
        raise RuntimeError("Writer models differ from the teacher bank's locked model snapshots")
    vd, rd = torch.device(args.dreamlite_device), torch.device(args.reader_device)
    if vd.type != "cuda" or rd.type != "cuda" or vd == rd or not torch.cuda.is_available():
        raise ValueError("Writer and Reader require distinct CUDA devices")
    pipe = legacy._load_pipeline(args, vd, torch.float32)
    processor, reader = legacy._load_reader(args, rd, torch.bfloat16)
    torch.manual_seed(args.seed)
    pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
        lora_dropout=0.0, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    pipe.unet.eval()
    lora_trainable_parameters(pipe.unet)
    sampler = DifferentiableDreamLiteMobileSampler.from_pipeline(pipe, checkpoint_unet=False)
    contexts = {}
    for group in bank["groups"]:
        qid = group["question_id"]
        if group.get("source_kind") != "blank_gray_1024":
            raise ValueError("Initial protocol is restricted to single-event gray-source conditions")
        source = resolve_payload(group, "source_latent", args.bank_manifest).to(vd, dtype=torch.float32)
        actual = legacy.encode_model_latent(pipe.vae, legacy.blank_source_rgb(device=vd, dtype=torch.float32))
        if source.shape != actual.shape or not torch.equal(source, actual):
            raise ValueError("Teacher source latent differs from the current FP32 gray-source encoder")
        if is_official_flow(args):
            from PIL import Image
            condition = encode_image_edit_condition(pipe, Image.new("RGB", (1024, 1024), (128, 128, 128)),
                group["event_text"], device=vd, dtype=source.dtype, prompt_style=args.prompt_style)
        else:
            condition = encode_latent_path_condition(pipe, source, group["event_text"])
        raw_sigmas = OFFICIAL_RAW_SIGMAS if is_official_flow(args) else EFFECTIVE_SIGMAS
        actual_timesteps, effective = sampler._prepare_timesteps(source, 4, raw_sigmas,
                                                               sigmas_are_effective=not is_official_flow(args))
        expected_t = torch.tensor(effective, device=vd) * int(pipe.scheduler.config.num_train_timesteps)
        if not torch.allclose(actual_timesteps.float(), expected_t.float(), atol=2e-3, rtol=2e-6):
            raise RuntimeError("Flow-matching training timestep units differ from the actual scheduler")
        blank = decode_model_latents_unit_interval(pipe.vae, source, clamp=True).cpu()
        donor = group.get("donor_control", bank.get("donor_control", {}))
        if not donor or not donor.get("answer") or donor["answer"].casefold() == group["answer"].casefold():
            raise ValueError("A fixed different-answer donor is mandatory for image-dependence evaluation")
        if "image_path" in donor:
            donor_pixels = resolve_payload(donor, "image", args.bank_manifest)
        else:
            donor_z = resolve_payload(donor, "latent", args.bank_manifest).to(vd, dtype=torch.float32)
            donor_pixels = decode_model_latents_unit_interval(pipe.vae, donor_z, clamp=True).cpu()
        if donor_pixels.ndim == 3:
            donor_pixels = donor_pixels.unsqueeze(0)
        if donor_pixels.shape[0:2] != (1, 3) or donor_pixels.min() < 0 or donor_pixels.max() > 1:
            raise ValueError("Donor is not a raw unit-RGB image")
        contexts[qid] = {"source": source, "condition": condition, "effective_sigmas": effective,
                         "blank": blank, "donor": donor_pixels, "donor_answer": donor["answer"]}
    return dict(pipe=pipe, reader=reader, processor=processor, sampler=sampler, contexts=contexts,
                vae_device=vd, reader_device=rd, snapshots=snapshots,
                termination=assistant_termination_contract(reader, processor))


def summarize_evaluation(rows: list[dict], geometry: dict) -> dict:
    cells = {}
    for row in rows:
        key = row["condition"] + "/" + row["prompt_id"]
        cell = cells.setdefault(key, {"n": 0, "exact_match": 0, "answer_eos": 0, "answer_prefix": 0, "overgeneration": 0})
        cell["n"] += 1
        for out, metric in (("exact_match", "strict_correct"), ("answer_prefix", "answer_prefix_token_exact"),
                            ("overgeneration", "overgeneration")):
            cell[out] += int(row["scorer"][metric])
        cell["answer_eos"] += int(row["scorer"]["strict_correct"] and
                                  row["scorer"].get("answer_followed_immediately_by_eos", False))
    for cell in cells.values():
        cell.update({k + "_rate": cell[k] / cell["n"] for k in ("exact_match", "answer_eos", "answer_prefix", "overgeneration")})
    per_noise = {}
    for row in rows:
        if row["condition"] == "matched":
            prompts = per_noise.setdefault((row["question_id"], row["noise_seed"]), {})
            if row["prompt_id"] in prompts:
                raise RuntimeError("Duplicate Writer prompt in a question/noise pair")
            prompts[row["prompt_id"]] = row["scorer"]
    if any(len(v) != 5 for v in per_noise.values()):
        raise RuntimeError("Each Writer question/noise pair requires all five prompts")
    return {"cells": cells,
            "matched_all_five_prompts_correct": sum(all(s["strict_correct"] for s in v.values()) for v in per_noise.values()),
            "matched_all_five_prompts_answer_eos": sum(all(s["strict_correct"] and
                s.get("answer_followed_immediately_by_eos", False) for s in v.values()) for v in per_noise.values()),
            "matched_question_noise_pairs": len(per_noise), "geometry": geometry,
            "controls": "One deterministic generation per question/prompt/control; not inflated by duplicating noise seeds"}


def paired_evaluation(before: list[dict], after: list[dict]) -> dict:
    def index(rows):
        values = {(r["question_id"], r["condition"], r["noise_seed"], r["prompt_id"]): r for r in rows}
        if len(values) != len(rows):
            raise RuntimeError("Duplicate Writer evaluation cells")
        return values
    base, trained = index(before), index(after)
    if set(base) != set(trained):
        raise RuntimeError("Baseline and trained Writer are not evaluated on identical question/noise/prompt cells")
    flips = {"incorrect_to_correct": 0, "correct_to_incorrect": 0, "both_correct": 0, "both_incorrect": 0}
    for key, left in base.items():
        right = trained[key]
        if left["query"] != right["query"] or left["gold"] != right["gold"]:
            raise RuntimeError("A paired evaluation prompt or gold changed")
        if left["condition"] != "matched":
            if left["image_sha256"] != right["image_sha256"]:
                raise RuntimeError("A frozen blank/donor control changed between baseline and trained Writer")
            continue
        old, new = bool(left["scorer"]["strict_correct"]), bool(right["scorer"]["strict_correct"])
        cell = "both_correct" if old and new else "correct_to_incorrect" if old else "incorrect_to_correct" if new else "both_incorrect"
        flips[cell] += 1
    return {"paired_matched_cells": sum(flips.values()), "exact_match_transitions": flips,
            "exact_match_delta_count": flips["incorrect_to_correct"] - flips["correct_to_incorrect"]}


@torch.no_grad()
def verify_training_teacher_readback(args, runtime, groups, teachers):
    """Recheck the original positive control before spending optimizer updates.

    This result uses an oracle latent directly and is never Writer accuracy.
    No member is silently dropped, replaced, or selected using this readback.
    """
    rows = []
    for group in groups:
        ids, _ = member_split(group["teacher_ids"])
        for tid in ids:
            pause_if_requested(runtime)
            pixels = decode_model_latents_unit_interval(runtime["pipe"].vae,
                teachers[tid].to(runtime["vae_device"]), clamp=True)
            query = group["question_variants"]["original_open"]
            ce = qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"],
                query=query, target=group["answer"], termination=runtime["termination"], lambda_eos=1.,
                require_image_grad=False, deterministic_ce=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            generation = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                image=pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"],
                max_new_tokens=32, do_sample=False)
            score = generation_diagnostics(generation, group["answer"], ce.target_ids[0,:ce.answer_token_count].cpu().tolist())
            rows.append({"teacher_id": tid, "question_id": group["question_id"], "query":query,
                "gold":group["answer"], "image_sha256":canonical_tensor_sha256(pixels.cpu()),
                **generation, "scorer":score})
            write_json(args.output_dir / "teacher-readback.json", {"scope":"direct oracle positive control, not Writer output", "rows":rows})
            if not score["strict_correct"] or not score["answer_followed_immediately_by_eos"]:
                raise RuntimeError(f"Previously correct teacher no longer passes raw answer/EOS readback: {tid}")


@torch.no_grad()
def evaluate(args, runtime, bank, teachers, phase: str) -> dict:
    directory = args.output_dir / phase
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        for relative, sha in saved["artifact_hashes"].items():
            if file_sha256(directory / relative) != sha:
                raise RuntimeError("Completed evaluation artifact changed")
        return saved["summary"]
    # Partial phase is recoverable by deterministic replay; retain its previous rows.
    rows_path = directory / "generations.jsonl"
    if rows_path.exists():
        rows_path.rename(directory / f"interrupted-generations-{time.time_ns()}.jsonl")
    rows, geometries = [], {}
    for group in bank["groups"]:
        pause_if_requested(runtime)
        context = runtime["contexts"][group["question_id"]]
        condition = context["condition"]
        outputs = []
        images = [("blank", None, context["blank"]), ("donor", None, context["donor"])]
        for i in range(args.eval_seeds):
            pause_if_requested(runtime)
            noise_seed = stable_seed(args.seed, "heldout-evaluation-noise", i)
            noise = torch.randn(context["source"].shape, generator=torch.Generator().manual_seed(noise_seed),
                                dtype=torch.float32).to(runtime["vae_device"])
            count=context.get("num_inference_steps",4)
            output = context.get("inference_sampler",runtime["sampler"])(source_latents=context["source"], noise_latents=noise,
                prompt_embeds=condition.prompt_embeds, prompt_attention_mask=condition.attention_mask,
                num_steps=count, edit_start_sigma=1.0 if is_official_flow(args) else START_SIGMA, return_trajectory=True)
            expected_sigmas = context.get("effective_sigmas", EFFECTIVE_SIGMAS)
            if (len(output.effective_sigmas) != count or output.trajectory is None or len(output.trajectory) != count+1
                    or any(abs(a-b) > 2e-6 for a, b in zip(output.effective_sigmas, expected_sigmas))):
                raise RuntimeError("Writer schedule differs from preregistration")
            if is_official_flow(args) and not torch.equal(output.trajectory[0], noise):
                raise RuntimeError("Official Writer must start from the supplied Gaussian noise")
            latent = output.latents.cpu()
            outputs.append(latent)
            pixels = decode_model_latents_unit_interval(runtime["pipe"].vae, output.latents, clamp=True).cpu()
            filename = hashlib.sha256(group["question_id"].encode()).hexdigest()[:16] + f"-seed-{i:02d}.pt"
            atomic_tensor(directory / filename, {"question_id": group["question_id"], "noise_seed": noise_seed,
                "latent": latent, "image": pixels, "trajectory": [x.cpu() for x in output.trajectory]})
            images.append(("matched", noise_seed, pixels))
        ids = group["teacher_ids"]
        _, held = member_split(ids)
        if getattr(args, "teacher_split", "holdout") == "all":
            held = []
        geometries[group["question_id"]] = bank_geometry(torch.cat(outputs), torch.cat([teachers[t] for t in ids]), ids, held)
        for image_kind, noise_seed, pixels in images:
            for prompt_id, query in group["question_variants"].items():
                pause_if_requested(runtime)
                loss = qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                    image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"],
                    query=query, target=group["answer"], termination=runtime["termination"], lambda_eos=1.,
                    require_image_grad=False, deterministic_ce=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                gold_ids = loss.target_ids[0, :loss.answer_token_count].cpu().tolist()
                generation = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                    image=pixels.to(runtime["reader_device"]), query=query, device=runtime["reader_device"],
                    max_new_tokens=32, do_sample=False)
                row = {"phase": phase, "question_id": group["question_id"], "condition": image_kind,
                    "noise_seed": noise_seed, "prompt_id": prompt_id, "query": query, "gold": group["answer"],
                    "image_sha256": canonical_tensor_sha256(pixels), **generation,
                    "answer_ce": float(loss.answer_loss), "eos_ce": float(loss.eos_loss),
                    "scorer": generation_diagnostics(generation, group["answer"], gold_ids)}
                rows.append(row)
                append_jsonl(rows_path, row)
        print(json.dumps({"stage": phase, "question": group["question_id"], "generation_rows": len(rows)}), flush=True)
    expected = len(bank["groups"]) * (args.eval_seeds + 2) * 5
    if len(rows) != expected:
        raise RuntimeError("Incomplete Writer evaluation coverage")
    summary = summarize_evaluation(rows, geometries)
    write_json(directory / "summary.json", summary)
    write_json(complete, {"summary": summary, "generation_rows": expected,
        "artifact_hashes": {p.name: file_sha256(p) for p in directory.iterdir()
                            if p.name in {"generations.jsonl", "summary.json"} or p.suffix == ".pt"}})
    return summary


def run(args) -> dict:
    import fcntl
    from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bank, teachers = load_teacher_bank(args.bank_manifest)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if commit != args.expected_commit or dirty:
        raise RuntimeError("U-Net run requires the exact immutable clean source commit")
    configure_strict_cuda_determinism(args.seed)
    stop_requested = [False]
    def stop_handler(_signal, _frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    groups = training_groups(bank, args.target_mode)
    official = is_official_flow(args)
    inference_steps=28 if args.model_variant=="base" else 4
    binding = {"schema": "latent-bank-unet/v2", "git_commit": commit, "source_hashes": source_hashes(),
        "bank_manifest_sha256": file_sha256(args.bank_manifest), "route": bank["route"],
        "steps": args.steps, "seed": args.seed, "lr": args.lr, "lora_rank": args.lora_rank,
        "eval_seeds": args.eval_seeds, "objective": "official target-noise flow matching" if official else "legacy source-anchored flow matching",
        "flow_protocol": args.flow_protocol, "upstream_commit": OFFICIAL_REFERENCE_COMMIT,
        "model_variant":args.model_variant,"inference_steps":inference_steps,
        "prompt_style": args.prompt_style if official else "legacy_decoded_source_diptych",
        "gradient_accumulation_steps": args.gradient_accumulation_steps, "weight_decay": args.weight_decay,
        "target_mode": args.target_mode,
        "raw_inference_sigmas": np.linspace(1.,1./inference_steps,inference_steps).tolist() if official else None,
        "training_timestep": "floor(1000*sigma)" if official else "1000*sigma",
        "dreamlite_dtype": "float32", "reader_dtype": "bfloat16", "source_sigma": 1.0 if official else START_SIGMA,
        "unet_input_keys": ["interpolated_flow_state", "source_latent", "event_condition", "effective_sigma", "time_ids"],
        "flow_state_construction_inputs": (["fresh_random_noise", "sampled_successful_target", "sigma"] if official
                                          else ["source_latent", "fresh_random_noise", "sampled_successful_target", "effective_sigma"]),
        "excluded_additional_condition_inputs": ["question", "answer", "target_latent", "teacher_id", "question_id"],
        "dreamlite_model_path": str(args.dreamlite.resolve()), "reader_model_path": str(args.reader_model.resolve()),
        "target_split": {g["question_id"]: dict(zip(("train", "heldout"), member_split(g["teacher_ids"]))) for g in groups},
        "generalization_scope": "single-question Writer mechanism" if len(bank["groups"]) == 1 else "seen-question Writer; no held-out-question claim"}
    identity_path = args.output_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != binding:
        raise RuntimeError("Output directory belongs to another bank/source/budget")
    write_json(identity_path, binding)
    runtime = load_runtime(args, bank)
    runtime["should_pause"] = lambda: stop_requested[0] or bool(args.deadline_unix and time.time() >= args.deadline_unix - 90)
    pause_if_requested(runtime)
    runtime_binding = {"snapshots": runtime["snapshots"], "termination": runtime["termination"],
        "additional_protocol_binding":runtime.get("protocol_binding",{}),
        "scheduler_config": dict(runtime["pipe"].scheduler.config),
        "effective_inference_sigmas": {k: list(v["effective_sigmas"]) for k,v in runtime["contexts"].items()},
        "condition_sha256": {k: canonical_tensor_sha256(v["condition"].prompt_embeds) for k,v in runtime["contexts"].items()}}
    runtime_path = args.output_dir / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime_binding:
        raise RuntimeError("Model/runtime identity changed")
    write_json(runtime_path, runtime_binding)
    binding = {**binding, **runtime_binding}
    pipe, reader = runtime["pipe"], runtime["reader"]
    frozen = frozen_versions(pipe, reader)
    parameters = lora_trainable_parameters(pipe.unet)
    initial_parameters = {n: p.detach().cpu().clone() for n,p in pipe.unet.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .999), eps=1e-8, weight_decay=args.weight_decay)
    checkpoint = args.output_dir / "checkpoint-latest.pt"
    verify_training_teacher_readback(args, runtime, groups, teachers)
    # Baseline is evaluated on the untrained adapter (LoRA B=0), before resume load.
    baseline = evaluate(args, runtime, bank, teachers, "baseline")
    step = 0
    if checkpoint.exists():
        if not args.resume:
            raise RuntimeError("Checkpoint exists; use --resume for exact recovery")
        loaded = load_training_checkpoint(checkpoint, trainable_module=pipe.unet, optimizer=optimizer,
                                          expected_manifest=binding)
        step = int(loaded["optimizer_step"])
        if not 0 <= step <= args.steps:
            raise RuntimeError("Checkpoint optimizer cursor is outside its bound budget")
        if step:
            write_json(args.output_dir / "metrics" / f"step-{step:06d}.json", loaded["trainer_state"]["metrics_row"])
    elif args.resume and (args.output_dir / "training.jsonl").exists():
        raise RuntimeError("Metrics exist without a recoverable optimizer checkpoint")
    # Canonical metric files recover the tiny checkpoint-to-log write interval.
    log_rows = []
    for i in range(1, step + 1):
        metric = json.loads((args.output_dir / "metrics" / f"step-{i:06d}.json").read_text(encoding="utf-8"))
        if metric["optimizer_step"] != i:
            raise RuntimeError("Optimizer metric sequence is corrupt")
        log_rows.append(metric)
    log_path = args.output_dir / "training.jsonl"
    temporary_log = log_path.with_suffix(".jsonl.tmp")
    temporary_log.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in log_rows), encoding="utf-8")
    temporary_log.replace(log_path)
    question_updates = Counter(q for row in log_rows for q in {m["question_id"] for m in row["microbatches"]})
    teacher_updates = Counter(t for row in log_rows for t in {m["teacher_id"] for m in row["microbatches"]})
    started = time.monotonic()
    while step < args.steps:
        pause_if_requested(runtime)
        optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for micro in range(args.gradient_accumulation_steps):
            draw_index = step * args.gradient_accumulation_steps + micro
            group, teacher_id, noise_seed, sigma = balanced_draw(groups, args.seed, draw_index,
                                                               max_sigma=1.0 if official else START_SIGMA)
            context = runtime["contexts"][group["question_id"]]
            noise = torch.randn(context["source"].shape, generator=torch.Generator().manual_seed(noise_seed),
                                dtype=torch.float32).to(runtime["vae_device"])
            target = teachers[teacher_id].to(runtime["vae_device"])
            state, true_velocity = (official_flow_bridge(noise, target, sigma) if official
                                    else anchored_flow_bridge(context["source"], noise, target, sigma))
            condition = context["condition"]
            prediction = predict_velocity(runtime["sampler"], state, context["source"], sigma,
                condition.prompt_embeds, condition.attention_mask, integer_timestep=official)
            loss = (prediction.float() - true_velocity.float()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite U-Net flow-matching loss")
            (loss / args.gradient_accumulation_steps).backward()
            microbatches.append({"question_id": group["question_id"], "teacher_id": teacher_id,
                "noise_seed": noise_seed, "effective_sigma": sigma, "flow_matching_mse": float(loss.detach())})
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any((g != 0).any() for g in grads):
            raise RuntimeError("U-Net LoRA has no finite nonzero gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Nonfinite U-Net adapter parameter")
        step += 1
        row = {"optimizer_step": step, "question_id": group["question_id"], "teacher_id": teacher_id,
               "noise_seed": noise_seed, "effective_sigma": sigma,
               "flow_matching_mse": sum(m["flow_matching_mse"] for m in microbatches) / len(microbatches),
               "microbatches": microbatches,
               "lora_grad_norm_before_clip": norm, "elapsed_since_resume_seconds": time.monotonic() - started}
        # Exact optimizer + all RNG state at every step: termination never requires inventing missing updates.
        save_training_checkpoint(checkpoint, trainable_module=pipe.unet, optimizer=optimizer,
            epoch=0, episode_cursor=step, optimizer_step=step, manifest=binding, trainer_state={"metrics_row": row})
        write_json(args.output_dir / "metrics" / f"step-{step:06d}.json", row)
        append_jsonl(args.output_dir / "training.jsonl", row)
        question_updates.update({m["question_id"] for m in microbatches})
        teacher_updates.update({m["teacher_id"] for m in microbatches})
        if step % 16 == 0 or step == 1:
            print(json.dumps({"stage": "unet_training", **row}), flush=True)
            frozen_audit(pipe, reader, frozen)
        del loss, prediction, state, target, true_velocity, grads
        if stop_requested[0] or (args.deadline_unix and time.time() >= args.deadline_unix - 90):
            result = {"status": "paused", "optimizer_steps": step, "checkpoint": str(checkpoint),
                      "reason": "signal or configured deadline; resume preserves optimizer/RNG"}
            write_json(args.output_dir / "terminal.json", result)
            return result
    # Keep a durable final adapter/optimizer before costly Reader evaluation.
    atomic_tensor(args.output_dir / "checkpoint-final.pt", torch.load(checkpoint, map_location="cpu", weights_only=False))
    after = evaluate(args, runtime, bank, teachers, "trained")
    paired = paired_evaluation(*[[json.loads(line) for line in (args.output_dir / phase / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
                                 for phase in ("baseline", "trained")])
    frozen_audit(pipe, reader, frozen)
    if source_hashes() != binding["source_hashes"] or file_sha256(args.bank_manifest) != binding["bank_manifest_sha256"]:
        raise RuntimeError("Source or bank manifest changed during U-Net training")
    load_teacher_bank(args.bank_manifest)
    for snapshot in runtime["snapshots"].values():
        verify_snapshot_binding(snapshot)
    runtime.get("verify_additional_bindings",lambda:None)()
    delta = sum(float((p.detach().cpu() - initial_parameters[n]).double().square().sum())
                for n,p in pipe.unet.named_parameters() if p.requires_grad)
    if not delta > 0:
        raise RuntimeError("No actual U-Net parameter update occurred")
    result = {"status": "completed", "route": bank["route"], "optimizer_steps": step,
        "unet_trainable_parameters": sum(p.numel() for p in parameters), "unet_adapter_delta_l2": delta**.5,
        "optimizer_updates_by_question": dict(question_updates), "optimizer_updates_by_teacher": dict(teacher_updates),
        "baseline": baseline, "trained": after, "paired_evaluation": paired, "generalization_scope": binding["generalization_scope"],
        "success_interpretation": "Training completion is not QA success; inspect paired raw exact match, controls and mode coverage",
        "oracle_question_coverage": bank.get("question_coverage"),
        "excluded_questions_without_success": bank.get("excluded_question_ids", []),
        "route_comparison_limit": "Same total optimizer budget does not imply same per-question exposure when bank question counts differ; route EM differences are not a causal bank-quality comparison",
        "checkpoint_sha256": file_sha256(args.output_dir / "checkpoint-final.pt")}
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "terminal.json", {"status": "completed", "optimizer_steps": step,
        "result_sha256": file_sha256(args.output_dir / "result.json"), "checkpoint_sha256": result["checkpoint_sha256"]})
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dreamlite", type=Path)
    p.add_argument("--model-variant",choices=("mobile","base"),default="mobile")
    p.add_argument("--teacher-dreamlite",type=Path)
    p.add_argument("--official-source",type=Path)
    p.add_argument("--base-manifest",type=Path)
    p.add_argument("--reader-model", type=Path)
    p.add_argument("--dreamlite-device", default="cuda:0")
    p.add_argument("--reader-device", default="cuda:1")
    p.add_argument("--expected-commit")
    p.add_argument("--steps", type=int, default=3500)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--flow-protocol", choices=("official", "legacy_anchored"), default="official")
    p.add_argument("--prompt-style", choices=("official_raw", "mobile_diptych"), default="official_raw")
    p.add_argument("--target-mode", choices=("bank", "single"), default="bank")
    p.add_argument("--eval-seeds", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--deadline-unix", type=float, default=0.)
    return p


def main() -> int:
    args = parser().parse_args()
    if (args.steps <= 0 or args.eval_seeds < 2 or args.lora_rank <= 0 or args.lr <= 0
            or args.gradient_accumulation_steps <= 0 or args.weight_decay < 0):
        raise ValueError("Positive training budget/rank/lr and at least two held-out noise seeds required")
    if args.validate_only:
        bank, teachers = load_teacher_bank(args.bank_manifest)
        print(json.dumps({"status": "bank_validated", "route": bank["route"], "teachers": len(teachers),
                          "questions": len(bank["groups"])}))
        return 0
    if not args.dreamlite or not args.reader_model or not args.expected_commit:
        raise ValueError("Training needs locked DreamLite, Reader and exact commit")
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "optimizer_steps": result["optimizer_steps"]}), flush=True)
        return 0 if result["status"] == "completed" else 75
    except TrainingPaused as error:
        checkpoint = args.output_dir / "checkpoint-latest.pt"
        step = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["optimizer_step"]) if checkpoint.exists() else 0
        write_json(args.output_dir / "terminal.json", {"status": "paused", "optimizer_steps": step,
            "checkpoint": str(checkpoint) if checkpoint.exists() else None, "reason": str(error)})
        return 75
    except BaseException as error:
        write_json(args.output_dir / "failure.json", {"status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "time_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
