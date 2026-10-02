"""Old R11 P0/P1/P2: EOS audit, frozen checkpoint replay, paired A/B/C training.

Two separate GPUs retain the original BF16 VAE/Reader execution contract. Raw
32-token scientific generations and task-schema deployment generations are saved
to distinct files. This is a per-question oracle diagnostic, not shared memory.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiments import run_r11_mcq_open_multistart as old
from scripts.experiments import run_r11_open_answer_replay as replay
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, encode_model_latent
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer, normalize_short_answer
from vision_memory.reader.open_eos import (assistant_termination_contract, qwen3vl_answer_eos_ce,
    generation_diagnostics, deployment_max_tokens, normalize_deployment_answer)
from vision_memory.reader.qwen3vl import (qwen3vl_target_only_ce, _joint_prompt_target_tokenization,
                                        R3_QWEN_READER_RESIZE_CONTRACT)
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism

SCHEMA = "vision_memory.r11-open-eos-paired.v1"
STEPS = [0, 16, 32, 64, 128, 192, 256]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config["schema"] != SCHEMA.replace(".v1", "-config.v1") or config["seeds"] != list(range(8))
            or config["checkpoint_steps"] != STEPS or config["primary_endpoint"] != 256
            or config["optimizer"] != {"name": "Adam", "steps": 256, "lr": .05, "betas": [.9, .999],
                                       "eps": 1e-8, "weight_decay": 0., "amsgrad": False}
            or config["best_checkpoint_selection"] or config["extend_to_512"]):
        raise ValueError("Fixed paired protocol changed.")
    if config["arms"]["A"]["append_eos"] or set(config["arms"]) != {"A", "B", "C"}:
        raise ValueError("A must retain the old no-EOS loss.")
    for arm in ("B", "C"):
        if config["arms"][arm]["lambda_eos"] != 1.0 or not config["arms"][arm]["append_eos"]:
            raise ValueError("EOS arm must use separately weighted answer mean plus EOS lambda=1.")
    if (config["arms"]["C"]["training_prompts"] != ["original_open", "paraphrase_open"]
            or config["arms"]["A"]["training_prompts"] != ["original_open"]
            or config["arms"]["B"]["training_prompts"] != ["original_open"]):
        raise ValueError("Training prompt membership changed.")
    return config


def training_prompt(arm: str, step_zero: int) -> str:
    if arm not in ("A", "B", "C") or not 0 <= step_zero < 256:
        raise ValueError("Unknown arm or update.")
    return "paraphrase_open" if arm == "C" and step_zero % 2 else "original_open"


def verified_old_latent(root: Path, seed: int, arm: str, step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    directory = root / "runs" / f"noise-seed-{seed:02d}-{arm}"
    index = read_rows(directory / "latent_index.jsonl")
    matches = [x for x in index if x["optimizer_step"] == step]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate old latent index record.")
    row = matches[0]
    path = (directory / row["path"]).resolve(strict=True)
    if directory.resolve() not in path.parents or replay.sha256_file(path) != row["file_sha256"]:
        raise ValueError("Old checkpoint path/file SHA mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    latent = payload["latent_fp32"]
    if (payload["optimizer_step"] != step or latent.dtype != torch.float32
            or tuple(latent.shape) != (1, 4, 128, 128)
            or canonical_tensor_sha256(latent) != row["latent_sha256"]):
        raise ValueError("Old checkpoint tensor/step mismatch.")
    return latent, {"path": str(path), **row}


def teacher(runtime, image, prompt, *, eos=True, require_grad=False):
    common = {"model": runtime["reader"], "processor": runtime["processor"],
              "image": image[0].to(runtime["reader_device"]), "query": runtime["target"]["inputs"][prompt],
              "target": runtime["target"]["scorer_metadata"]["gold"], "device": runtime["reader_device"],
              "require_image_grad": require_grad, "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
              "deterministic_ce": True}
    if eos:
        return qwen3vl_answer_eos_ce(**common, termination=runtime["termination"], lambda_eos=1.0)
    return qwen3vl_target_only_ce(**common)


def tokenization_audit(runtime, config):
    rows = []
    processor = runtime["processor"]
    for example in config["p4_tokenization_examples"]:
        query = f"State the current {example['field']} preference. Return only a short phrase."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, ids = _joint_prompt_target_tokenization(processor, prompt, example["answer"])
        rows.append({**example, "answer_token_ids": ids[0].tolist(), "answer_token_count": ids.numel(),
                     "deployment_max_new_tokens": deployment_max_tokens(example["answer_type"]),
                     "scope": "tokenization_contract_only_no_memory_result"})
    return rows


@torch.no_grad()
def evaluate(runtime, config, image, *, run_id, arm, seed, step, output_dir, conditions, deployment=False):
    records = []
    with old.preserve_rng():
        for condition in conditions:
            selected_image = image if condition == "matched" else runtime["controls"][condition]
            for prompt in config["evaluation_prompts"]:
                loss = teacher(runtime, selected_image, prompt)
                gold_ids = loss.target_ids[0, :loss.answer_token_count].cpu().tolist()
                generated = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                    image=selected_image.to(runtime["reader_device"]), query=runtime["target"]["inputs"][prompt],
                    device=runtime["reader_device"], max_new_tokens=32)
                diagnostics = generation_diagnostics(generated, runtime["target"]["scorer_metadata"]["gold"], gold_ids)
                trained = prompt in config["arms"].get(arm, {"training_prompts": ["original_open"]})["training_prompts"]
                row = {"schema": SCHEMA + ".generation", "run_id": run_id, "arm": arm, "seed": seed,
                       "step": step, "condition": condition, "prompt_id": prompt,
                       "prompt_exposed_in_training": trained, "decoding": "raw_greedy_32_original_eos",
                       "latent_training_endpoint_selected": False, **generated, **diagnostics,
                       "teacher_forced_answer_ce": float(loss.answer_loss), "teacher_forced_eos_ce": float(loss.eos_loss),
                       "teacher_forced_answer_token_accuracy": float(loss.teacher_forced_answer_accuracy),
                       "teacher_forced_eos_token_accuracy": float(loss.teacher_forced_eos_accuracy)}
                replay.append_jsonl(output_dir / "raw_generations.jsonl", row)
                records.append(row)
                if deployment:
                    limit = deployment_max_tokens(config["deployment_generation"]["answer_type"])
                    serving = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                        image=selected_image.to(runtime["reader_device"]), query=runtime["target"]["inputs"][prompt],
                        device=runtime["reader_device"], max_new_tokens=limit, stop_on_newline=True)
                    cleaned = normalize_deployment_answer(serving["raw"])
                    replay.append_jsonl(output_dir / "deployment_generations.jsonl", {
                        "run_id": run_id, "arm": arm, "seed": seed, "step": step, "condition": condition,
                        "prompt_id": prompt, "decoding": "schema_budget_newline_stop", "max_new_tokens": limit,
                        **serving, "normalized_deployment_answer": cleaned,
                        "exact_match": cleaned == normalize_short_answer(runtime["target"]["scorer_metadata"]["gold"]),
                        "included_in_raw_scientific_metrics": False})
        print(json.dumps({"stage": "generation_checkpoint", "run_id": run_id, "step": step,
                          "records": len(records)}), flush=True)
    return records


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["step"], row["condition"], row["prompt_id"])].append(row)
    output = []
    for (arm, step, condition, prompt), values in sorted(groups.items()):
        if len({x["seed"] for x in values}) != len(values):
            raise ValueError("Duplicate seed in one evaluation cell.")
        fields = {key: sum(float(x[key]) for x in values) / len(values) for key in (
            "strict_correct", "answer_prefix_token_exact", "generated_answer_token_accuracy", "overgeneration",
            "teacher_forced_answer_token_accuracy", "teacher_forced_answer_ce", "teacher_forced_eos_ce")}
        output.append({"arm": arm, "step": step, "condition": condition, "prompt_id": prompt,
                       "count": len(values), "seeds": sorted(x["seed"] for x in values), **fields})
    cells = {(x["arm"], x["step"], x["condition"], x["prompt_id"]): x for x in output}
    dependence = []
    consistency = []
    for arm in sorted({x["arm"] for x in rows}):
        for prompt in ("original_open", "paraphrase_open", "new_rewrite_open"):
            matched = cells.get((arm, 256, "matched", prompt))
            if matched:
                dependence.append({"arm": arm, "prompt_id": prompt, **{
                    "matched_minus_" + control: matched["strict_correct"] - cells[(arm, 256, control, prompt)]["strict_correct"]
                    for control in ("blank", "fixed_donor") if (arm, 256, control, prompt) in cells}})
        endpoint = [x for x in rows if x["arm"] == arm and x["step"] == 256 and x["condition"] == "matched"]
        by_seed = defaultdict(dict)
        for row in endpoint:
            by_seed[row["seed"]][row["prompt_id"]] = row
        for seed, prompts in sorted(by_seed.items()):
            if len(prompts) == 3:
                consistency.append({"arm": arm, "seed": seed,
                    "original_and_paraphrase_both_correct": prompts["original_open"]["strict_correct"] and prompts["paraphrase_open"]["strict_correct"],
                    "all_three_correct": all(x["strict_correct"] for x in prompts.values()),
                    "all_three_same_normalized_answer": len({x["normalized"] for x in prompts.values()}) == 1,
                    "paraphrase_is_trained_for_C": arm == "C", "new_rewrite_never_trained": True})
    extension = []
    for arm in sorted({x["arm"] for x in rows}):
        fixed = [cells.get((arm, step, "matched", "original_open")) for step in (64, 128, 256)]
        if all(fixed):
            extension.append({"arm": arm, "may_discuss_512": all(
                fixed[i]["strict_correct"] < fixed[i+1]["strict_correct"]
                and fixed[i]["teacher_forced_answer_ce"] > fixed[i+1]["teacher_forced_answer_ce"] for i in (0, 1)),
                "automatically_extended": False})
    return {"schema": SCHEMA + ".summary", "raw_cells": output, "image_dependence": dependence,
            "paraphrase_consistency": consistency, "extension_diagnostic": extension, "formal_shared_memory_success": False,
            "metric_unit": "seed; prompt views and checkpoints are repeated measurements",
            "raw_32_token_generation_only": True}


def load_runtime(args, config):
    if not torch.cuda.is_available() or args.vae_device == args.reader_device:
        raise RuntimeError("Original BF16 execution requires two separate CUDA devices.")
    determinism = configure_strict_cuda_determinism(0)
    parent = json.loads((args.old_multistart_root / "manifest.json").read_text())
    bindings = replay.snapshot_bindings(args)
    if bindings != parent["model_snapshot_payloads_start"]:
        raise ValueError("Original model checkpoint binding changed.")
    vae_device, reader_device = torch.device(args.vae_device), torch.device(args.reader_device)
    vae = replay.load_vae(args.dreamlite, vae_device)
    processor, reader = replay.r4._load_reader(args, reader_device, torch.bfloat16)
    reader.requires_grad_(False).eval()
    replay.frozen_audit(vae, reader)
    target = dict(replay.load_config(ROOT / "configs/experiments/r11_open_answer_replay.json")["targets"][1])
    target["inputs"] = {**target["inputs"], "new_rewrite_open": config["new_rewrite_open"]}
    controls_file = args.old_multistart_root / "fixed_controls.pt"
    controls = torch.load(controls_file, map_location="cpu", weights_only=True)
    audit = json.loads((args.old_multistart_root / "control_audit.json").read_text())
    reference = controls["reference_fp32"]
    if canonical_tensor_sha256(reference) != replay.INITIAL_LATENT_SHA256:
        raise ValueError("Original blank reference changed.")
    for name in ("blank", "fixed_donor"):
        if canonical_tensor_sha256(controls[name]) != audit["image_sha256"][name]:
            raise ValueError("Frozen image control hash changed.")
    with torch.no_grad():
        reference_now = encode_model_latent(vae, replay.r4._initial_rgb_tensor(
            resolution=1024, device=vae_device, dtype=torch.bfloat16)).float().cpu()
        if not torch.equal(reference_now, reference):
            raise ValueError("Reference VAE encoding no longer reproduces old experiment.")
    termination = assistant_termination_contract(reader, processor)
    runtime_versions = replay.runtime_versions()
    if runtime_versions["packages"] != parent["runtime"]["packages"]:
        raise ValueError("Packages differ from original experiment; no unlabelled numerical migration.")
    runtime = {"vae": vae, "reader": reader, "processor": processor, "reference": reference,
               "controls": controls, "target": target, "termination": termination,
               "vae_device": vae_device, "reader_device": reader_device}
    replay.write_json(args.output_dir / "manifest.json", {"schema": SCHEMA + ".manifest", "config": config,
        "config_sha256": replay.sha256_file(args.config), "git_commit": replay.command_output(["git", "rev-parse", "HEAD"]),
        "runtime": runtime_versions, "determinism": determinism, "model_snapshot_payloads": bindings,
        "parent_manifest_sha256": replay.sha256_file(args.old_multistart_root / "manifest.json"),
        "termination_contract": termination, "actual_arms": args.arms, "actual_seeds": args.seeds,
        "mode": args.mode, "p4_tokenization_audit": tokenization_audit(runtime, config),
        "formal_shared_memory_success": False})
    return runtime


def replay_checkpoints(args, config, runtime):
    rows = []
    for arm in args.replay_arms:
        for seed in args.seeds:
            for step in STEPS:
                latent, audit = verified_old_latent(args.old_multistart_root, seed, arm, step)
                replay.append_jsonl(args.output_dir / "old_checkpoint_audit.jsonl", {"old_arm": arm, "seed": seed, **audit})
                with torch.no_grad():
                    image = decode_model_latents_unit_interval(runtime["vae"], latent.to(runtime["vae_device"], dtype=torch.bfloat16)).cpu()
                rows.extend(evaluate(runtime, config, image, run_id=f"old-{arm}-{seed:02d}",
                    arm="old_" + arm, seed=seed, step=step, output_dir=args.output_dir,
                    conditions=config["evaluation_conditions"] if step == 256 else ["matched"], deployment=step == 256))
    return rows


def train(args, config, runtime):
    rows = []
    for seed in args.seeds:
        initial = old.make_initial_latent(runtime["reference"], seed=seed, rho=.1)
        old_initial, _ = verified_old_latent(args.old_multistart_root, seed, "open", 0)
        if not torch.equal(initial["latent_fp32"], old_initial):
            raise ValueError("Paired seed no longer reproduces original starting tensor.")
        for arm in args.arms:
            configure_strict_cuda_determinism(0)
            run_id = f"seed-{seed:02d}-{arm}"
            directory = args.output_dir / "runs" / run_id
            directory.mkdir(parents=True, exist_ok=False)
            oracle = VAELatentOracle(vae=runtime["vae"], initial_latent=initial["latent_fp32"].to(runtime["vae_device"]),
                                    compute_dtype=torch.bfloat16)
            optimizer = torch.optim.Adam([oracle.latent_fp32], lr=.05, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
            replay.write_json(directory / "manifest.json", {"arm": arm, "seed": seed,
                "initial_tensor_sha256": initial["latent_sha256"], "optimizer": config["optimizer"],
                "train_prompts": config["arms"][arm]["training_prompts"], "append_eos": arm != "A"})
            for step in range(257):
                if step in STEPS:
                    old.save_tensor_payload(directory / f"step-{step:03d}.pt", {"latent_fp32": oracle.latent_fp32,
                        "optimizer": optimizer.state_dict(), "rng": old.capture_rng(), "step": step})
                    if arm == "A":
                        original, audit = verified_old_latent(args.old_multistart_root, seed, "open", step)
                        exact = torch.equal(original, oracle.latent_fp32.detach().cpu())
                        replay.append_jsonl(directory / "legacy_parity.jsonl", {"step": step, "exact_tensor_parity": exact,
                            "max_abs_difference": float((original - oracle.latent_fp32.detach().cpu()).abs().max()), "old": audit})
                        if not exact:
                            raise RuntimeError("A does not exactly replay old Open trajectory; stop paired comparison.")
                    with torch.no_grad():
                        image = oracle.image().cpu()
                    rows.extend(evaluate(runtime, config, image, run_id=run_id, arm=arm, seed=seed, step=step,
                        output_dir=args.output_dir, conditions=config["evaluation_conditions"] if step == 256 else ["matched"],
                        deployment=step == 256))
                if step == 256:
                    break
                optimizer.zero_grad(set_to_none=True)
                output = teacher(runtime, oracle.image(), training_prompt(arm, step), eos=arm != "A", require_grad=True)
                output.loss.backward()
                gradient = oracle.latent_fp32.grad
                if gradient is None or not torch.isfinite(gradient).all() or not torch.isfinite(output.loss):
                    raise RuntimeError("Invalid connected latent gradient/loss.")
                metric = {"run_id": run_id, "step": step + 1, "loss_before_step": float(output.loss.detach()),
                          "training_prompt": training_prompt(arm, step), "gradient_l2": float(gradient.double().norm())}
                if arm != "A":
                    metric.update(answer_ce=float(output.answer_loss.detach()), eos_ce=float(output.eos_loss.detach()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                replay.append_jsonl(directory / "metrics.jsonl", metric)
                if (step + 1) % 16 == 0:
                    print(json.dumps({"stage": "training", **metric}), flush=True)
                del output
            replay.frozen_audit(runtime["vae"], runtime["reader"])
            replay.write_json(directory / "terminal.json", {"status": "completed", "steps": 256})
            del oracle, optimizer
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/r11_open_eos_paired.json")
    p.add_argument("--old-multistart-root", type=Path, required=True)
    p.add_argument("--dreamlite", type=Path, required=True)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--vae-device", default="cuda:0")
    p.add_argument("--reader-device", default="cuda:1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("audit", "replay", "train", "all"), default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    p.add_argument("--replay-arms", nargs="+", choices=("open", "mcq"), default=["open"])
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(x not in range(8) for x in args.seeds):
        p.error("Seeds must be unique members of 0..7.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = load_config(args.config)
        runtime = load_runtime(args, config)
        rows = []
        if args.mode in ("replay", "all"):
            rows += replay_checkpoints(args, config, runtime)
        if args.mode in ("train", "all"):
            rows += train(args, config, runtime)
        summary = aggregate(rows)
        summary.update(actual_seeds=args.seeds, actual_arms=args.arms, complete_eight_seed_design=args.seeds == list(range(8)))
        replay.write_json(args.output_dir / "summary.json", summary)
        replay.write_json(args.output_dir / "terminal.json", {"status": "completed", "mode": args.mode,
            "elapsed_seconds": time.monotonic() - started, "raw_generation_count": len(rows), "formal_shared_memory_success": False})
        return 0
    except BaseException as error:
        replay.write_json(args.output_dir / "terminal.json", {"status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
            "formal_shared_memory_success": False})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
