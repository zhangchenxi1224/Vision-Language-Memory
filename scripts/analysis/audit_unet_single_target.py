"""Causal probes on the completed single-target run; original artifacts are read-only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch
import torch.nn.functional as F
from scripts.train import train_latent_bank_unet as training
from scripts.train.run_unet_learnability import verify_bank
from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism
from vision_memory.training.checkpoint import load_trainable_weights
from vision_memory.training.latent_bank_unet import file_sha256
from vision_memory.training.learnability import teacher_order, training_pair, evaluation_seed


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def spectrum(value):
    value = value.detach().float()
    h, w = value.shape[-2:]
    fy = torch.fft.fftfreq(h, device=value.device)[:, None]
    fx = torch.fft.fftfreq(w, device=value.device)[None, :]
    radius = (fx.square() + fy.square()).sqrt()
    power = torch.fft.fft2(value, norm="ortho").abs().square()
    return {name: float(power[..., mask].sum() / power.sum().clamp_min(1e-20))
            for name, mask in [("low_lt_0125", radius < .125),
                               ("mid_0125_025", (radius >= .125) & (radius < .25)),
                               ("high_ge_025", radius >= .25)]}


def run(args):
    config = json.loads(args.config.read_text())
    bank, teachers = verify_bank(config)
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(config["seed"])
    checkpoint = args.original / "single-rank4/checkpoint-2048.pt"
    original_sha = file_sha256(checkpoint)
    assert original_sha == "44945f17849494772536234a5c4182f72b2ed8e004223e2fe48232d5eed9246b"
    runtime_args = SimpleNamespace(bank_manifest=Path(config["bank_manifest"]),
        dreamlite=Path(config["dreamlite"]), reader_model=Path(config["reader_model"]),
        seed=config["seed"], lora_rank=4, dreamlite_device="cuda:0", reader_device="cuda:1")
    rt = training.load_runtime(runtime_args, bank)
    pipe, reader, sampler = rt["pipe"], rt["reader"], rt["sampler"]
    group = bank["groups"][0]
    ctx = rt["contexts"][group["question_id"]]
    source, condition = ctx["source"], ctx["condition"]
    ids = teacher_order(bank, config["anchor_run"])
    noise_seed, teacher_id = training_pair("single", 0, ids, config["seed"])
    target = teachers[teacher_id].to(source.device)
    saved = load_trainable_weights(checkpoint, trainable_module=pipe.unet)
    assert saved["optimizer_step"] == 2048
    assert saved["manifest"]["bank_sha256"] == config["bank_sha256"]
    assert saved["manifest"]["anchor_id"] == teacher_id
    lora_names = {n for n, p in pipe.unet.named_parameters() if p.requires_grad}
    sampler.checkpoint_unet = True
    report = {"started_epoch": time.time(), "hostname": socket.gethostname(),
        "source_commit": args.expected_commit, "original_checkpoint_sha256": original_sha,
        "bank_sha256": config["bank_sha256"], "teacher_id": teacher_id,
        "noise_seed": noise_seed, "training_scope": "one fixed question, noise and teacher",
        "qa": [], "arms": {}, "checks": {}}

    def save():
        training.write_json(args.output / "diagnosis.json", report)

    def check_time():
        if time.time() > args.deadline - 120:
            raise RuntimeError("Diagnostic lease deadline reached; completed evidence preserved")

    def forward(seed=noise_seed):
        check_time()
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(seed),
                            dtype=source.dtype).to(source.device)
        return sampler(source_latents=source, noise_latents=noise,
            prompt_embeds=condition.prompt_embeds, prompt_attention_mask=condition.attention_mask,
            num_steps=4, edit_start_sigma=.5).latents

    def qa_loss(z, grad):
        pixels = decode_model_latents_unit_interval(pipe.vae, z, clamp=True)
        loss = qwen3vl_answer_eos_ce(model=reader, processor=rt["processor"],
            image=pixels[0].to(rt["reader_device"]), device=rt["reader_device"],
            query=group["question_variants"]["original_open"], target=group["answer"],
            termination=rt["termination"], lambda_eos=1., require_image_grad=grad,
            deterministic_ce=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        return loss.answer_loss + loss.eos_loss, loss, pixels

    @torch.no_grad()
    def qa(label, z, extra=None):
        check_time()
        combined, losses, pixels = qa_loss(z, False)
        generation = generate_short_answer(model=reader, processor=rt["processor"],
            image=pixels.to(rt["reader_device"]), query=group["question_variants"]["original_open"],
            device=rt["reader_device"], do_sample=False, max_new_tokens=32)
        score = generation_diagnostics(generation, group["answer"],
            losses.target_ids[0, :losses.answer_token_count].cpu().tolist())
        row = {"label": label, "raw": generation["raw"], "strict_correct": score["strict_correct"],
            "answer_and_eos": score["answer_followed_immediately_by_eos"],
            "eos_reached": generation["eos_reached"], "answer_ce": float(losses.answer_loss),
            "eos_ce": float(losses.eos_loss), "rms_to_target": rms(z-target),
            "latent_sha256": canonical_tensor_sha256(z), "image_sha256": canonical_tensor_sha256(pixels),
            **(extra or {})}
        report["qa"].append(row)
        save()
        print(json.dumps({"qa": row}), flush=True)
        return row

    def gradient(checkpointed):
        sampler.checkpoint_unet = checkpointed
        pipe.unet.zero_grad(set_to_none=True)
        z = forward()
        loss = (z-target).square().mean()
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in pipe.unet.named_parameters()
                     if n in lora_names and p.grad is not None}
        value, pred = float(loss), z.detach()
        del loss, z
        pipe.unet.zero_grad(set_to_none=True)
        return value, pred, gradients

    params = dict(pipe.unet.named_parameters())
    report["parameters"] = {"lora": sum(params[n].numel() for n in lora_names),
                            "total_unet": sum(p.numel() for p in params.values()),
                            "lora_tensors": len(lora_names)}
    report["parameters"]["fraction"] = report["parameters"]["lora"] / report["parameters"]["total_unet"]
    versions = training.frozen_versions(pipe, reader)
    with torch.no_grad():
        student = forward()
    old_output = torch.load(args.original / "single-rank4/outputs/step2048-train-000.pt",
                            map_location="cpu", weights_only=True)["latent"]
    report["checks"]["saved_output_replay_bitwise"] = torch.equal(student.cpu(), old_output)
    report["checks"]["saved_output_max_error"] = float((student.cpu()-old_output).abs().max())
    assert report["checks"]["saved_output_replay_bitwise"]
    assert all(torch.equal(params[n].detach().cpu(), saved["trainable_state"][n]) for n in lora_names)
    report["checks"]["checkpoint_weights_bitwise"] = True
    teacher_row = qa("original_correct_teacher", target)
    assert teacher_row["strict_correct"] and teacher_row["answer_and_eos"]
    teacher_record = next(t for t in bank["teachers"] if t["teacher_id"] == teacher_id)
    assert teacher_row["image_sha256"] == teacher_record["qa"]["image_sha256"]["matched"]
    qa("original_student_2048", student)
    report["spectrum"] = {"teacher": spectrum(target), "student": spectrum(student),
                          "residual": spectrum(target-student)}
    for alpha in [.5, .75, .9, .95, .99]:
        qa(f"interpolation_{alpha}", student.lerp(target, alpha), {"alpha": alpha,
            "scope": "this single straight-line direction only; not a global basin estimate"})
    for factor in [2, 4]:
        smooth = F.interpolate(F.avg_pool2d(target, factor), size=target.shape[-2:], mode="bilinear", align_corners=False)
        qa(f"teacher_lowpass_{factor}", smooth, {"perturbation": "average pooling and bilinear upsampling"})

    loss0, pred0, grad0 = gradient(False)
    loss1, pred1, grad1 = gradient(True)
    assert grad0.keys() == grad1.keys()
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grad0.values()))
    diff = math.sqrt(sum(float((grad1[n]-g).double().square().sum()) for n, g in grad0.items()))
    report["gradient"] = {"loss_no_checkpoint": loss0, "loss_checkpoint": loss1,
        "prediction_max_difference": float((pred0-pred1).abs().max()), "norm": norm,
        "checkpoint_relative_gradient_difference": diff/max(norm, 1e-20),
        "tensors_with_gradient": len(grad0), "nonzero_tensors": sum(bool(g.count_nonzero()) for g in grad0.values()),
        "finite_difference": []}
    assert norm > 0 and diff / norm < 1e-4
    originals = {n: params[n].detach().clone() for n in grad0}
    directions = {n: g.to(source.device)/norm for n, g in grad0.items()}
    for eps in [1e-2, 1e-3]:
        values = []
        with torch.no_grad():
            for sign in [-1, 1]:
                for n, p0 in originals.items():
                    params[n].copy_(p0 + sign*eps*directions[n])
                values.append(float((forward()-target).square().mean()))
            for n, p0 in originals.items():
                params[n].copy_(p0)
        numeric = (values[1]-values[0])/(2*eps)
        report["gradient"]["finite_difference"].append({"epsilon": eps,
            "minus_loss": values[0], "plus_loss": values[1], "numeric_directional_derivative": numeric,
            "autograd_directional_derivative": norm, "relative_error": abs(numeric-norm)/max(norm, 1e-20)})
    assert any(r["relative_error"] < .05 for r in report["gradient"]["finite_difference"])
    del grad0, grad1, originals, directions, pred0, pred1
    probe = student.detach().clone().requires_grad_(True)
    loss, _, _ = qa_loss(probe, True)
    qgrad = torch.autograd.grad(loss, probe)[0].detach()
    msegrad = 2*(probe.detach()-target)/target.numel()
    report["gradient"]["latent_mse_vs_qa_eos_cosine"] = float(F.cosine_similarity(msegrad.flatten(), qgrad.flatten(), dim=0))
    report["gradient"]["latent_qa_gradient_rms"] = rms(qgrad)
    del probe, loss, qgrad, msegrad
    # Do not call the version-counter audit after the deliberate finite-difference
    # changes to trainable LoRA; it excludes LoRA and still checks frozen weights.
    training.frozen_audit(pipe, reader, versions)
    report["checks"]["frozen_modules_audit_passed"] = True
    report["audit_completed_epoch"] = time.time()
    save()
    print(json.dumps({"audit_complete": True, "gradient": report["gradient"], "parameters": report["parameters"]}), flush=True)

    # Matched factorial controls: all start from exactly the same 2048-step
    # checkpoint, all reset AdamW, all use 64 updates and the same fixed pair.
    # Test noises are reported only; never select an arm from them.
    base_weights = {n: p.detach().cpu().clone() for n, p in params.items()}
    external_versions = {f"{label}.{name}": int(p._version)
        for label, module in [("vae", pipe.vae), ("reader", reader), ("condition", pipe.text_encoder)]
        for name, p in module.named_parameters()}
    arms = [("lora_mse_lr1e4", "lora", 1e-4, 0.), ("lora_mse_lr1e5", "lora", 1e-5, 0.),
            ("full_mse_lr1e4", "full", 1e-4, 0.), ("full_mse_lr1e5", "full", 1e-5, 0.),
            ("lora_mse_qa_lr1e4", "lora", 1e-4, .1)]
    for name, scope, lr, qa_weight in arms:
        check_time()
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(base_weights[n].to(p.device))
                p.requires_grad_(scope == "full" or n in lora_names)
        pipe.unet.zero_grad(set_to_none=True)
        with torch.no_grad():
            restart_error = float((forward()-student).abs().max())
        assert restart_error < 1e-6, "Control did not start from the same original endpoint"
        optimizer = torch.optim.AdamW([p for p in params.values() if p.requires_grad], lr=lr, weight_decay=0.)
        directory = args.output / name
        directory.mkdir()
        report["arms"][name] = {"status": "running", "scope": scope, "lr": lr, "qa_weight": qa_weight,
            "optimizer_reset": True, "steps": args.steps, "initial_checkpoint_sha256": original_sha,
            "restart_output_max_error": restart_error}
        save()
        started = time.time()
        try:
            for step in range(1, args.steps+1):
                optimizer.zero_grad(set_to_none=True)
                z = forward()
                mse = (z-target).square().mean()
                objective = mse
                value_qa = None
                if qa_weight:
                    value_qa, _, _ = qa_loss(z, True)
                    objective = mse + qa_weight*value_qa
                assert torch.isfinite(objective)
                objective.backward()
                trainable = [p for p in params.values() if p.requires_grad]
                gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
                optimizer.step()
                row = {"arm": name, "step": step, "mse_before_update": float(mse.detach()),
                       "qa_eos_before_update": float(value_qa.detach()) if value_qa is not None else None,
                       "gradient_norm": gn, "epoch": time.time()}
                training.write_json(directory / "status.json", row)
                if step % 16 == 0: print(json.dumps(row), flush=True)
                del z, mse, objective, value_qa
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): result = forward()
            row = qa(name, result, {"scope": scope, "lr": lr, "qa_weight": qa_weight, "steps": args.steps})
            training.atomic_tensor(directory / "endpoint.pt", {"latent": result.cpu(), "teacher_id": teacher_id, "noise_seed": noise_seed})
            # Keep the actual learned weights for a successful or failed causal
            # control; this is deliberately separate from the original checkpoints.
            training.atomic_tensor(directory / "weights.pt", {"scope": scope, "arm": name,
                "initial_checkpoint_sha256": original_sha,
                "trainable_state": {n: p.detach().cpu() for n, p in params.items() if p.requires_grad}})
            report["arms"][name].update(status="completed", result=row, elapsed_seconds=time.time()-started)
            for i in range(2):
                with torch.no_grad(): z_test = forward(evaluation_seed(config["seed"], "test", i))
                qa(name+f"_test_{i}", z_test, {"scope": "two diagnostic heldout noises; no arm selection"})
            del result
        except torch.cuda.OutOfMemoryError as error:
            report["arms"][name].update(status="oom", error=str(error), elapsed_seconds=time.time()-started)
            print(json.dumps({"arm": name, "status": "oom"}), flush=True)
            # Allocation failure is a missing control result, never scientific failure.
        finally:
            pipe.unet.zero_grad(set_to_none=True)
            z = mse = objective = value_qa = None
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            save()
    assert file_sha256(checkpoint) == original_sha
    assert external_versions == {f"{label}.{name}": int(p._version)
        for label, module in [("vae", pipe.vae), ("reader", reader), ("condition", pipe.text_encoder)]
        for name, p in module.named_parameters()}
    report["status"] = "completed"
    report["completed_epoch"] = time.time()
    save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--steps", type=int, default=64)
    args = p.parse_args()
    assert socket.gethostname().startswith("vlm-unet-diag-h200x2-20260909--")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == args.expected_commit
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    assert args.steps == 64
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with ExitStack() as locks:
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
            devices = {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines()}
            selected = {devices[0], devices[1]}
            for uid in sorted(selected):
                handle = locks.enter_context((Path("/tmp") / f"vlm-oracle-writer-{uid}.lock").open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
            assert not selected.intersection(active.splitlines()), "GPU occupied; do not interrupt its owner"
            run(args)
    except BaseException:
        training.write_json(args.output / "failure.json", {"error": traceback.format_exc(), "epoch": time.time()})
        raise


if __name__ == "__main__":
    main()
