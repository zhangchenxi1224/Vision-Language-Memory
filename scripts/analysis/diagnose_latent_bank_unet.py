"""Bounded, zero-update diagnosis of a completed Direct-to-Writer experiment."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from scripts.train import train_latent_bank_unet as training
from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism
from vision_memory.training.checkpoint import load_trainable_weights
from vision_memory.training.latent_bank_unet import (
    anchored_flow_bridge, file_sha256, load_teacher_bank, member_split,
    predict_velocity, stable_seed,
)


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


@torch.no_grad()
def diagnose(args):
    bank, teachers = load_teacher_bank(args.bank_manifest)
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(args.seed)
    runtime = training.load_runtime(args, bank)
    group = bank["groups"][0]
    assert len(bank["groups"]) == 1
    ctx = runtime["contexts"][group["question_id"]]
    source, cond = ctx["source"], ctx["condition"]
    train_ids, held_ids = member_split(group["teacher_ids"])
    mean = torch.stack([teachers[t] for t in train_ids]).mean(0).to(source.device)
    teacher = teachers[train_ids[0]].to(source.device)
    first_record = next(t for t in bank["teachers"] if t["teacher_id"] == train_ids[0])
    original = torch.load(Path(first_record["source_run"]) / "endpoint_raw.pt", map_location="cpu", weights_only=True)
    report = {"started_epoch":time.time(), "hostname":socket.gethostname(), "optimizer_updates":0,
              "bank_sha256":file_sha256(args.bank_manifest), "checkpoint_sha256":file_sha256(args.checkpoint),
              "train_teachers":len(train_ids), "heldout_teachers":len(held_ids), "qa":[], "flow_probes":{}}

    def save():
        training.write_json(args.output_dir / "diagnosis.json", report)

    def qa(label, z):
        pixels = decode_model_latents_unit_interval(runtime["pipe"].vae, z.to(source.device), clamp=True)
        loss = qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
            image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"],
            query=group["question_variants"]["original_open"], target=group["answer"],
            termination=runtime["termination"], lambda_eos=1., require_image_grad=False,
            deterministic_ce=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        generation = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
            image=pixels.to(runtime["reader_device"]), query=group["question_variants"]["original_open"],
            device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
        score = generation_diagnostics(generation, group["answer"], loss.target_ids[0,:loss.answer_token_count].cpu().tolist())
        row = {"label":label, "raw":generation["raw"], "strict_correct":score["strict_correct"],
               "answer_prefix":score["answer_prefix_token_exact"], "answer_ce":float(loss.answer_loss),
               "eos_ce":float(loss.eos_loss), "distance_to_source_rms":rms(z.to(source.device)-source),
               "distance_to_train_mean_rms":rms(z.to(source.device)-mean),
               "latent_sha256":canonical_tensor_sha256(z), "image_sha256":canonical_tensor_sha256(pixels)}
        report["qa"].append(row)
        save()
        print(json.dumps(row), flush=True)
        return pixels

    decoded = qa("successful_teacher_0", teacher)
    report["teacher_decode_bitwise_original"] = torch.equal(decoded.cpu(), original["image"])
    assert report["teacher_decode_bitwise_original"]
    qa("successful_teacher_1", teachers[train_ids[1]].to(source.device))
    qa("successful_teacher_2", teachers[train_ids[2]].to(source.device))
    qa("gray_source", source)
    qa("mean_of_77_training_teachers", mean)
    qa("midpoint_of_two_successful_teachers", (teacher+teachers[train_ids[1]].to(source.device))*.5)
    noise_seed = stable_seed(args.seed, "heldout-evaluation-noise", 0)
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(noise_seed), dtype=torch.float32).to(source.device)
    start = .5*source+.5*noise
    report["initial_noise_seed"] = noise_seed
    report["teacher_dispersion_rms"] = rms(torch.stack([teachers[t] for t in train_ids]).to(source.device)-mean)

    def velocity(state, sigma):
        return predict_velocity(runtime["sampler"], state, source, sigma, cond.prompt_embeds, cond.attention_mask)

    def integrate(steps):
        state = start.clone()
        for i in range(steps):
            state = state-(.5/steps)*velocity(state, .5*(steps-i)/steps)
        return state

    def flow_probes(label):
        rows = []
        for sigma in [.5,.375,.25,.125,.05]:
            for tid in train_ids[:4]+held_ids[:4]:
                target = teachers[tid].to(source.device)
                state, gold_velocity = anchored_flow_bridge(source, noise, target, sigma)
                pred = velocity(state, sigma)
                rows.append({"sigma":sigma, "teacher_id":tid, "training_teacher":tid in train_ids,
                             "velocity_mse":rms(pred-gold_velocity)**2,
                             "one_step_endpoint_error_rms":rms(state-sigma*pred-target)})
        report["flow_probes"][label] = rows
        save()
        print(json.dumps({"flow_probe_phase":label, "n":len(rows)}), flush=True)

    flow_probes("baseline")
    base = integrate(4)
    qa("baseline_manual_4step", base)
    load_trainable_weights(args.checkpoint, trainable_module=runtime["pipe"].unet)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report["adapter_load_bitwise"] = all(torch.equal(p.detach().cpu(), saved["trainable_state"][n])
        for n,p in runtime["pipe"].unet.named_parameters() if p.requires_grad)
    assert report["adapter_load_bitwise"]
    flow_probes("trained")
    official = runtime["sampler"](source_latents=source, noise_latents=noise,
        prompt_embeds=cond.prompt_embeds, prompt_attention_mask=cond.attention_mask,
        num_steps=4, edit_start_sigma=.5, return_trajectory=True)
    saved_generation = torch.load(next((args.checkpoint.parent / "trained").glob("*-seed-00.pt")),
                                  map_location="cpu", weights_only=True)
    report["trained_inference_replays_saved_latent"] = torch.equal(official.latents.cpu(), saved_generation["latent"])
    report["trained_inference_replay_max_error"] = float((official.latents.cpu()-saved_generation["latent"]).abs().max())
    assert report["trained_inference_replays_saved_latent"]
    manual = integrate(4)
    report["training_forward_vs_sampler_4step_max_error"] = float((manual-official.latents).abs().max())
    report["trained_vs_baseline_latent_rms"] = rms(official.latents-base)
    assert report["training_forward_vs_sampler_4step_max_error"] < 1e-5
    qa("trained_actual_4step", official.latents)
    qa("trained_manual_16step", integrate(16))
    qa("trained_manual_64step", integrate(64))
    for alpha in [.25,.5,.75,.9]:
        qa(f"trained_to_teacher_alpha_{alpha}", (1-alpha)*official.latents+alpha*teacher)
    exact_velocity = (start-teacher)/.5
    controlled = start.clone()
    for _ in range(4):
        controlled = controlled-.125*exact_velocity
    report["exact_velocity_control_target_max_error"] = float((controlled-teacher).abs().max())
    qa("exact_velocity_control", controlled)
    report["completed_epoch"] = time.time()
    report["status"] = "completed"
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ["qa","flow_probes"]}), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader-model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Cooperate with automatic Writer handoff, and use only the first GPU pair.
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    devices={int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines()}
    selected={devices[0],devices[1]}
    with ExitStack() as locks:
        for uuid in sorted(selected):
            handle=locks.enter_context((Path('/tmp')/f'vlm-oracle-writer-{uuid}.lock').open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
        assert not (selected & set(active.splitlines())), "Diagnostic GPU pair already occupied"
        diagnose(args)


if __name__ == "__main__":
    main()
