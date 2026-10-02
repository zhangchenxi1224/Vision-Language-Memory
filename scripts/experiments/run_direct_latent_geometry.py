"""Direct final-latent geometry: FP32 frozen VAE, BF16 Reader, no U-Net execution."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiments import run_r11_mcq_open_multistart as old
from scripts.experiments import run_r11_open_answer_replay as replay
from scripts.experiments import direct_geometry_eos_training as training
from scripts.train import latent_r11_vae_oracle as legacy
from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism
from vision_memory.training.direct_latent_geometry import panel, initial_array, CHECKPOINTS, DISTRIBUTIONS, study_members, spectrum, question_prompts
from vision_memory.reader.open_eos import assistant_termination_contract
from vision_memory.training.r10_alignment import R10_SELECTION_SEED
from vision_memory.training.r11_new_oracle import R11_NEW_TARGET_IDS, R11_NEW_TARGETS_PAYLOAD_SHA256


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): replay.sha256_file(p)
            for parent in (ROOT / "src", ROOT / "scripts") for p in sorted(parent.rglob("*.py"))}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_run(directory, spec):
    terminal = load(directory / "terminal.json")
    if terminal.get("status") != "completed" or terminal.get("optimizer_steps") != 256:
        raise RuntimeError(f"Preserved incomplete run requires audit: {directory}")
    if any(terminal.get(k) != v for k, v in spec.items()):
        raise RuntimeError("Run identity mismatch")
    if len(rows(directory / "metrics.jsonl")) != 256:
        raise RuntimeError("Missing optimizer receipts")
    for filename, expected in (("latent_index.jsonl", list(range(257))), ("checkpoint_index.jsonl", list(CHECKPOINTS))):
        index = rows(directory / filename)
        if [r["optimizer_step"] for r in index] != expected:
            raise RuntimeError("Incomplete trajectory/checkpoint grid")
        for r in index:
            p = (directory / r["path"]).resolve()
            if not p.is_relative_to(directory.resolve()) or replay.sha256_file(p) != r["file_sha256"]:
                raise RuntimeError("Artifact path or SHA mismatch")
    if replay.sha256_file(directory / "endpoint_raw.pt") != terminal["endpoint_file_sha256"]:
        raise RuntimeError("Endpoint changed")
    generations = rows(directory / "generations.jsonl")
    expected = {(c,p) for c in ('matched','blank','fixed_donor') for p in question_prompts()}
    if len(generations) != len(expected) or {(r['condition'],r['prompt_id']) for r in generations} != expected:
        raise RuntimeError("Incomplete endpoint evaluation")
    if any(r['query'] != question_prompts()[r['prompt_id']] or not r['gold_eos_appended'] for r in generations):
        raise RuntimeError('Prompt or EOS protocol changed')
    probes=rows(directory / 'checkpoint_generations.jsonl')
    if [r['optimizer_step'] for r in probes] != list(CHECKPOINTS):
        raise RuntimeError('Missing step-wise generation curve')
    matched=[r for r in generations if r['condition']=='matched']
    original=next(r for r in matched if r['prompt_id']=='original_open')
    return {**spec, "qa_pass": original['scorer']['strict_correct'],
            "robust_qa_pass": all(r['scorer']['strict_correct'] for r in matched),
            "answer_prefix_correct": original['scorer']['answer_prefix_token_exact'],
            "overgeneration": original['scorer']['overgeneration'],
            "prompt_correct": {r['prompt_id']:r['scorer']['strict_correct'] for r in matched},
            "blank_correct": sum(r['scorer']['strict_correct'] for r in generations if r['condition']=='blank'),
            "donor_correct": sum(r['scorer']['strict_correct'] for r in generations if r['condition']=='fixed_donor'),
            "elapsed_seconds": terminal["elapsed_seconds"], "run_path": str(directory)}


def summarize(root, config):
    completed = []
    for lane in range(2):
        for spec in config["runs"][lane::2]:
            directory = root / f"lane-{lane}" / "runs" / spec["run_id"]
            if (directory / "terminal.json").exists():
                completed.append(verify_run(directory, spec))
    result = {"schema": "vision_memory.direct-latent-geometry.summary.v1",
              "completed_count": len(completed), "planned_count": len(config["runs"]),
              "complete": len(completed) == len(config["runs"]), "formal_shared_writer_success": False,
              "per_run": completed, "success_count": sum(r["qa_pass"] for r in completed), "cells": []}
    for study, labels in (("distribution", DISTRIBUTIONS), ("scale", (.25,.5,1.,2.,4.)), ("gaussian_density", ("all32",))):
        for label in labels:
            planned = study_members(config["runs"], study, label)
            actual = study_members(completed, study, label)
            result["cells"].append({"study": study, "label": label, "planned": len(planned),
                                    "completed": len(actual), "success": sum(r["qa_pass"] for r in actual)})
    if result["complete"]:
        ordered = sorted(completed, key=lambda r: r["run_id"])
        endpoint, initial = [], []
        for r in ordered:
            p = Path(r["run_path"])
            endpoint.append(torch.load(p / "endpoint_raw.pt", map_location="cpu", weights_only=True)["latent_fp32"].numpy().reshape(-1))
            initial.append(torch.load(p / "latents/step-000.pt", map_location="cpu", weights_only=True)["latent_fp32"].numpy().reshape(-1))
        end, start = np.asarray(endpoint, dtype=np.float64), np.asarray(initial, dtype=np.float64)
        success = np.asarray([r["qa_pass"] for r in ordered])
        for name, values in (("all_endpoint", end), ("all_delta", end-start),
                             ("successful_endpoint", end[success]), ("successful_delta", (end-start)[success])):
            result[name] = spectrum(values)
        result["isotropic_sample_rank_reference"] = spectrum(np.random.default_rng(912).standard_normal(end.shape))
        result["geometry_limits"] = ["single question; no between-task geometry", "success-selected distribution is not the full readable set",
                                     "rank is bounded by sample count; no low-dimensional or cluster claim from PCA alone",
                                     "Direct and Frozen input coordinates/initialization centers differ; success difference is not a causal estimate of reachable volume"]
        np.savez_compressed(root / "endpoints_and_initials.npz", endpoint=end.astype('float32'), initial=start.astype('float32'),
                            success=success, run_ids=np.asarray([r['run_id'] for r in ordered]))
    replay.write_json(root / "summary.json", result)
    return result


def run_lane(args):
    import fcntl
    import diffusers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lane.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = load(args.config)
        if (config["runs"] != panel() or config["vae_dtype"] != "float32" or config["reader_dtype"] != "bfloat16"
                or config['target']['inputs'] != question_prompts()
                or config['training'] != {'optimizer':'Adam','steps':256,'lr':.05,'lambda_eos':1.,'prompts':['original_open']}):
            raise ValueError("Protocol changed")
        commit = replay.command_output(["git", "rev-parse", "HEAD"])
        if commit != args.expected_commit or replay.command_output(["git", "status", "--porcelain"]):
            raise RuntimeError("Require exact clean training checkout")
        binding = {"commit": commit, "config_sha256": replay.sha256_file(args.config), "lane": args.lane,
                   "source_hashes": source_hashes()}
        identity = args.output_dir / "identity.json"
        if identity.exists() and load(identity) != binding:
            raise RuntimeError("Output belongs to different source/config")
        replay.write_json(identity, binding)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("Lane requires exactly two visible GPUs")
        configure_strict_cuda_determinism(0)
        for name in ('train','dev'):
            if replay.sha256_file(getattr(args,name)) != config['data_sha256'][name]:
                raise RuntimeError("Locked dataset changed")
        data = legacy.r5._load_data(args, optimizer_steps=0)
        targets = legacy.select_f1_targets(data.train_pools)
        if tuple(t.segment_id for t in targets) != R11_NEW_TARGET_IDS or legacy.r5.canonical_sha256([t.to_dict() for t in targets]) != R11_NEW_TARGETS_PAYLOAD_SHA256:
            raise RuntimeError("Locked target panel changed")
        target = config["target"]
        selected = targets[1]
        if (selected.segment_id != target["segment_id"] or selected.query.text != target["original"]["query"]
                or list(selected.query.choices) != target["original"]["choices"] or selected.query.target_index != target["original"]["answer_index"]
                or selected.query.target != target['scorer_metadata']['gold']):
            raise RuntimeError("Target content differs from actual data")
        snapshots = replay.snapshot_bindings(args)
        index = load(args.dreamlite / "model_index.json")
        library, cls = index["vae"]
        if library != "diffusers" or not cls.startswith("Autoencoder"):
            raise RuntimeError("Unexpected VAE class")
        vd, rd = torch.device("cuda:0"), torch.device("cuda:1")
        vae = getattr(diffusers,cls).from_pretrained(args.dreamlite,subfolder="vae",local_files_only=True,torch_dtype=torch.float32).to(vd).requires_grad_(False).eval()
        processor, reader = replay.r4._load_reader(args,rd,torch.bfloat16)
        reader.requires_grad_(False).eval()
        replay.frozen_audit(vae,reader)
        control_root = args.old_multistart_root
        controls_payload = torch.load(control_root / "fixed_controls.pt",map_location="cpu",weights_only=True)
        control_audit = load(control_root / "control_audit.json")
        donor = controls_payload["fixed_donor"]
        if canonical_tensor_sha256(donor) != control_audit["image_sha256"]["fixed_donor"]:
            raise RuntimeError("Stored donor image changed")
        with torch.no_grad():
            reference = legacy.encode_model_latent(vae,replay.r4._initial_rgb_tensor(resolution=1024,device=vd,dtype=torch.float32)).cpu()
            blank = legacy.decode_model_latents_unit_interval(vae,reference.to(vd),clamp=True).cpu()
        if tuple(reference.shape) != (1,4,128,128) or reference.dtype != torch.float32:
            raise RuntimeError('Unexpected VAE latent coordinates')
        controls = {"blank":blank,"fixed_donor":donor}
        termination=assistant_termination_contract(reader,processor)
        runtime={'vae':vae,'reader':reader,'processor':processor,'vae_device':vd,'reader_device':rd,
                 'target':target,'controls':controls,'termination':termination}
        old.save_tensor_payload(args.output_dir / "reference_and_controls.pt",{"reference":reference,**controls})
        manifest = {**binding,"protocol":config,"snapshots_start":snapshots,"target_from_data":selected.to_dict(),
                    "reference_sha256":canonical_tensor_sha256(reference),"runtime":replay.runtime_versions(),
                    "termination_contract":termination,
                    "only_trainable":"final_latent_fp32", "unet_forward_count":0,"condition_encoder_forward_count":0,
                    "donor_source":str(control_root / "fixed_controls.pt"),"donor_sha256":canonical_tensor_sha256(donor),
                    "donor_note":"fixed historical orange RGB; original BF16 decode retained as image control",
                    "vae_class":type(vae).__name__,"reader_class":type(reader).__name__,"frozen_start":replay.frozen_audit(vae,reader)}
        replay.write_json(args.output_dir / "manifest.json",manifest)
        completed=[]; durations=[]; started=time.monotonic()
        try:
            for spec in config["runs"][args.lane::2]:
                directory=args.output_dir / "runs" / spec["run_id"]
                if directory.exists():
                    completed.append(verify_run(directory,spec));continue
                reserve=max([900.,*[d*2+120 for d in durations]])
                if time.time()+reserve >= args.deadline:
                    break
                value,statistics=initial_array(reference.numpy(),spec)
                initial=torch.from_numpy(value)
                replay.write_json(args.output_dir / "status.json",{"state":"running","active_run":spec,"completed":len(completed),
                                                                          "initialization_statistics":statistics,"utc_epoch":time.time()})
                result=training.run_one(spec=spec,initial=initial,reference=reference,runtime=runtime,output_dir=args.output_dir)
                replay.write_json(directory / "initialization.json",statistics)
                verified=verify_run(directory,spec);completed.append(verified);durations.append(result['elapsed_seconds'])
                print(json.dumps({"stage":"run_complete","id":spec['run_id'],"qa_pass":verified['qa_pass'],"completed":len(completed),"seconds":result['elapsed_seconds']}),flush=True)
                replay.write_json(args.output_dir / "progress.json",{"completed":completed,"planned":48})
            if source_hashes()!=binding['source_hashes'] or replay.sha256_file(args.config)!=binding['config_sha256']:
                raise RuntimeError("Code/config changed while running")
            if {k:verify_snapshot_binding(v) for k,v in snapshots.items()}!=snapshots:
                raise RuntimeError("Model snapshot changed")
            status='completed' if len(completed)==48 else 'paused_at_run_boundary'
            terminal={"status":status,"completed":len(completed),"planned":48,"elapsed_seconds":time.monotonic()-started,
                      "snapshots_end_verified":True,"source_end_verified":True,"frozen_end":replay.frozen_audit(vae,reader)}
            replay.write_json(args.output_dir / "terminal.json",terminal)
            replay.write_json(args.output_dir / "status.json",terminal)
        except BaseException as exc:
            replay.write_json(args.output_dir / "status.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/experiments/direct_latent_geometry.json')
    p.add_argument('--lane',type=int,choices=[0,1],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--dreamlite',type=Path,required=True);p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--old-multistart-root',type=Path,required=True)
    args=p.parse_args()
    args.pairing_seed=0;args.schedule_seed=R10_SELECTION_SEED;args.split_seed=20260730
    args.selected_step_count=0;args.gradient_mode='full'
    run_lane(args)


if __name__=='__main__':main()
