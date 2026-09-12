"""Zero-update native DreamLite event/prompt control; no answer rasterization."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

RAW_EVENT = "R3 Train Standard Templates 03: For the indigo desk train 001123, remember that the preferred music is ambient."
PREFIX = ("Create a clean, legible memory note on the image. Use large black text on a plain white background. "
          "Record the information in this update: ")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.train.official_base_runtime import load_base_runtime
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256, stable_seed
    command = json.loads((a.parent_run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    bank, _ = load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), "--parent-run", str(a.parent_run),
                                "--output", str(a.output), "--device", a.device, "--worker"], env=env)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Probe checkout must be immutable")
    if len(bank["groups"]) != 1 or bank["groups"][0]["event_text"] != RAW_EVENT:
        raise ValueError("This fixed intervention requires the audited ambient event")
    args.dreamlite_device = args.reader_device = a.device
    import torch
    from PIL import Image
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.mem_get_info(torch.device(a.device))[0] < 70*1024**3:
        raise RuntimeError("Inference control needs at least 70GiB free; do not evict any process")
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = load_base_runtime(args, bank, inference_only=True)
    pipe = runtime["pipe"]
    if any(p.requires_grad for module in (pipe.unet,pipe.vae,pipe.text_encoder,runtime["reader"]) for p in module.parameters()):
        raise RuntimeError("Zero-update control must freeze every parameter")
    frozen = train.frozen_versions(pipe,runtime["reader"])
    group = bank["groups"][0]
    context = runtime["contexts"][group["question_id"]]
    seeds = [stable_seed(args.seed,"official-prompt-control-noise",i) for i in range(4)]
    forbidden = {stable_seed(args.seed,"training-noise",i) for i in range(14000)}
    forbidden.update(stable_seed(args.seed,"heldout-evaluation-noise",i) for i in range(8))
    if len(set(seeds))!=4 or forbidden.intersection(seeds):
        raise RuntimeError("Diagnostic noise overlaps training/previous benchmark")
    identity = {"probe_commit":commit, "bank_sha256":file_sha256(args.bank_manifest),
        "snapshots":runtime["snapshots"],"protocol_binding":runtime["protocol_binding"],
        "optimizer_updates":0,"checkpoint_loaded":False,"native_steps":28,"guidance_scale":7.5,
        "events":[RAW_EVENT,RAW_EVENT.replace("ambient.","jazz.")], "prompt_forms":["raw_event","memory_note"],
        "memory_note_prefix":PREFIX,"noise_seeds":seeds,"device":a.device,
        "scope":"Pretrained native pipeline outputs only; synthetic value intervention on same entity/slot; no external answer renderer",
        "allocation":"Read-only probe on the mostly idle Reader GPU during separate fixed-source Base3500 optimization; not a timing benchmark"}
    train.write_json(a.output/"identity.json",identity)
    cells = {}
    with torch.no_grad():
        for event_index,event in enumerate(identity["events"]):
            gold = ["ambient","jazz"][event_index]
            for form in identity["prompt_forms"]:
                prompt = event if form=="raw_event" else PREFIX+event
                sampler = NativeBaseEditSampler(pipe,source_image=Image.new("RGB",(1024,1024),(128,128,128)),event_text=prompt)
                for index,seed in enumerate(seeds):
                    noise=torch.randn(context["source"].shape,generator=torch.Generator().manual_seed(seed)).to(runtime["vae_device"])
                    out=sampler(source_latents=context["source"],noise_latents=noise,num_steps=28)
                    pixels=train.decode_model_latents_unit_interval(pipe.vae,out.latents,clamp=True)
                    label=f"event{event_index}-{form}-seed{index}"
                    train.atomic_tensor(a.output/(label+".pt"),{"latent":out.latents.cpu(),"image":pixels.cpu(),"noise_seed":seed})
                    rgb=(pixels[0].cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype("uint8")
                    Image.fromarray(rgb).save(a.output/(label+".png"))
                    for prompt_id,query in group["question_variants"].items():
                        generation=train.generate_short_answer(model=runtime["reader"],processor=runtime["processor"],
                            image=pixels.to(runtime["reader_device"]),query=query,device=runtime["reader_device"],max_new_tokens=32,do_sample=False)
                        ce=train.qwen3vl_answer_eos_ce(model=runtime["reader"],processor=runtime["processor"],
                            image=pixels[0].to(runtime["reader_device"]),device=runtime["reader_device"],query=query,target=gold,
                            termination=runtime["termination"],lambda_eos=1.,require_image_grad=False,deterministic_ce=True,
                            reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
                        score=train.generation_diagnostics(generation,gold,ce.target_ids[0,:ce.answer_token_count].cpu().tolist())
                        row={"event_index":event_index,"event_text":event,"writer_prompt":prompt,"form":form,
                            "noise_seed":seed,"prompt_id":prompt_id,"query":query,"gold":gold,
                            "image_sha256":canonical_tensor_sha256(pixels.cpu()),"image_artifact":label+".pt",
                            "answer_ce":float(ce.answer_loss),"eos_ce":float(ce.eos_loss),**generation,"scorer":score}
                        train.append_jsonl(a.output/"generations.jsonl",row)
                        key=f"event{event_index}/{form}/{prompt_id}"
                        cell=cells.setdefault(key,{"n":0,"exact_match":0,"answer_eos":0})
                        cell["n"]+=1
                        cell["exact_match"]+=int(score["strict_correct"])
                        cell["answer_eos"]+=int(score["strict_correct"] and score["answer_followed_immediately_by_eos"])
                    print(label,flush=True)
    train.frozen_audit(pipe,runtime["reader"],frozen)
    runtime["verify_additional_bindings"]()
    train.write_json(a.output/"complete.json",{"cells":cells,"identity":identity,
        "artifact_hashes":{p.name:file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    print(json.dumps(cells),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
