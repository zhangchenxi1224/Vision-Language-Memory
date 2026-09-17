"""Train-side full-state latent teachers; fixed endpoints and explicit failures."""
from __future__ import annotations
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from scripts.eval.prefeval_rgb import load_reader, read_png, append, load_overlay
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, encode_model_latent
from vision_memory.prefeval.rgb_protocol import digest, scope_name, queries, ABSENT
from vision_memory.reader.open_answer import generate_short_answer, score_short_answer
from vision_memory.reader.open_eos import assistant_termination_contract, qwen3vl_answer_eos_ce, generation_diagnostics
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT


def schedule(targets, selected, shards):
    """Keep each initializer dependency component on one GPU; order topologically."""
    def root(sid):
        seen = set()
        while targets[sid]["predecessor"]:
            if sid in seen:
                raise ValueError("Cyclic teacher initializer")
            seen.add(sid)
            sid = targets[sid]["predecessor"]
        return sid
    components = defaultdict(list)
    for sid in selected:
        components[root(sid)].append(sid)
    assignments, costs = [[] for _ in range(shards)], [0]*shards
    for key, members in sorted(components.items(), key=lambda kv: (-sum(len(targets[s]["state"]) for s in kv[1]), kv[0])):
        shard = min(range(shards), key=lambda i: (costs[i], i))
        assignments[shard].extend(members)
        costs[shard] += sum(len(targets[s]["state"]) for s in members)
    result = []
    for assigned in assignments:
        done, ordered = set(), []
        def visit(sid):
            if sid in done:
                return
            pred = targets[sid]["predecessor"]
            if pred:
                if pred not in assigned:
                    raise ValueError("Unscheduled initializer dependency")
                visit(pred)
            done.add(sid)
            ordered.append(sid)
        for sid in sorted(assigned):
            visit(sid)
        result.append(ordered)
    return result


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_vae(vae):
    if (type(vae).__name__ != 'AutoencoderTiny' or vae.dtype != torch.float32
            or vae.config.latent_channels != 4 or vae.config.scaling_factor != 1.
            or vae.config.shift_factor != 0.):
        raise ValueError('Require locked FP32 AutoencoderTiny model coordinates')


def check_completed(out, binding):
    result=json.loads((out/'result.json').read_text(encoding='utf-8'))
    if result['binding']!=binding: raise ValueError('Completed target binding changed')
    for name,sha in result['artifacts'].items():
        if file_sha(out/name)!=sha: raise ValueError('Completed endpoint artifact changed')
    trace=[json.loads(s) for s in (out/'optimization.jsonl').read_text().splitlines()]
    if [r['step'] for r in trace]!=list(range(1,257)):
        raise ValueError('Incomplete optimization trace')
    return result


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    overlay = load_overlay(args.overlay,manifest)
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from scripts.experiments.refine_historical_writer_targets import snapshot_bindings, M
    determinism=configure_strict_cuda_determinism(0)
    if args.base.resolve()!=M/'DreamLite-base-a9a0f15-20260907' or args.reader.resolve()!=M/'Qwen3-VL-4B-Instruct':
        raise ValueError('Require verified model paths')
    snapshots=snapshot_bindings()
    selected = manifest["sentinel_targets"] if args.sentinel else sorted(manifest["targets"])
    assignment = schedule(manifest["targets"], selected, args.shards)[args.shard]
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha": digest(manifest), "steps":256, "lr":.05,
        "optimizer":"Adam", "betas":[.9,.999], "eps":1e-8, "weight_decay":0.,
        "initialization":"deterministic Tiny encoding of gray128 for roots; registered predecessor endpoint otherwise",
        "qualification":"reopened uint8 RGB PNG; two held-out recovery forms, status and content checks",
        "max_new_tokens":manifest["max_recovery_tokens"], "shards":args.shards,
        "assignment":assignment, "shard":args.shard,"overlay_sha":digest(overlay),
        "snapshots":snapshots,"determinism":determinism}
    idpath = args.output / f"identity-{args.shard}.json"
    if idpath.exists() and json.loads(idpath.read_text()) != identity:
        raise ValueError("Teacher identity changed")
    idpath.write_text(json.dumps(identity,indent=2))
    from diffusers import AutoencoderTiny
    vae = AutoencoderTiny.from_pretrained(args.base, subfolder="vae", local_files_only=True,
                                      torch_dtype=torch.float32).to(args.device)
    vae.eval().requires_grad_(False)
    validate_vae(vae)
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    versions = [(p, int(p._version)) for module in (vae, reader) for p in module.parameters()]
    gray = torch.full((1,3,1024,1024),128/255,device=args.device,dtype=torch.float32)
    with torch.no_grad():
        initial = encode_model_latent(vae, gray).detach()
        if initial.shape!=(1,4,128,128) or not torch.isfinite(initial).all():
            raise ValueError('Invalid gray latent')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.official_source,text=True).strip()!='a6e20c8cc94027f37dd7c5a81b0b3b472aa18409':
            raise ValueError('Wrong official source')
        sys.path.insert(0,str(args.official_source))
        from dreamlite import DreamLitePipelineLoRA
        from diffusers.image_processor import VaeImageProcessor
        preprocessed=VaeImageProcessor(vae_scale_factor=8).preprocess(Image.new('RGB',(1024,1024),(128,128,128)))
        official=DreamLitePipelineLoRA.prepare_image_latents(SimpleNamespace(vae=vae),preprocessed,
                    dtype=torch.float32,device=args.device)
        if not torch.equal(initial,official):raise ValueError('Gray initializer differs from official source encoding')
    (args.output/f'gray-encoding-{args.shard}.json').write_text(json.dumps(dict(
        latent_shape=list(initial.shape),latent_sha=canonical_tensor_sha256(initial),official_exact=True,
        vae_class=type(vae).__name__,vae_config=dict(vae.config),vae_weight_sha=file_sha(args.base/'vae/diffusion_pytorch_model.safetensors')),
        indent=2))
    for number, sid in enumerate(assignment):
        out = args.output / sid
        out.mkdir(exist_ok=True)
        result_path = out / "result.json"
        target = manifest["targets"][sid]
        if target["split"] != "train":
            raise ValueError("Evaluation state must never receive an optimized teacher")
        predecessor = target["predecessor"]
        init = (torch.load(args.output/predecessor/"latent.pt",map_location=args.device,weights_only=True)
                if predecessor else initial)
        binding=dict(identity_sha=digest(identity),target=target,queries=overlay['teachers'][sid],
                     initialization_sha=canonical_tensor_sha256(init))
        if result_path.exists():
            check_completed(out,binding)
            continue
        oracle = VAELatentOracle(vae=vae, initial_latent=init, compute_dtype=torch.float32)
        optimizer = torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8)
        started = time.monotonic()
        trace = out / "optimization.jsonl"
        if trace.exists():
            raise ValueError("Interrupted teacher has no certified endpoint; inspect before restarting")
        torch.save(init.detach().cpu(),out/'initial-latent.pt')
        training = overlay['teachers'][sid]['training']
        for step in range(256):
            optimizer.zero_grad(set_to_none=True)
            pixels = oracle.image()
            active = [q for q in training if q["form"] == step % 3]
            changed = [q for q in active if q["scope"] == target["changed_scope"]]
            unchanged = [q for q in active if q["scope"] != target["changed_scope"]]
            categories = [group for group in (changed,unchanged) if group]
            losses = []
            cursor = 0
            for group in categories:
                for q in group:
                    ce = qwen3vl_answer_eos_ce(model=reader,processor=processor,
                        image=pixels[0],query=q["query"],target=q["target"],device=args.device,
                        termination=termination,lambda_eos=1.,require_image_grad=True,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    weighted = ce.loss / (len(categories)*len(group))
                    if not torch.isfinite(weighted):
                        raise RuntimeError("Nonfinite teacher loss")
                    cursor += 1
                    weighted.backward(retain_graph=cursor < len(active))
                    losses.append({"scope":q["scope"],"answer_ce":float(ce.answer_loss.detach()),
                                   "eos_ce":float(ce.eos_loss.detach()),"weight":1/(len(categories)*len(group))})
            grad = oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise RuntimeError("Teacher latent has no finite nonzero gradient")
            optimizer.step()
            append(trace, {"step":step+1,"losses":losses,"seconds":time.monotonic()-started})
            if (step+1)%32==0:
                print(json.dumps({"target":sid,"number":number,"of":len(assignment),"step":step+1,
                                  "loss":sum(r["weight"]*(r["answer_ce"]+r["eos_ce"]) for r in losses)}),flush=True)
        torch.save(oracle.latent_fp32.detach().cpu(),out/"latent.pt")
        torch.save(optimizer.state_dict(),out/'optimizer.pt')
        with torch.no_grad():
            image = oracle.image().detach().cpu()[0]
        array = image.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
        Image.fromarray(array).save(out/"memory.png")
        png = read_png(out/"memory.png")
        rows=[]
        for q in overlay['teachers'][sid]['qualification']:
            generated=generate_short_answer(model=reader,processor=processor,image=png,
                query=q["query"],device=args.device,max_new_tokens=manifest["max_recovery_tokens"],
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            score=score_short_answer(generated["raw"],q["target"])
            score["strict_correct"]=bool(score["strict_correct"] and generated["eos_reached"])
            diagnostic=generation_diagnostics(generated,q['target'],processor.tokenizer.encode(q['target'],add_special_tokens=False))
            rows.append({"query":q,"generation":generated,"score":score,'token_diagnostics':diagnostic})
        result={"id":sid,"state":target["state"],"capacity":len(target["state"]),
            "active_k":sum(v is not None for v in target["state"].values()),
            "qualified":all(r["score"]["strict_correct"] for r in rows),
            "correct":sum(r["score"]["strict_correct"] for r in rows),"total":len(rows),
            "rows":rows,"seconds":time.monotonic()-started,
            "png_sha256":hashlib.sha256((out/"memory.png").read_bytes()).hexdigest(),
            "latent_sha256":hashlib.sha256((out/"latent.pt").read_bytes()).hexdigest(),
            "predecessor":predecessor,"binding":binding,
            "artifacts":{name:file_sha(out/name) for name in
                ('initial-latent.pt','latent.pt','optimizer.pt','optimization.jsonl','memory.png')}}
        for p,version in versions:
            if int(p._version)!=version or p.requires_grad or p.grad is not None:
                raise RuntimeError("Frozen teacher components changed")
        result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(json.dumps({k:result[k] for k in ('id','capacity','qualified','correct','total','seconds')}),flush=True)
        del optimizer,oracle
    (args.output/f"complete-{args.shard}.json").write_text(json.dumps({"targets":assignment}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--overlay",type=Path,required=True)
    parser.add_argument("--official-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--reader",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--sentinel",action="store_true")
    main(parser.parse_args())
