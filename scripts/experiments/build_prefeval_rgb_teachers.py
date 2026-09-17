"""Train-side full-state latent teachers; fixed endpoints and explicit failures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from diffusers import AutoencoderKL
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, encode_model_latent
from vision_memory.prefeval.rgb_protocol import digest, scope_name, queries, ABSENT
from vision_memory.reader.open_answer import generate_short_answer, score_short_answer
from vision_memory.reader.open_eos import assistant_termination_contract, qwen3vl_answer_eos_ce
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


def qualification(target, manifest):
    result = list(target["qualification_queries"])
    for scope, value in sorted(target["state"].items()):
        result.append({"scope":scope, "kind":"active_status", "query":
            f"Is any preference currently active for {scope_name(scope)}? Answer yes or no only.",
            "target":"yes" if value is not None else "no"})
        if value is not None:
            # Read-only content verification; deliberately not part of teacher loss.
            alternatives = sorted({r["preference"] for r in manifest["records"].values()
                if r["split"] == "train" and r["topic"] == scope and r["preference"] != value})
            for proposed in (value, alternatives[0]):
                result.append({"scope":scope, "kind":"complete_constraint_check", "query":
                    f"Is this the complete current stored preference statement for {scope_name(scope)}? "
                    f"Statement: {proposed}\nAnswer yes or no only.",
                    "target":"yes" if proposed == value else "no"})
    return result


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = manifest["sentinel_targets"] if args.sentinel else sorted(manifest["targets"])
    assignment = schedule(manifest["targets"], selected, args.shards)[args.shard]
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha": digest(manifest), "steps":256, "lr":.05,
        "optimizer":"Adam", "betas":[.9,.999], "eps":1e-8, "weight_decay":0.,
        "initialization":"gray posterior mean for roots; registered predecessor endpoint otherwise",
        "qualification":"reopened uint8 RGB PNG; two held-out recovery forms, status and content checks",
        "max_new_tokens":manifest["max_recovery_tokens"], "shards":args.shards,
        "assignment":assignment, "shard":args.shard}
    idpath = args.output / f"identity-{args.shard}.json"
    if idpath.exists() and json.loads(idpath.read_text()) != identity:
        raise ValueError("Teacher identity changed")
    idpath.write_text(json.dumps(identity,indent=2))
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", local_files_only=True,
                                      torch_dtype=torch.float32).to(args.device)
    vae.eval().requires_grad_(False)
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    versions = [(p, int(p._version)) for module in (vae, reader) for p in module.parameters()]
    gray = torch.full((1,3,1024,1024),128/255,device=args.device,dtype=torch.float32)
    with torch.no_grad():
        initial = encode_model_latent(vae, gray).detach()
    for number, sid in enumerate(assignment):
        out = args.output / sid
        out.mkdir(exist_ok=True)
        result_path = out / "result.json"
        if result_path.exists():
            continue
        target = manifest["targets"][sid]
        if target["split"] != "train":
            raise ValueError("Evaluation state must never receive an optimized teacher")
        predecessor = target["predecessor"]
        init = (torch.load(args.output/predecessor/"latent.pt",map_location=args.device,weights_only=True)
                if predecessor else initial)
        oracle = VAELatentOracle(vae=vae, initial_latent=init, compute_dtype=torch.float32)
        optimizer = torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8)
        started = time.monotonic()
        trace = out / "optimization.jsonl"
        if trace.exists():
            raise ValueError("Interrupted teacher has no certified endpoint; inspect before restarting")
        training = queries(target["state"], training=True)
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
        with torch.no_grad():
            image = oracle.image().detach().cpu()[0]
        array = image.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
        Image.fromarray(array).save(out/"memory.png")
        png = read_png(out/"memory.png")
        rows=[]
        for q in qualification(target,manifest):
            generated=generate_short_answer(model=reader,processor=processor,image=png,
                query=q["query"],device=args.device,max_new_tokens=manifest["max_recovery_tokens"],
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            score=score_short_answer(generated["raw"],q["target"])
            score["strict_correct"]=bool(score["strict_correct"] and generated["eos_reached"])
            rows.append({"query":q,"generation":generated,"score":score})
        import hashlib
        result={"id":sid,"state":target["state"],"capacity":len(target["state"]),
            "active_k":sum(v is not None for v in target["state"].values()),
            "qualified":all(r["score"]["strict_correct"] for r in rows),
            "correct":sum(r["score"]["strict_correct"] for r in rows),"total":len(rows),
            "rows":rows,"seconds":time.monotonic()-started,
            "png_sha256":hashlib.sha256((out/"memory.png").read_bytes()).hexdigest(),
            "latent_sha256":hashlib.sha256((out/"latent.pt").read_bytes()).hexdigest(),
            "predecessor":predecessor}
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
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--reader",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--sentinel",action="store_true")
    main(parser.parse_args())
