"""Summarize real generations, optimizer draws and protocol evidence from a run."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def collect(root: Path):
    result_path=root/"train/result.json"
    result=json.loads(result_path.read_text()) if result_path.exists() else None
    terminal_path=root/"terminal.json"
    terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    metrics_path=root/"train/training.jsonl"
    rows=[json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    draws=[m for row in rows for m in row["microbatches"]]
    phases={}
    for phase in ("baseline","trained"):
        p=root/"train"/phase/"generations.jsonl"
        records=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
        matched=[r for r in records if r["condition"]=="matched"]
        cells={}
        for r in records:
            key=r["condition"]+"/"+r["prompt_id"]
            v=cells.setdefault(key,{"n":0,"exact_match":0,"answer_eos":0})
            v["n"]+=1
            v["exact_match"]+=int(r["scorer"]["strict_correct"])
            v["answer_eos"]+=int(r["scorer"]["strict_correct"] and r["scorer"]["answer_followed_immediately_by_eos"])
        phases[phase]={"cells":cells,"matched_raw":dict(Counter(r["raw"] for r in matched)),
                       "generation_rows":len(records)}
    verification=None
    if terminal and terminal.get("state")=="completed" and result is not None:
        verification=hashlib.sha256(result_path.read_bytes()).hexdigest()==terminal["training_result_sha256"]
        if not verification: raise ValueError("Run result SHA mismatch")
        cp=root/"train/checkpoint-final.pt"
        checkpoint_hash=hashlib.sha256()
        with cp.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):
                checkpoint_hash.update(block)
        digest=checkpoint_hash.hexdigest()
        if digest!=result["checkpoint_sha256"]: raise ValueError("Checkpoint SHA mismatch")
    return {"run":str(root),"terminal":terminal,"result_and_checkpoint_verified":verification,
        "optimizer_steps":len(rows),"training_draws":len(draws),
        "sigma_min":min((m["effective_sigma"] for m in draws),default=None),
        "sigma_max":max((m["effective_sigma"] for m in draws),default=None),
        "sigma_above_half":sum(m["effective_sigma"]>.5 for m in draws),
        "training_noise_count":len({m["noise_seed"] for m in draws}),
        "loss_first64_mean":sum(r["flow_matching_mse"] for r in rows[:64])/min(len(rows),64) if rows else None,
        "loss_last64_mean":sum(r["flow_matching_mse"] for r in rows[-64:])/min(len(rows),64) if rows else None,
        "loss_interpretation":"unpaired sampled training loss, not functional accuracy or a paired loss comparison",
        "phases":phases,"paired_evaluation":result.get("paired_evaluation") if result else None,
        "adapter_delta_l2":result.get("unet_adapter_delta_l2") if result else None,
        "trainable_scope":result.get("trainable_scope","lora") if result else None,
        "unet_parameter_delta_l2":result.get("unet_parameter_delta_l2",result.get("unet_adapter_delta_l2")) if result else None}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--archive",type=Path)
    a=p.parse_args()
    summary=collect(a.run)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    if a.archive:
        with tarfile.open(a.archive,"w:gz") as archive:
            for path in sorted(a.run.rglob("*")):
                if path.is_file() and path.suffix in {".json",".jsonl",".log"}:
                    archive.add(path,arcname=path.relative_to(a.run))
            archive.add(a.output,arcname="verified-summary.json")
    print(json.dumps(summary,ensure_ascii=False))
