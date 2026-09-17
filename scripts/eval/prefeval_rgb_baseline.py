"""Fixed 4f native writes; only saved/reopened RGB survives between transitions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from PIL import Image
import torch
from scripts.eval.prefeval_rgb import (load_reader,load_overlay,read_png,append,mcq_score,
    verify_reader_snapshot)
from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
from vision_memory.dreamlite.writer_package import load_writer_package
from vision_memory.prefeval.rgb_protocol import digest,writer_input
from vision_memory.reader.open_answer import generate_short_answer,score_short_answer
from vision_memory.reader.open_eos import generation_diagnostics
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256,configure_strict_cuda_determinism

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels_sha(image):return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()

def write_from_png(pipe,previous_png,event,seed,output_png):
    with Image.open(previous_png) as im: source=im.copy()
    boundary=writer_input(source,event)
    memory=OfficialRGBMemory(pipe,image=boundary['image'],guidance_scale=1.,inference_condition='native')
    before=pixels_sha(source)
    written=memory.write(boundary['event'],seed=seed)
    written.image.save(output_png)
    with Image.open(output_png) as im: reopened=im.copy()
    if pixels_sha(reopened)!=pixels_sha(written.image):raise ValueError('PNG roundtrip changed pixels')
    return dict(source_file_sha=sha(previous_png),source_pixels_sha=before,
        output_file_sha=sha(output_png),output_pixels_sha=pixels_sha(reopened),
        source_latent_sha=canonical_tensor_sha256(written.source_latent),
        noise_sha=canonical_tensor_sha256(written.noise),native_steps=len(written.trajectory),
        model_inputs=['previous_png','event','external_noise'],seed=seed)

def main(a):
    m=json.loads(a.manifest.read_text(encoding='utf-8'));o=load_overlay(a.overlay,m)
    deterministic=configure_strict_cuda_determinism(0)
    a.output.mkdir(parents=True,exist_ok=False)
    pipe,package=load_writer_package(a.package,base_model=a.base,official_source=a.official_source,device=a.device)
    if package['parent_checkpoint_sha256']!='7294684170578dfc617b4fafcea97e6480c08642ca4f1e5967ff8966aa103182':
        raise ValueError('Baseline must use exact 4f parameters')
    reader_binding=verify_reader_snapshot(a.reader)
    processor,reader=load_reader(a.reader,a.device)
    identity=dict(overlay_sha=digest(o),manifest_sha=digest(m),package=package,reader=reader_binding,
                  determinism=deterministic,shard=a.shard,shards=a.shards)
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    started=time.monotonic();writes=0
    for index,ep in enumerate(o['baseline']['episodes']):
        if index%a.shards!=a.shard:continue
        directory=a.output/ep['id'].replace(':','-');directory.mkdir()
        previous=directory/'gray.png';Image.new('RGB',(1024,1024),(128,128,128)).save(previous)
        previous_output_pixels=None
        for tr in ep['transitions']:
            target=directory/f"write-{tr['ordinal']:02}.png"
            seed=o['baseline']['seed']+index*100+tr['ordinal']
            audit=write_from_png(pipe,previous,tr['event'],seed,target)
            if previous_output_pixels is not None and audit['source_pixels_sha']!=previous_output_pixels:
                raise ValueError('Successor did not use reopened predecessor')
            before=sha(target);image=read_png(target);rows=[]
            for q in tr['queries']:
                result=generate_short_answer(model=reader,processor=processor,image=image,query=q['query'],
                    device=a.device,max_new_tokens=128,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                score=mcq_score(result['raw'],q['target_index']) if q['kind']=='official_mcq' else score_short_answer(result['raw'],q['target'])
                score['strict_correct']=bool(score['strict_correct'] and result['eos_reached'])
                diagnostic=None if q['kind']=='official_mcq' else generation_diagnostics(result,q['target'],processor.tokenizer.encode(q['target'],add_special_tokens=False))
                rows.append(dict(query=q,generation=result,score=score,token_diagnostics=diagnostic))
            if sha(target)!=before:raise ValueError('Reads mutated persistent image')
            append(a.output/'writes.jsonl',dict(episode=ep['id'],split=ep['split'],transition=tr,
                audit=audit,source=str(previous.relative_to(a.output)),output=str(target.relative_to(a.output)),
                rows=rows,seconds=time.monotonic()-started))
            writes+=1;print(json.dumps(dict(writes=writes,episode=ep['id'],ordinal=tr['ordinal'],
                 correct=sum(r['score']['strict_correct'] for r in rows),reads=len(rows))),flush=True)
            previous_output_pixels=audit['output_pixels_sha'];previous=target
    (a.output/'complete.json').write_text(json.dumps(dict(writes=writes,seconds=time.monotonic()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('manifest','overlay','output','package','base','reader','official-source'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
