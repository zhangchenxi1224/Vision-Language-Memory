"""Actual RGB rollouts and unchanged official-format Reader evaluations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from vision_memory.prefeval.official_ab import records, seed_for, mcq, EVAL_FORMS, sha, writer_event
from scripts.experiments.prefeval_official_ab import write, append
from scripts.train.train_prefeval_official_fm import load_pipe, M

def official_choice(response):
    # PrefEval 50795054 utils_mcq.extract_choice, without stricter/fallback scoring.
    from bs4 import BeautifulSoup
    choice_tag=BeautifulSoup(response,'html.parser').find('choice')
    try:
        if choice_tag:
            match=re.search(r'[ABCD]',choice_tag.string)
            if match: return match.group(0)
    except Exception: return None
    return None

def text_generate(reader,processor,messages,device,max_tokens=300):
    import torch
    tok=processor.tokenizer
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=tok(prompt,return_tensors='pt').to(device)
    with torch.no_grad(): result=reader.generate(**batch,do_sample=False,max_new_tokens=max_tokens)
    ids=result[0,batch.input_ids.shape[1]:].tolist()
    return dict(raw=tok.decode(ids,skip_special_tokens=True),generated_token_ids=ids,
                prompt_token_count=batch.input_ids.shape[1],generated_token_count=len(ids))

def benchmark_history(a,r,processor,reader):
    cache=a.output/'benchmark-history'/r['id'].replace(':','-')
    path=cache.with_suffix('.json')
    if path.exists(): return json.loads(path.read_text())['history']
    disclosure=r['benchmark']['input']['disclosure']
    # Official explicit benchmark: the tested model supplies the acknowledgment.
    ack=text_generate(reader,processor,disclosure,a.device)
    pool=json.loads((a.report/'data/context-pools.json').read_text())['benchmark']
    history=disclosure+[dict(role='assistant',content=ack['raw'])]+pool[:20]
    write(path,dict(id=r['id'],history=history,acknowledgment=ack,source='frozen tested Reader, no SFT teacher acknowledgment'))
    return history

def rollout(a):
    import numpy as np
    import torch
    from PIL import Image
    from scripts.eval.prefeval_rgb import load_reader
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    pipe,_=load_pipe(a.device)
    ckpath=a.output/'writers'/a.arm/a.stage/'checkpoint-final.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    pipe.unet.load_state_dict(ck['unet']);pipe.unet.eval().requires_grad_(False)
    checkpoint_sha=sha(ckpath);del ck
    processor=reader=None
    if a.split!='training':processor,reader=load_reader(M/'Qwen3-VL-4B-Instruct',a.device)
    selected=records(a.report)
    if a.split=='training': selected=[r for r in selected if r['split']=='train']
    selected=selected[a.shard::a.shards]
    for r in selected:
        history=r['history'] if a.split=='training' else benchmark_history(a,r,processor,reader)
        rr=dict(r,history=history)
        for repetition in range(1 if a.split=='training' else 2):
            if a.split=='training': folder=a.output/'rollouts'/a.arm/'training'/r['id'].replace(':','-')
            else: folder=a.output/'rollouts'/a.arm/a.stage/r['id'].replace(':','-')/f'seed-{repetition}'
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                if json.loads((folder/'complete.json').read_text())['checkpoint_sha256']!=checkpoint_sha:raise ValueError('Rollout parent changed')
                continue
            image=Image.new('RGB',(1024,1024),(128,128,128));images=[]
            positions=11 if a.split=='training' or a.stage=='retain' else 1
            for position in range(positions):
                path=folder/f'memory-{position:02d}.png'
                seed=seed_for('rgb-rollout',r['id'],repetition,position)
                # Each step loads the persisted RGB from the prior write.
                with torch.no_grad():
                    source=pipe.prepare_image_latents(pipe.image_processor.preprocess(image),dtype=torch.float32,device=a.device)
                    noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).to(a.device)
                    sampler=NativeBaseEditSampler(pipe,source_image=image,event_text=writer_event(rr,position),guidance_scale=1.)
                    value=sampler(source_latents=source,noise_latents=noise,num_steps=28,return_trajectory=False)
                    pixels=decode_model_latents_unit_interval(pipe.vae,value.latents,clamp=True)
                    array=(pixels[0]*255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy()
                Image.fromarray(array).save(path)
                image=Image.open(path).convert('RGB')
                images.append(dict(position=position,png_sha256=sha(path),noise_seed=seed))
            write(folder/'complete.json',dict(id=r['id'],checkpoint_sha256=checkpoint_sha,images=images))
            print(json.dumps(dict(rollout=r['id'],arm=a.arm,stage=a.stage,seed=repetition)),flush=True)

def evaluate(a):
    import torch
    from scripts.eval.prefeval_rgb import load_reader, read_png
    from vision_memory.reader.open_answer import generate_short_answer
    processor,reader=load_reader(M/'Qwen3-VL-4B-Instruct',a.device)
    questions=json.loads(a.questions.read_text())['records']
    selected=records(a.report)
    if a.phase=='teachers':selected=[r for r in selected if r['split']=='train']
    selected=selected[a.shard::a.shards]
    report=a.output/'evaluations'/a.phase/(a.arm if a.phase!='references' else 'common')/a.stage
    report.mkdir(parents=True,exist_ok=True)
    blank=torch.full((3,1024,1024),128/255)
    for r in selected:
        conditions=[]
        if a.phase=='teachers':
            path=a.output/'teachers'/a.arm/r['id'].replace(':','-')/'memory.png'
            conditions=[('teacher',0,0,path)]
        elif a.phase=='references':
            positions=(0,) if a.stage=='write' else (0,5,10)
            conditions=[(kind,pos,0,None) for kind in ('blank','text') for pos in positions]
        else:
            root=a.output/'rollouts'/a.arm/a.stage/r['id'].replace(':','-')
            positions=(0,) if a.stage=='write' else (0,5,10)
            conditions=[('matched',pos,seed,root/f'seed-{seed}'/f'memory-{pos:02d}.png') for seed in range(2) for pos in positions]
            all_rows=records(a.report)
            ix=next(i for i,v in enumerate(all_rows) if v['id']==r['id'])
            donor=all_rows[(ix+1)%len(all_rows)]
            droot=a.output/'rollouts'/a.arm/a.stage/donor['id'].replace(':','-')
            conditions += [('mismatched',pos,0,droot/'seed-0'/f'memory-{pos:02d}.png') for pos in positions]
        for condition,pos,seed,path in conditions:
            result=report/(r['id'].replace(':','-')+f'-{condition}-{pos}-{seed}.json')
            if result.exists():continue
            image=read_png(path) if path else blank
            history=benchmark_history(a,r,processor,reader) if condition=='text' else None
            rows=[]
            for family in EVAL_FORMS:
                question=questions[r['id']]['forms'][family]
                for task in ('generation','mcq'):
                    order=None;gold=None
                    if task=='mcq':
                        query,gold,order=mcq(r,question,3*(seed_for('eval-options',r['id'],seed)%4),namespace='evaluation')
                        max_tokens=32
                    else:
                        query=question+' (Please respond within 300 words.)';max_tokens=300
                    if condition=='text':
                        g=text_generate(reader,processor,history[:2+2*pos]+[dict(role='user',content=query)],a.device,max_tokens)
                    else:
                        g=generate_short_answer(model=reader,processor=processor,image=image,query=query,
                                               device=a.device,max_new_tokens=max_tokens)
                    predicted=official_choice(g['raw']) if task=='mcq' else None
                    rows.append(dict(family=family,task=task,query=query,generation=g,order=order,
                        gold=gold,predicted=predicted,correct=(predicted==gold[8]) if gold else None,
                        scoring='official_mcq_extract_choice' if gold else 'pending_official_judge'))
            write(result,dict(id=r['id'],split=r['split'],condition=condition,position=pos,seed=seed,
                image_sha256=sha(path) if path else None,rows=rows,
                judge_only_preference=r['benchmark']['evaluation_only']['preference']))
            print(json.dumps(dict(evaluated=r['id'],condition=condition,position=pos,seed=seed)),flush=True)
    write(report/f'complete-{a.shard}.json',dict(count=len(selected),shards=a.shards))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['rollout','teachers','students','references'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--report',type=Path,default=ROOT/'reports/prefeval-official-alignment-20260923')
    p.add_argument('--arm',choices=['A','B'],default='A');p.add_argument('--stage',choices=['write','retain'],default='write')
    p.add_argument('--split',choices=['training','benchmark'],default='benchmark')
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    import socket
    if not socket.gethostname().startswith('dl-clear-retain-h200x4-20260914'):raise RuntimeError('Wrong notebook')
    try:
        if a.phase=='rollout':rollout(a)
        else:evaluate(a)
    except BaseException:
        import traceback
        write(a.output/f'failure-{a.phase}-{a.arm}-{a.shard}-{int(time.time())}.json',dict(error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
