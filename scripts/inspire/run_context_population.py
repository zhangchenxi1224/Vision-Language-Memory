"""First bounded phase of fixed-topic history coverage: targets then FM128."""
import argparse
from collections import Counter
import concurrent.futures
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_route_functional import read,read_lines,save_once,sha,digest
from scripts.experiments.prefeval_k1_data import load_records,official_mcq
from scripts.experiments.prefeval_multitarget_bank import select_rows
from scripts.experiments.prefeval_prompt_matching import validate_teacher_manifest,target_cache_binding,_validate_cached_target
from scripts.experiments.prefeval_context_coverage import training_context
from scripts.inspire.run_context_fit import RUN,BANKS,PARENT_SHA,OLD
from scripts.inspire.run_writer_readout import execute,accrued_seconds,other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json

ALL_IDS=ROOT/'configs/experiments/context_population32_ids.json'
ADDED_IDS=ROOT/'configs/experiments/context_population_added16_ids.json'
OLD_IDS=ROOT/'configs/experiments/context_coverage_ids.json'
FIT_SHA='b2527682b9560ea29c4c9bbaa5f67b361a3fd8057de3026d15c81bf27ad6ecea'


def population():
    rows=select_rows(load_records('pilot'),ALL_IDS)
    added=select_rows(load_records('pilot'),ADDED_IDS)
    original=read(OLD_IDS)['ids'];ids=[r['base_pair_id'] for r in rows];new=[r['base_pair_id'] for r in added]
    topics=Counter(r['topic'] for r in rows)
    if (len(ids)!=32 or len(new)!=16 or len(topics)!=8 or set(topics.values())!={4}
            or set(ids)!=set(original)|set(new) or set(original)&set(new)
            or set(topics)!=set(p.split(':')[0] for p in original)
            or set(ids)&{r['base_pair_id'] for r in load_records('dev')}):
        raise ValueError('Frozen same-topic population changed')
    return rows,added


def remaining_seconds(output):
    previous=other_seconds(output.parent)+sum(accrued_seconds(output.parent/p) for p in
        ('writer-readout-v1','route-functional-v1','context-fit-v1','context-dev-v1'))
    current=accrued_seconds(output)
    return min(1.5*3600-current,3*3600-current,16*3600-previous-current),previous,current


def check_parent():
    parent=OLD/'train/checkpoint-final.pt';fit=RUN/'context-fit-v1/context-diverse/train/checkpoint-final.pt'
    if sha(parent)!=PARENT_SHA or sha(fit)!=FIT_SHA:
        raise ValueError('Parent or paired16 endpoint changed')
    return {str(parent):PARENT_SHA,str(fit):FIT_SHA}


def verify_bank(output,prefeval,*,original_only=False):
    import torch
    rows,added=population();new={r['base_pair_id'] for r in added};files={};bindings={};mcq=official_mcq(prefeval)
    checked=[];new_updates=0;common=None
    for row in rows:
        pid=row['base_pair_id'];is_new=pid in new
        if original_only and is_new:continue
        base=(output/'new-teachers') if is_new else BANKS['context-diverse']
        folder=base/pid.replace(':','_');done=read(folder/'complete.json');b=done['binding']
        bindings[pid]=validate_teacher_manifest(done,arm='B',supervision='prompt_matching',steps=288)
        if done['pair_id']!=pid or b['context_suite']!='diverse-v1' or b['split']!='pilot' or b['lr']!=.05:
            raise ValueError('Teacher scientific protocol mismatch')
        shared={k:b[k] for k in ('arm','steps','lr','forms_sha256','mcq_source_sha256','split',
            'supervision','temperature','quantization','teacher_max_new_tokens','reader_path',
            'reader_weights_sha256','reader_config_sha256','target_pipeline_sha256','reader_objective_sha256',
            'teacher_history_scope','context_suite','snapshot_steps')}
        if common is None:common=shared
        if common!=shared:raise ValueError('New/old teacher settings are not matched')
        for name,key in [('memory.png','png_sha256'),('latent.pt','latent_sha256'),('teacher-targets.json','teacher_targets_sha256')]:
            p=folder/name
            if sha(p)!=done[key]:raise ValueError('Target hash changed')
            files[str(p)]=done[key]
        for p in (folder/'complete.json',folder/'optimization.jsonl'):files[str(p)]=sha(p)
        z=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
        if z.shape!=(1,4,128,128) or z.dtype!=torch.float32 or not torch.isfinite(z).all():raise ValueError('Bad target latent')
        from PIL import Image
        with Image.open(folder/'memory.png') as image:
            if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Bad target PNG')
            image.load()
        log=read_lines(folder/'optimization.jsonl')
        if [x['step'] for x in log]!=list(range(1,289)) or any(not math.isfinite(x['grad_norm']) or x['grad_norm']<=0 for x in log):
            raise ValueError('Incomplete teacher gradient log')
        expected=[list(training_context(row,s,mcq)) for s in range(288)]
        if b['contexts']['schedule'][pid]!=expected:raise ValueError('Teacher contexts differ')
        if is_new:
            new_updates+=len(log)
            eos=read(base/f"termination-{b['shard']}.json")['assistant_end_token_id']
            cfg=read(Path(b['reader_path'])/'config.json');pad=cfg.get('pad_token_id',cfg.get('text_config',{}).get('pad_token_id'))
            for name,receipt in read(folder/'teacher-targets.json').items():
                p=output/'targets'/name
                if p.name!=name or sha(p)!=receipt['cache_sha256']:raise ValueError('Teacher cache receipt changed')
                target=torch.load(p,map_location='cpu',weights_only=True)
                _validate_cached_target(target,target_cache_binding(b,row,receipt['binding']['query']),assistant_end_token_id=eos,pad_token_id=pad)
                if target['target_ids']!=receipt['target_ids'] or target['logits_sha256']!=receipt['logits_sha256']:raise ValueError('Teacher tensor receipt changed')
                files[str(p)]=receipt['cache_sha256']
        checked.append(dict(pair_id=pid,path=str(folder),latent_sha256=done['latent_sha256'],binding_sha256=bindings[pid]))
    if not original_only:
        for p in checked:
            link=output/'bank'/p['pair_id'].replace(':','_');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(p['path'],target_is_directory=True)
            if link.resolve()!=Path(p['path']).resolve():raise ValueError('Foreign bank link')
    return dict(files=files,targets=checked,new_teacher_updates_verified=new_updates)


def build_jobs(args,phase):
    if phase=='teachers':
        return [dict(name=f'teacher-{i}',gpu=i,command=[sys.executable,'scripts/experiments/prefeval_k1_teacher.py',
            '--arm','B','--supervision','prompt_matching','--context-suite','diverse-v1','--split','pilot',
            '--ids-file',str(ADDED_IDS),'--shards','2','--shard',str(i),'--base',str(args.base),
            '--reader',str(args.reader),'--prefeval',str(args.prefeval),'--output',str(args.output/'new-teachers'),
            '--teacher-cache',str(args.output/'targets'),'--steps','288','--snapshot-steps','72,144,216,288']) for i in range(2)]
    if phase!='writer':raise ValueError('Unknown phase')
    return [dict(name='writer32',gpu=0,command=[sys.executable,'scripts/experiments/prefeval_k1_writer.py','train',
        '--arm','B','--split','pilot','--ids-file',str(ALL_IDS),'--base',str(args.base),
        '--official-source',str(args.official_source),'--checkpoint',str(OLD/'train/checkpoint-final.pt'),
        '--output',str(args.output/'writer32'),'--teachers',str(args.output/'bank'),
        '--teacher-supervision','prompt_matching','--teacher-steps','288','--steps','128'])]


def verify_writer(output,bank):
    import torch
    from vision_memory.training.latent_bank_unet import stable_seed
    folder=output/'writer32';m=read(folder/'manifest.json');done=read(folder/'complete.json')
    expected=dict(arm='B',stage='write',steps=128,effective_batch=4,split='pilot',parent_sha256=PARENT_SHA,
        seed=20260924,teacher_supervision='prompt_matching',teacher_steps=288)
    if any(m.get(k)!=v for k,v in expected.items()) or done!=dict(steps=128,checkpoint_sha256=sha(folder/'checkpoint-final.pt')):
        raise ValueError('Writer endpoint changed')
    if m['targets']!={x['pair_id']:x['latent_sha256'] for x in bank['targets']} or m['teacher_binding_hashes']!={x['pair_id']:x['binding_sha256'] for x in bank['targets']}:
        raise ValueError('Wrong training targets')
    log=read_lines(folder/'optimization.jsonl');rows,_=population();counts=Counter()
    if [r['step'] for r in log]!=list(range(1,129)):raise ValueError('Incomplete Writer updates')
    for r in log:
        if not math.isfinite(r['grad_norm']) or r['grad_norm']<=0 or len(r['draws'])!=4:raise ValueError('Bad Writer gradient')
        for micro,d in enumerate(r['draws']):
            draw=(r['step']-1)*4+micro;cycle,offset=divmod(draw,32)
            order=torch.randperm(32,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
            sigma=float(torch.rand((),generator=torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))))
            if d['pair_id']!=rows[order[offset]]['base_pair_id'] or d['sigma']!=sigma or d['position']!=0 or not math.isfinite(d['mse']):raise ValueError('Writer random draw changed')
            counts[d['pair_id']]+=1
    if set(counts.values())!={16} or len(counts)!=32:raise ValueError('Unbalanced training population')
    checkpoint=torch.load(folder/'checkpoint-final.pt',map_location='cpu',weights_only=True)
    if checkpoint['optimizer_step']!=128 or checkpoint['manifest']!=m or any(not torch.isfinite(p).all() for p in checkpoint['trainable_state'].values()):
        raise ValueError('Invalid final checkpoint tensors')
    return dict(steps=128,draws=512,preferences=32,draws_per_preference=16,gradients_finite_nonzero=True,checkpoint_sha256=done['checkpoint_sha256'])


def main(args):
    if args.output.resolve()!=RUN/'context-population32-v1':raise ValueError('Require isolated population output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        rows,added=population()
        save_once(args.output/'plan.json',dict(commit=commit,plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_POPULATION_PLAN.md'),
            ids_sha256=sha(ALL_IDS),added_ids_sha256=sha(ADDED_IDS),base=str(args.base),reader=str(args.reader),prefeval=str(args.prefeval),
            official_source=str(args.official_source),phase_cap_gpu_hours=1.5,iteration_cap_gpu_hours=3,campaign_cap=16,
            population=[r['base_pair_id'] for r in rows],added=[r['base_pair_id'] for r in added],parent=check_parent()))
        save_once(args.output/'original-bank.json',verify_bank(args.output,args.prefeval,original_only=True))
        left,previous,_=remaining_seconds(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [p for p in RUN.glob('*/active-owner/owner.json') if p!=claim/'owner.json']:raise ValueError('Other owner')
        for log in list((args.output/'new-teachers').glob('*/optimization.jsonl'))+list((args.output/'writer32').glob('optimization.jsonl')):
            if not (log.parent/'complete.json').exists():raise ValueError('Partial native output requires checkpoint/tail recovery audit')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(jobs):
            if time.monotonic()+30>=deadline:raise ValueError('No time for next stage')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:f.result()
        write_json(args.output/'status.json',dict(status='teachers',time=time.time(),commit=commit))
        group(build_jobs(args,'teachers'))
        bank=verify_bank(args.output,args.prefeval);save_once(args.output/'bank.json',bank)
        write_json(args.output/'status.json',dict(status='writer_training',time=time.time(),commit=commit))
        group(build_jobs(args,'writer'))
        save_once(args.output/'writer-audit.json',verify_writer(args.output,bank))
        write_json(args.output/'status.json',dict(status='ready_for_frozen_readout',time=time.time(),scientific_results='pending',
            gpu_hours=accrued_seconds(args.output)/3600,campaign_gpu_hours=(previous+accrued_seconds(args.output))/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','base','reader','prefeval','official-source'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
