"""Bounded 16-history shared-Writer context-transfer fit gate; no dev scoring."""
import argparse
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
from scripts.experiments.prefeval_route_functional import (
    read,read_lines,save_once,sha,digest,cohort_rows,load_reference,new_keys,index_rows,
    validate_new_row,donor_map,summarize,event_text,
)
from scripts.experiments.prefeval_writer_readout import OLD,PARENT_SHA
from scripts.inspire.run_writer_readout import execute,accrued_seconds,other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json

ENDPOINTS=('context-narrow','context-diverse')
RUN=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/context-coverage-20261006')
BANKS={ENDPOINTS[0]:RUN.parent/'prompt-matching-20261005/pilot-B/prompt_matching/teachers',
       ENDPOINTS[1]:RUN/'pilot/prompt_matching/teachers'}


def verify_banks():
    import torch
    from scripts.experiments.prefeval_prompt_matching import validate_teacher_manifest
    files,bindings,shape={}, {}, None
    checkpoint=OLD/'train/checkpoint-final.pt'
    if sha(checkpoint)!=PARENT_SHA or read(OLD/'train/complete.json')['checkpoint_sha256']!=PARENT_SHA:
        raise ValueError('Parent changed')
    files[str(checkpoint)]=PARENT_SHA
    for row in cohort_rows()['pilot']:
        pair=[]
        for endpoint in ENDPOINTS:
            folder=BANKS[endpoint]/row['base_pair_id'].replace(':','_')
            done=read(folder/'complete.json')
            validate_teacher_manifest(done,arm='B',supervision='prompt_matching',steps=288)
            if done['pair_id']!=row['base_pair_id'] or row['base_pair_id'] not in done['binding']['assignment']:
                raise ValueError('Wrong teacher history')
            for name,key in [('latent.pt','latent_sha256'),('memory.png','png_sha256')]:
                if sha(folder/name)!=done[key]:
                    raise ValueError('Teacher asset changed')
                files[str(folder/name)]=done[key]
            files[str(folder/'complete.json')]=sha(folder/'complete.json')
            target=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            current=(list(target.shape),str(target.dtype))
            if shape is None:shape=current
            if current!=shape or not torch.isfinite(target).all():
                raise ValueError('Incompatible latent bank')
            bindings[endpoint+'|'+row['base_pair_id']]=digest(done['binding'])
            pair.append(done['binding'])
        fields=('arm','steps','lr','forms_sha256','split','supervision','temperature','quantization',
            'teacher_max_new_tokens','reader_path','reader_weights_sha256','reader_config_sha256',
            'target_pipeline_sha256','reader_objective_sha256','teacher_history_scope')
        if any(pair[0][k]!=pair[1][k] for k in fields):
            raise ValueError('Unpaired teacher scientific setting')
        if pair[0].get('context_suite','mcq')!='mcq' or pair[1].get('context_suite')!='diverse-v1':
            raise ValueError('Wrong context contrast')
    return dict(files=files,teacher_bindings=bindings,latent_shape=shape[0],latent_dtype=shape[1],
                parent_sha256=PARENT_SHA,ids=[r['base_pair_id'] for r in cohort_rows()['pilot']])


def verify_training(output,banks):
    manifests,logs={},{}
    for endpoint in ENDPOINTS:
        folder=output/endpoint/'train';done=read(folder/'complete.json');m=read(folder/'manifest.json')
        expected=dict(arm='B',stage='write',steps=128,effective_batch=4,split='pilot',
            parent_sha256=PARENT_SHA,seed=20260924,teacher_supervision='prompt_matching',teacher_steps=288)
        if done['steps']!=128 or done['checkpoint_sha256']!=sha(folder/'checkpoint-final.pt') or any(m.get(k)!=v for k,v in expected.items()):
            raise ValueError('Writer endpoint mismatch')
        targets={p:banks['files'][str(BANKS[endpoint]/p.replace(':','_')/'latent.pt')] for p in banks['ids']}
        bindings={p:banks['teacher_bindings'][endpoint+'|'+p] for p in banks['ids']}
        if m['targets']!=targets or m['teacher_binding_hashes']!=bindings:
            raise ValueError('Writer trained on unexpected targets')
        log=read_lines(folder/'optimization.jsonl')
        if [r['step'] for r in log]!=list(range(1,129)) or any(not math.isfinite(r['grad_norm']) or r['grad_norm']<=0 or len(r['draws'])!=4 for r in log):
            raise ValueError('Incomplete updates or invalid gradient')
        for row in log:
            if any(not math.isfinite(d['mse']) or d['pair_id'] not in banks['ids'] or d['position']!=0 for d in row['draws']):
                raise ValueError('Invalid optimization draw')
        manifests[endpoint]=m;logs[endpoint]=log
    signature=lambda log:[[(d['pair_id'],d['position'],d['sigma']) for d in r['draws']] for r in log]
    if signature(logs[ENDPOINTS[0]])!=signature(logs[ENDPOINTS[1]]):
        raise ValueError('Unpaired draw order/sigma')
    return dict(steps_per_arm=128,draws_per_arm=512,finite_nonzero_gradient_updates=256,paired_draws=True)


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    assets,files={},{}
    for endpoint in ENDPOINTS:
        checkpoint=output/endpoint/'train/checkpoint-final.pt'
        images=output/endpoint/'images';m=read(images/'manifest.json')
        expected=dict(checkpoint_sha256=sha(checkpoint),split='pilot',steps=28,cfg=1,noise_chains=2,
            inter_turns=0,state='only reopened uint8 RGB PNG; fresh Gaussian each write')
        if m!=expected or len(list(images.glob('*/seed-*/complete.json')))!=32:
            raise ValueError('PNG protocol or denominator mismatch')
        for p in (checkpoint,checkpoint.parent/'complete.json',checkpoint.parent/'manifest.json',images/'manifest.json'):
            files[str(p)]=sha(p)
        for row in cohort_rows()['pilot']:
            pid=row['base_pair_id']
            for chain in range(2):
                folder=images/pid.replace(':','_')/f'seed-{chain}';png=folder/'prefix-00.png';checksum=sha(png)
                if read(folder/'complete.json')!=dict(binding=m,png_hashes={png.name:checksum}):
                    raise ValueError('PNG hash/binding mismatch')
                expected_write=dict(position=0,source_png_sha256=None,output_png_sha256=checksum,
                    noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),event=event_text(row['history'][:2]))
                if read_lines(folder/'writes.jsonl')!=[expected_write]:
                    raise ValueError('History/noise mismatch')
                with Image.open(png) as image:
                    if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Bad PNG')
                    image.load()
                assets[f'{endpoint}|{pid}|{chain}']=dict(path=str(png),sha256=checksum,complete_sha256=sha(folder/'complete.json'))
    return dict(files=files,assets=assets)


def build_jobs(args,phase):
    jobs=[]
    for gpu,endpoint in enumerate(ENDPOINTS):
        train=args.output/endpoint/'train'
        common=[sys.executable,'scripts/experiments/prefeval_k1_writer.py',phase,'--arm','B','--split','pilot',
            '--ids-file',str(ROOT/'configs/experiments/context_coverage_ids.json'),
            '--base',str(args.base),'--official-source',str(args.official_source),'--checkpoint',
            str(OLD/'train/checkpoint-final.pt' if phase=='train' else train/'checkpoint-final.pt')]
        if phase=='train':
            common+=['--output',str(train),'--teachers',str(BANKS[endpoint]),'--teacher-supervision','prompt_matching',
                '--teacher-steps','288','--steps','128','--snapshot-steps','4']
        else:
            common+=['--output',str(args.output/endpoint/'images'),'--inter-turns','0','--noise-chains','2']
        jobs.append(dict(name=phase+'-'+endpoint,gpu=gpu,command=common))
    return jobs


def remaining_seconds(output):
    previous=other_seconds(output.parent)+sum(accrued_seconds(output.parent/p) for p in ('writer-readout-v1','route-functional-v1'))
    current=accrued_seconds(output)
    return min(3600-current,16*3600-previous-current),previous


def report(output,reference,protocol):
    spec=read(protocol);old,ref=load_reference(reference,spec,protocol,'pilot');frozen=read(output/'inputs.json')
    if ref!=frozen['reference'] or verify_images(output)!=frozen['upstream'] or verify_banks()!=read(output/'banks.json'):
        raise ValueError('Frozen fit inputs changed')
    verify_training(output,read(output/'banks.json'))
    rows=cohort_rows()['pilot'];queries={q['id']:q for q in spec['queries']};donors=donor_map(rows);new=[]
    for shard in range(2):
        done=read(output/f'finished-{shard}.json');path=output/f'readout-{shard}.jsonl';ident=output/f'identity-{shard}.json'
        identity=read(ident);values=read_lines(path)
        if (done['readout_sha256']!=sha(path) or done['identity_sha256']!=sha(ident) or done['rows']!=len(values)
                or identity['inputs_sha256']!=sha(output/'inputs.json') or identity['protocol_sha256']!=sha(protocol)
                or identity['source_sha256']!=sha(ROOT/'scripts/experiments/prefeval_route_functional.py')
                or identity['assignment']!=[r['base_pair_id'] for r in rows[shard::2]]):
            raise ValueError('Readout receipt mismatch')
        index_rows(values,new_keys(identity['assignment'],spec,ENDPOINTS))
        for r in values:validate_new_row(r,ref['targets'],frozen['upstream']['assets'],donors,queries,'pilot')
        new.extend(values)
    result=summarize(old,new,[r['base_pair_id'] for r in rows],spec,ENDPOINTS,
        {'context-narrow':'b730','context-diverse':'context-narrow'})
    save_once(output/'comparison.json',result)
    return result


def main(args):
    if args.output.resolve()!=RUN/'context-fit-v1':raise ValueError('Require isolated fit output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty code')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        protocol=ROOT/'configs/experiments/context_readout_audit.json';reference=RUN/'writer-readout-v1'
        save_once(args.output/'plan.json',dict(commit=commit,protocol_sha256=sha(protocol),
            plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_FIT_PLAN.md'),
            base=str(args.base),reader=str(args.reader),official_source=str(args.official_source),
            ids_sha256=sha(ROOT/'configs/experiments/context_coverage_ids.json'),gpu_hours_cap=1,campaign_cap=16))
        banks=verify_banks();save_once(args.output/'banks.json',banks)
        _,ref=load_reference(reference,read(protocol),protocol,'pilot')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPUs occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [p for p in RUN.glob('*/active-owner/owner.json') if p!=claim/'owner.json']:raise ValueError('Other campaign owner')
        left,previous=remaining_seconds(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(jobs):
            if time.monotonic()+30>=deadline:raise ValueError('No time for another stage within frozen budget')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:f.result()
        for endpoint in ENDPOINTS:
            train=args.output/endpoint/'train'
            if (train/'optimization.jsonl').exists() and not (train/'complete.json').exists():
                raise ValueError('Partial native training requires explicit checkpoint/tail recovery audit')
        write_json(args.output/'status.json',dict(status='training',time=time.time(),commit=commit))
        group(build_jobs(args,'train'))
        save_once(args.output/'training-audit.json',verify_training(args.output,banks))
        write_json(args.output/'status.json',dict(status='rollout',time=time.time(),commit=commit))
        group(build_jobs(args,'rollout'))
        save_once(args.output/'inputs.json',dict(cohort='pilot',endpoints=list(ENDPOINTS),reference=ref,upstream=verify_images(args.output)))
        write_json(args.output/'status.json',dict(status='readout',time=time.time(),commit=commit))
        group([dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,'scripts/experiments/prefeval_route_functional.py',
            '--output',str(args.output),'--reader',str(args.reader),'--protocol',str(protocol),'--shard',str(i)]) for i in range(2)])
        result=report(args.output,reference,protocol)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),new_rows=result['new_rows'],
            combined_rows=result['combined_rows'],gpu_hours=accrued_seconds(args.output)/3600,
            campaign_gpu_hours=(previous+accrued_seconds(args.output))/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','base','reader','official-source'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
