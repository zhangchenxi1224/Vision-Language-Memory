"""Bounded internal-dev readout of the frozen paired context-fit Writers."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire.run_context_fit import RUN,ENDPOINTS,verify_banks,verify_training
from scripts.inspire.run_writer_readout import execute,accrued_seconds,other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.experiments.prefeval_route_functional import (
    read,read_lines,save_once,sha,cohort_rows,load_reference,new_keys,index_rows,
    validate_new_row,donor_map,summarize,event_text,
)

CHECKPOINTS=dict(zip(ENDPOINTS,(
    '30c8e1d85287dedff159324eb7ea00bda68d6df3cde6064172a9b9294113b9c0',
    'b2527682b9560ea29c4c9bbaa5f67b361a3fd8057de3026d15c81bf27ad6ecea')))


def verify_source(source):
    status=read(source/'status.json')
    if status['status']!='completed' or (status['new_rows'],status['combined_rows'])!=(1536,4224):
        raise ValueError('Fit source incomplete')
    files={}
    for e in ENDPOINTS:
        folder=source/e/'train'
        if sha(folder/'checkpoint-final.pt')!=CHECKPOINTS[e]:
            raise ValueError('Wrong fixed final checkpoint')
        for name in ('checkpoint-final.pt','complete.json','manifest.json','optimization.jsonl'):
            p=folder/name;files[str(p)]=sha(p)
    banks=verify_banks()
    if banks!=read(source/'banks.json'):
        raise ValueError('Fit teacher binding changed')
    verify_training(source,banks)
    for name in ('comparison.json','banks.json','plan.json','training-audit.json'):
        files[str(source/name)]=sha(source/name)
    return files


def verify_images(output,source):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    assets,files={},{}
    for endpoint in ENDPOINTS:
        images=output/endpoint/'images';m=read(images/'manifest.json')
        expected=dict(checkpoint_sha256=CHECKPOINTS[endpoint],split='dev',steps=28,cfg=1,noise_chains=2,
            inter_turns=0,state='only reopened uint8 RGB PNG; fresh Gaussian each write')
        if m!=expected or len(list(images.glob('*/seed-*/complete.json')))!=180:
            raise ValueError('PNG protocol or denominator mismatch')
        files[str(images/'manifest.json')]=sha(images/'manifest.json')
        files[str(source/endpoint/'train/checkpoint-final.pt')]=CHECKPOINTS[endpoint]
        for row in cohort_rows()['dev']:
            pid=row['base_pair_id']
            for chain in range(2):
                folder=images/pid.replace(':','_')/f'seed-{chain}';png=folder/'prefix-00.png';checksum=sha(png)
                if read(folder/'complete.json')!=dict(binding=m,png_hashes={png.name:checksum}):
                    raise ValueError('PNG binding changed')
                expected_write=dict(position=0,source_png_sha256=None,output_png_sha256=checksum,
                    noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),event=event_text(row['history'][:2]))
                if read_lines(folder/'writes.jsonl')!=[expected_write]:
                    raise ValueError('Unpaired history/noise')
                with Image.open(png) as image:
                    if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Bad PNG')
                    image.load()
                assets[f'{endpoint}|{pid}|{chain}']=dict(path=str(png),sha256=checksum,complete_sha256=sha(folder/'complete.json'))
    return dict(files=files,assets=assets)


def build_jobs(args):
    return [dict(name='rollout-'+e,gpu=gpu,command=[sys.executable,
        'scripts/experiments/prefeval_k1_writer.py','rollout','--arm','B','--split','dev',
        '--base',str(args.base),'--official-source',str(args.official_source),
        '--checkpoint',str(args.source/e/'train/checkpoint-final.pt'),
        '--output',str(args.output/e/'images'),'--inter-turns','0','--noise-chains','2'])
        for gpu,e in enumerate(ENDPOINTS)]


def remaining_seconds(output):
    previous=other_seconds(output.parent)+sum(accrued_seconds(output.parent/p)
        for p in ('writer-readout-v1','route-functional-v1','context-fit-v1'))
    current=accrued_seconds(output)
    return min(2*3600-current,16*3600-previous-current),previous,current


def report(output,source,reference,protocol):
    if verify_source(source)!=read(output/'source.json'):
        raise ValueError('Fit source changed')
    spec=read(protocol);old,ref=load_reference(reference,spec,protocol,'dev');frozen=read(output/'inputs.json')
    if ref!=frozen['reference'] or verify_images(output,source)!=frozen['upstream']:
        raise ValueError('Frozen dev inputs changed')
    rows=cohort_rows()['dev'];queries={q['id']:q for q in spec['queries']};donors=donor_map(rows);new=[]
    for shard in range(2):
        done=read(output/f'finished-{shard}.json');path=output/f'readout-{shard}.jsonl';ident=output/f'identity-{shard}.json'
        identity=read(ident);values=read_lines(path)
        if (done['readout_sha256']!=sha(path) or done['identity_sha256']!=sha(ident) or done['rows']!=len(values)
                or identity['inputs_sha256']!=sha(output/'inputs.json') or identity['protocol_sha256']!=sha(protocol)
                or identity['source_sha256']!=sha(ROOT/'scripts/experiments/prefeval_route_functional.py')
                or identity['assignment']!=[r['base_pair_id'] for r in rows[shard::2]]):
            raise ValueError('Readout receipt mismatch')
        index_rows(values,new_keys(identity['assignment'],spec,ENDPOINTS))
        for row in values:validate_new_row(row,ref['targets'],frozen['upstream']['assets'],donors,queries,'dev')
        new.extend(values)
    result=summarize(old,new,[r['base_pair_id'] for r in rows],spec,ENDPOINTS,
        {'context-narrow':'b730','context-diverse':'context-narrow'},cohort='dev')
    save_once(output/'comparison.json',result)
    return result


def main(args):
    if args.output.resolve()!=RUN/'context-dev-v1' or args.source.resolve()!=RUN/'context-fit-v1':
        raise ValueError('Require isolated output and fixed fit source')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        protocol=ROOT/'configs/experiments/context_readout_audit.json';reference=RUN/'writer-readout-v1'
        save_once(args.output/'plan.json',dict(commit=commit,protocol_sha256=sha(protocol),
            plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_DEV_PLAN.md'),
            source=str(args.source),base=str(args.base),reader=str(args.reader),official_source=str(args.official_source),
            ids=[r['base_pair_id'] for r in cohort_rows()['dev']],gpu_hours_cap=2,campaign_cap=16,
            expected_new_rows=8640,expected_combined_rows=23760))
        save_once(args.output/'source.json',verify_source(args.source))
        _,ref=load_reference(reference,read(protocol),protocol,'dev')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [p for p in RUN.glob('*/active-owner/owner.json') if p!=claim/'owner.json']:raise ValueError('Another owner exists')
        left,previous,_=remaining_seconds(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(jobs):
            if time.monotonic()+30>=deadline:raise ValueError('No time for another stage')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:f.result()
        write_json(args.output/'status.json',dict(status='rollout',time=time.time(),commit=commit))
        group(build_jobs(args))
        save_once(args.output/'inputs.json',dict(cohort='dev',endpoints=list(ENDPOINTS),reference=ref,upstream=verify_images(args.output,args.source)))
        write_json(args.output/'status.json',dict(status='readout',time=time.time(),commit=commit))
        group([dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,'scripts/experiments/prefeval_route_functional.py',
            '--output',str(args.output),'--reader',str(args.reader),'--protocol',str(protocol),'--shard',str(i)]) for i in range(2)])
        result=report(args.output,args.source,reference,protocol)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),new_rows=result['new_rows'],
            combined_rows=result['combined_rows'],gpu_hours=accrued_seconds(args.output)/3600,
            campaign_gpu_hours=(previous+accrued_seconds(args.output))/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source','base','reader','official-source'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
