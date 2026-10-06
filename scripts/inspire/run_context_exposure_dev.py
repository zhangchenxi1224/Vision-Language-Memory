"""Read-only internal-dev evaluation of the final fixed32/256 Writer."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_exposure as e
from scripts.inspire import run_context_dev as d
from scripts.inspire.run_writer_readout import execute,accrued_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.experiments.prefeval_route_functional import (
 read,read_lines,sha,save_once,load_reference,load_target,index_rows,new_keys,
 donor_map,validate_new_row,cohort_rows,summarize as generic_summary,event_text,KEYS,
)
p=e.p
RUN=e.RUN
SOURCE=RUN/'context-exposure-v1'
PREVIOUS=RUN/'context-dev-v1'
PROTOCOL=ROOT/'configs/experiments/context_readout_audit.json'
ENDPOINT='writer32-256'
CHECKPOINT_SHA='5aaffb9983f2c16b176790237109502fc02516dda8a3e5c0ee9ecedba5f036a8'
ENDPOINTS=(*d.ENDPOINTS,ENDPOINT)


def remaining_seconds(output):
    _,prior,source=e.budget(SOURCE)
    previous=prior+source
    current=accrued_seconds(output)
    return min(2*3600-current,16*3600-previous-current),previous,current


def verify_source():
    if read(SOURCE/'status.json')['status']!='completed':raise ValueError('Exposure source incomplete')
    if sha(SOURCE/'train/checkpoint-final.pt')!=CHECKPOINT_SHA:raise ValueError('Wrong fixed final256')
    e.report(SOURCE)
    return {str(SOURCE/name):sha(SOURCE/name) for name in (
        'plan.json','comparison.json','training-audit.json','resume-extension.json',
        'train/checkpoint-final.pt','train/resume.pt','train/manifest.json','train/complete.json','train/optimization.jsonl')}


def previous_inputs():
    d.report(PREVIOUS,RUN/'context-fit-v1',RUN/'writer-readout-v1',PROTOCOL)
    old,ref=load_reference(RUN/'writer-readout-v1',read(PROTOCOL),PROTOCOL,'dev')
    prior=[]
    for shard in range(2):
        prior.extend(read_lines(PREVIOUS/f'readout-{shard}.jsonl'))
        for name in (f'readout-{shard}.jsonl',f'identity-{shard}.json',f'finished-{shard}.json'):
            ref['files'][str(PREVIOUS/name)]=sha(PREVIOUS/name)
    for name in ('inputs.json','source.json','comparison.json'):
        ref['files'][str(PREVIOUS/name)]=sha(PREVIOUS/name)
    return old,prior,ref


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    images=output/'images';m=read(images/'manifest.json')
    expected=dict(checkpoint_sha256=CHECKPOINT_SHA,split='dev',steps=28,cfg=1,noise_chains=2,
        inter_turns=0,state='only reopened uint8 RGB PNG; fresh Gaussian each write')
    if m!=expected or len(list(images.glob('*/seed-*/complete.json')))!=180:
        raise ValueError('PNG protocol or denominator mismatch')
    files={str(images/'manifest.json'):sha(images/'manifest.json'),str(SOURCE/'train/checkpoint-final.pt'):CHECKPOINT_SHA}
    if sha(SOURCE/'train/checkpoint-final.pt')!=CHECKPOINT_SHA:raise ValueError('Checkpoint changed')
    assets={}
    for row in cohort_rows()['dev']:
        pid=row['base_pair_id']
        for chain in range(2):
            folder=images/pid.replace(':','_')/f'seed-{chain}';png=folder/'prefix-00.png';checksum=sha(png)
            if read(folder/'complete.json')!=dict(binding=m,png_hashes={png.name:checksum}):raise ValueError('PNG binding changed')
            expected_write=dict(position=0,source_png_sha256=None,output_png_sha256=checksum,
                noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),event=event_text(row['history'][:2]))
            if read_lines(folder/'writes.jsonl')!=[expected_write]:raise ValueError('Unpaired history/noise')
            with Image.open(png) as image:
                if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Bad PNG')
                image.load()
            assets[f'{ENDPOINT}|{pid}|{chain}']=dict(path=str(png),sha256=checksum,complete_sha256=sha(folder/'complete.json'))
    return dict(files=files,assets=assets)


def summarize(old,prior,new,spec):
    ids=[r['base_pair_id'] for r in cohort_rows()['dev']]
    result=generic_summary(old,prior+new,ids,spec,ENDPOINTS,
        {'context-narrow':'b730','context-diverse':'context-narrow',ENDPOINT:'context-diverse'},cohort='dev')
    if (len(old),len(prior),len(new),result['combined_rows'])!=(15120,8640,4320,28080):
        raise ValueError('Wrong full dev denominator')
    result.update(new_rows=len(new),reused_rows=len(old)+len(prior),
        comparison='Fixed final32/256 vs historical16/128; equal draws/history, unequal total compute; exploratory dev90')
    return result


def evaluate(args):
    import torch
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0); torch.set_num_threads(1)
    spec = read(p.PROTOCOL); p.validate_protocol(spec); frozen = read(args.output/'inputs.json'); ref = frozen['reference']
    for path, h in {**ref['files'], **frozen['images']['files']}.items():
        if sha(Path(path)) != h: raise ValueError('Frozen source changed')
    assets = frozen['images']['assets']; donors = donor_map(cohort_rows()['dev'])
    for asset in assets.values():
        if sha(Path(asset['path'])) != asset['sha256']: raise ValueError('PNG changed')
    binding = p.reader_binding(args.reader)
    if binding != ref['reader']: raise ValueError('Reader changed')
    rows = cohort_rows()['dev'][args.shard::2]; queries = {q['id']: q for q in spec['queries']}
    identity = dict(inputs_sha256=sha(args.output/'inputs.json'), source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(p.PROTOCOL), assignment=[r['base_pair_id'] for r in rows])
    ident = args.output/f'identity-{args.shard}.json'; save_once(ident, identity)
    dest = args.output/f'readout-{args.shard}.jsonl'; records = read_lines(dest) if dest.exists() else []
    expected = new_keys(identity['assignment'], spec, (ENDPOINT,)); keys = [tuple(r[k] for k in KEYS) for r in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected: raise ValueError('Foreign/duplicate partial rows')
    for r in records:
        validate_new_row(r, ref['targets'], assets, donors, queries, 'dev')
        if not math.isfinite(r['kl']): raise ValueError('Nonfinite partial row')
    keys = set(keys); processor, reader = load_reader(args.reader, 'cuda:0')
    eos = assistant_termination_contract(reader, processor)['assistant_end_token_id']
    save_once(args.output/f'termination-{args.shard}.json', dict(eos=eos, pad=processor.tokenizer.pad_token_id))
    with torch.no_grad():
        for row in rows:
            pid = row['base_pair_id']
            for q in spec['queries']:
                meta = ref['targets'][f'{pid}|{q["id"]}']
                target = load_target(meta, binding, row, q['query'], eos, processor.tokenizer.pad_token_id)
                teacher = target['logits'].to('cuda:0'); out = None
                for key in sorted(new_keys([pid], {'queries': [q]}, (ENDPOINT,))):
                    if key in keys: continue
                    _, _, endpoint, control, chain = key; donor = donors[pid] if control == 'mismatch' else None
                    asset = assets[f'{endpoint}|{donor or pid}|{chain}']
                    out = qwen3vl_continuation_logits(model=reader, processor=processor, image=read_png(asset['path']).to('cuda:0'),
                        query=q['query'], target_ids=torch.tensor([target['target_ids']]), device='cuda:0',
                        require_image_grad=False, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                    kl = float(soft_target_kl_divergence(out.target_logits, teacher))
                    if not math.isfinite(kl): raise ValueError('Nonfinite KL')
                    value = dict(zip(KEYS, key)); value.update(split='dev', family=q['family'], kl=kl, png=asset,
                        donor_pair_id=donor, teacher_target=meta['path'], teacher_logits_sha256=target['logits_sha256'], target_ids=target['target_ids'])
                    append(dest, value); keys.add(key)
                del teacher, target, out
            print(json.dumps(dict(completed=pid, shard=args.shard, rows=len(keys))), flush=True)
    if keys != expected: raise ValueError('Incomplete denominator')
    save_once(args.output/f'finished-{args.shard}.json', dict(rows=len(keys), identity_sha256=sha(ident), readout_sha256=sha(dest)))


def report(output):
    if verify_source()!=read(output/'source.json'):raise ValueError('Training source changed')
    old,prior,ref=previous_inputs();frozen=read(output/'inputs.json');spec=read(PROTOCOL)
    if ref!=frozen['reference'] or verify_images(output)!=frozen['images']:raise ValueError('Frozen dev inputs changed')
    rows=cohort_rows()['dev'];donors=donor_map(rows);queries={q['id']:q for q in spec['queries']};new=[]
    termination=read(e.PREVIOUS/'termination-0.json')
    for shard in range(2):
        path=output/f'readout-{shard}.jsonl';ident=output/f'identity-{shard}.json';records=read_lines(path)
        identity=read(ident);done=read(output/f'finished-{shard}.json')
        if (done!=dict(rows=len(records),identity_sha256=sha(ident),readout_sha256=sha(path)) or
            identity!=dict(inputs_sha256=sha(output/'inputs.json'),source_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),assignment=[r['base_pair_id'] for r in rows[shard::2]])):
            raise ValueError('Readout receipt changed')
        index_rows(records,new_keys(identity['assignment'],spec,(ENDPOINT,)))
        for row in records:validate_new_row(row,ref['targets'],frozen['images']['assets'],donors,queries,'dev')
        if read(output/f'termination-{shard}.json')!=termination:raise ValueError('Reader termination changed')
        new.extend(records)
    for row in rows:
        for q in spec['queries']:
            load_target(ref['targets'][f"{row['base_pair_id']}|{q['id']}"],ref['reader'],row,q['query'],termination['eos'],termination['pad'])
    result=summarize(old,prior,new,spec);save_once(output/'comparison.json',result)
    return result


def rollout_job(args):
    return dict(name='rollout',gpu=0,command=[sys.executable,'scripts/experiments/prefeval_k1_writer.py',
        'rollout','--arm','B','--split','dev','--base',str(args.base),'--official-source',str(args.official_source),
        '--checkpoint',str(SOURCE/'train/checkpoint-final.pt'),'--output',str(args.output/'images'),
        '--inter-turns','0','--noise-chains','2'])


def main(args):
    if args.output.resolve()!=RUN/'context-exposure-dev-v1':raise ValueError('Wrong isolated output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    ids=[r['base_pair_id'] for r in cohort_rows()['dev']]
    if len(ids)!=90 or set(ids)&{r['base_pair_id'] for r in p.population()[0]}:raise ValueError('Dev/train overlap')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json',dict(commit=commit,protocol_sha256=sha(PROTOCOL),
            plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_EXPOSURE_DEV_PLAN.md'),
            source=str(SOURCE),checkpoint_sha256=CHECKPOINT_SHA,base=str(args.base),reader=str(args.reader),
            official_source=str(args.official_source),ids=ids,new_rows=4320,combined_rows=28080,gpu_hours_cap=2,campaign_cap=16))
        save_once(args.output/'source.json',verify_source());_,_,ref=previous_inputs()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [x for x in RUN.glob('*/active-owner/owner.json') if x!=claim/'owner.json']:raise ValueError('Another owner')
        left,previous,_=remaining_seconds(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(jobs):
            if time.monotonic()+30>=deadline:raise ValueError('No time for next stage')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:f.result()
        write_json(args.output/'status.json',dict(status='rollout',time=time.time(),commit=commit))
        group([rollout_job(args)])
        save_once(args.output/'inputs.json',dict(reference=ref,images=verify_images(args.output)))
        write_json(args.output/'status.json',dict(status='readout',time=time.time(),commit=commit))
        group([dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,str(Path(__file__)),'evaluate',
            '--output',str(args.output),'--reader',str(args.reader),'--shard',str(i)]) for i in range(2)])
        result=report(args.output);cost=accrued_seconds(args.output)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),rows=result['combined_rows'],
            gpu_hours=cost/3600,campaign_gpu_hours=(previous+cost)/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','evaluate'))
    for name in ('output','base','reader','official-source'):parser.add_argument('--'+name,type=Path,required=name in ('output','reader'))
    parser.add_argument('--shard',type=int,choices=(0,1));args=parser.parse_args()
    (main if args.phase=='run' else evaluate)(args)
