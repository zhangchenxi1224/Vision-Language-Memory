"""Fixed trained-history role-layout diagnostic; native weights/targets remain read-only."""
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
from scripts.inspire import run_context_exposure_dev as d
from scripts.inspire.run_writer_readout import execute,accrued_seconds
from scripts.experiments.prefeval_route_functional import read,read_lines,sha,save_once,new_keys,index_rows,load_target,validate_new_row,KEYS
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.reporting.context_coverage_report import paired_interval
p=e.p
RUN=e.RUN
SOURCE=d.SOURCE
DEV=RUN/'context-exposure-dev-v1'
CHECKPOINT_SHA=d.CHECKPOINT_SHA
ENDPOINT='writer32-256-layout'
PLAN=ROOT/'reports/context-coverage-20261006/CONTEXT_LAYOUT_PLAN.md'


def layout_text(exchange):
    if len(exchange)!=2 or [m['role'] for m in exchange]!=['user','assistant']:
        raise ValueError('Only the original first exchange is permitted')
    if any(not isinstance(m['content'],str) for m in exchange):raise ValueError('Non-text history')
    return '\n\n'.join('### '+m['role']+'\n'+m['content'] for m in exchange)


def remaining_seconds(output):
    _,prior,dev=d.remaining_seconds(DEV)
    previous=prior+dev;current=accrued_seconds(output)
    return min(3600-current,16*3600-previous-current),previous,current


def previous_inputs():
    if read(DEV/'status.json')['status']!='completed' or not read(DEV/'audit-0548.json')['exact_report_recomputation']:
        raise ValueError('Need independently verified completed dev before next factor')
    e.report(SOURCE)
    old,ref=e.previous_inputs()
    files=dict(ref['files'])
    for shard in range(2):
        old.extend(read_lines(SOURCE/f'readout-{shard}.jsonl'))
        for name in (f'readout-{shard}.jsonl',f'identity-{shard}.json',f'finished-{shard}.json'):
            path=SOURCE/name;files[str(path)]=sha(path)
    for path in (SOURCE/'comparison.json',SOURCE/'inputs.json',SOURCE/'train/checkpoint-final.pt',DEV/'audit-0548.json',DEV/'comparison.json',DEV/'status.json',PLAN):
        files[str(path)]=sha(path)
    if sha(SOURCE/'train/checkpoint-final.pt')!=CHECKPOINT_SHA:raise ValueError('Wrong fixed weights')
    ids=[r['base_pair_id'] for r in p.population()[0]]
    index_rows(old,p.expected_keys(ids,read(p.PROTOCOL))|new_keys(ids,read(p.PROTOCOL),(e.ENDPOINT,)))
    if len(old)!=5376:raise ValueError('Wrong source denominator')
    return old,dict(ref,files=files)


def image_binding():
    return dict(checkpoint_sha256=CHECKPOINT_SHA,split='pilot',steps=28,cfg=1,noise_chains=2,inter_turns=0,
        state='only reopened uint8 RGB PNG; fresh Gaussian each write',layout='markdown-role-headings-v1',
        source_sha256=sha(Path(__file__)),plan_sha256=sha(PLAN))


def native_write(pipe,row,chain,text,destination,device):
    import torch
    from PIL import Image
    from scripts.experiments.prefeval_k1_writer import encode_source,_save_image,NativeBaseEditSampler,decode_model_latents_unit_interval,stable_seed
    image=Image.new('RGB',(1024,1024),(128,128,128))
    source=encode_source(pipe,image,device)
    seed=stable_seed(20260924,f"rollout:{row['base_pair_id']}:{chain}",0)
    generator=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn(source.shape,generator=generator,device=device,dtype=source.dtype)
    sampler=NativeBaseEditSampler(pipe,source_image=image,event_text=text,guidance_scale=1.)
    with torch.no_grad():
        generated=sampler(source_latents=source,noise_latents=noise,num_steps=28,return_trajectory=False)
        _save_image(destination,decode_model_latents_unit_interval(pipe.vae,generated.latents,clamp=True))
    return dict(position=0,source_png_sha256=None,output_png_sha256=sha(destination),noise_seed=seed,event=text)


def rollout(args):
    import torch
    from scripts.experiments.prefeval_k1_writer import load_pipe
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0);torch.set_num_threads(1)
    args.device='cuda:0';args.checkpoint=SOURCE/'train/checkpoint-final.pt'
    if sha(args.checkpoint)!=CHECKPOINT_SHA:raise ValueError('Fixed weights changed')
    for path,h in read(args.output/'reference.json')['files'].items():
        if sha(Path(path))!=h:raise ValueError('Source changed')
    pipe=load_pipe(args);pipe.unet.requires_grad_(False)
    rows=p.population()[0];images=args.output/'images';images.mkdir(exist_ok=True)
    binding=image_binding();save_once(images/'manifest.json',binding)
    # Same helper must reproduce a canonical source PNG bit-for-bit before perturbation.
    row=rows[0];pid=row['base_pair_id'];original=SOURCE/'images'/pid.replace(':','_')/'seed-0/prefix-00.png'
    calibration=args.output/'calibration';calibration.mkdir(exist_ok=True)
    trace=native_write(pipe,row,0,p.event_text(row['history'][:2]),calibration/'canonical.png',args.device)
    if trace['output_png_sha256']!=sha(original):raise ValueError('Canonical PNG parity failed; no layout rollout allowed')
    save_once(calibration/'parity.json',dict(pair_id=pid,chain=0,source=str(original),sha256=sha(original),write=trace,binding=binding))
    for row in rows:
        pid=row['base_pair_id']
        for chain in range(2):
            folder=images/pid.replace(':','_')/f'seed-{chain}';folder.mkdir(parents=True,exist_ok=True)
            png=folder/'prefix-00.png';done=folder/'complete.json'
            if done.exists():
                if read(done)!=dict(binding=binding,png_hashes={png.name:sha(png)}):raise ValueError('Partial PNG identity changed')
                continue
            trace=native_write(pipe,row,chain,layout_text(row['history'][:2]),png,args.device)
            # A single write per image; replace only this uncommitted local trace on retry.
            (folder/'writes.jsonl').write_text(json.dumps(trace,ensure_ascii=False)+'\n',encoding='utf8')
            save_once(done,dict(binding=binding,png_hashes={png.name:trace['output_png_sha256']}))
            print(json.dumps(dict(completed=pid,chain=chain)),flush=True)


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    images=output/'images';binding=image_binding();rows=p.population()[0]
    if read(images/'manifest.json')!=binding or len(list(images.glob('*/seed-*/complete.json')))!=64:raise ValueError('Wrong layout denominator/binding')
    calibration=output/'calibration';r=rows[0];pid=r['base_pair_id'];original=SOURCE/'images'/pid.replace(':','_')/'seed-0/prefix-00.png'
    h=sha(original);trace=dict(position=0,source_png_sha256=None,output_png_sha256=h,
        noise_seed=stable_seed(20260924,f'rollout:{pid}:0',0),event=p.event_text(r['history'][:2]))
    if sha(calibration/'canonical.png')!=h or read(calibration/'parity.json')!=dict(pair_id=pid,chain=0,source=str(original),sha256=h,write=trace,binding=binding):
        raise ValueError('Canonical parity evidence changed')
    files={str(path):sha(path) for path in (images/'manifest.json',calibration/'canonical.png',calibration/'parity.json',SOURCE/'train/checkpoint-final.pt')}
    if files[str(SOURCE/'train/checkpoint-final.pt')]!=CHECKPOINT_SHA:raise ValueError('Fixed weights changed')
    assets={}
    for row in rows:
        pid=row['base_pair_id']
        for chain in range(2):
            folder=images/pid.replace(':','_')/f'seed-{chain}';png=folder/'prefix-00.png';checksum=sha(png)
            if read(folder/'complete.json')!=dict(binding=binding,png_hashes={png.name:checksum}):raise ValueError('PNG binding changed')
            trace=dict(position=0,source_png_sha256=None,output_png_sha256=checksum,
                noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),event=layout_text(row['history'][:2]))
            if read_lines(folder/'writes.jsonl')!=[trace]:raise ValueError('Layout/history/noise mismatch')
            with Image.open(png) as image:
                if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Invalid PNG')
                image.load()
            assets[f'{ENDPOINT}|{pid}|{chain}']=dict(path=str(png),sha256=checksum,complete_sha256=sha(folder/'complete.json'))
    return dict(files=files,assets=assets)


def summarize(old,new,spec):
    groups=p.strata();ids=[r['base_pair_id'] for rows in groups.values() for r in rows]
    table=index_rows(old,p.expected_keys(ids,spec)|new_keys(ids,spec,(e.ENDPOINT,)))
    table.update(index_rows(new,new_keys(ids,spec,(ENDPOINT,))))
    if (len(old),len(new),len(table))!=(5376,1536,6912):raise ValueError('Wrong complete denominator')
    for pid in ids:
        for q in spec['queries']:
            reference=table[pid,q['id'],'text','text',0]
            for key in new_keys([pid],{'queries':[q]},(e.ENDPOINT,ENDPOINT)):
                if any(table[key][k]!=reference[k] for k in ('teacher_target','target_ids','teacher_logits_sha256')):raise ValueError('Unpaired teacher prefix/distribution')
    result=dict(combined_rows=len(table),new_rows=len(new),reused_rows=len(old),
        metric='Trained-history role-layout sensitivity only; no dev or generalization claim',strata={})
    for group,rows in groups.items():
        ids=[r['base_pair_id'] for r in rows];out=dict(independent_n=len(ids),families={})
        for family in ('recall','application','neutral'):
            qs=[q['id'] for q in spec['queries'] if q['family']==family]
            def vals(endpoint,control):
                ns=[0] if control in ('blank','text') else [0,1]
                return [sum(table[pid,q,endpoint,control,n]['kl'] for q in qs for n in ns)/(len(qs)*len(ns)) for pid in ids]
            mean=lambda x:sum(x)/len(x)
            blank=vals('blank','blank');f=dict(blank_kl=mean(blank),text_self_consistency_kl=mean(vals('text','text')),endpoints={})
            for endpoint in (e.ENDPOINT,ENDPOINT):
                a,b=vals(endpoint,'memory'),vals(endpoint,'mismatch')
                f['endpoints'][endpoint]=dict(memory_kl=mean(a),mismatch_kl=mean(b),
                    mismatch_minus_memory=paired_interval([x-y for x,y in zip(b,a)]),
                    blank_minus_memory=paired_interval([x-y for x,y in zip(blank,a)]))
            a,b=vals(e.ENDPOINT,'memory'),vals(ENDPOINT,'memory')
            x,y=vals(e.ENDPOINT,'mismatch'),vals(ENDPOINT,'mismatch')
            f['layout_minus_canonical']=paired_interval([v-u for u,v in zip(a,b)])
            f['canonical_minus_layout_specificity']=paired_interval([(u-v)-(w-z) for u,v,w,z in zip(x,a,y,b)])
            out['families'][family]=f
        result['strata'][group]=out
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
    assets = frozen['images']['assets']; donors, _ = p.donors_and_groups()
    for asset in assets.values():
        if sha(Path(asset['path'])) != asset['sha256']: raise ValueError('PNG changed')
    binding = p.reader_binding(args.reader)
    if binding != ref['reader']: raise ValueError('Reader changed')
    rows = p.population()[0][args.shard::2]; queries = {q['id']: q for q in spec['queries']}
    identity = dict(inputs_sha256=sha(args.output/'inputs.json'), source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(p.PROTOCOL), assignment=[r['base_pair_id'] for r in rows])
    ident = args.output/f'identity-{args.shard}.json'; save_once(ident, identity)
    dest = args.output/f'readout-{args.shard}.jsonl'; records = read_lines(dest) if dest.exists() else []
    expected = new_keys(identity['assignment'], spec, (ENDPOINT,)); keys = [tuple(r[k] for k in KEYS) for r in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected: raise ValueError('Foreign/duplicate partial rows')
    for r in records:
        validate_new_row(r, ref['targets'], assets, donors, queries, 'pilot')
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
                    value = dict(zip(KEYS, key)); value.update(split='pilot', family=q['family'], kl=kl, png=asset,
                        donor_pair_id=donor, teacher_target=meta['path'], teacher_logits_sha256=target['logits_sha256'], target_ids=target['target_ids'])
                    append(dest, value); keys.add(key)
                del teacher, target, out
            print(json.dumps(dict(completed=pid, shard=args.shard, rows=len(keys))), flush=True)
    if keys != expected: raise ValueError('Incomplete denominator')
    save_once(args.output/f'finished-{args.shard}.json', dict(rows=len(keys), identity_sha256=sha(ident), readout_sha256=sha(dest)))


def report(output):
    old,ref=previous_inputs();frozen=read(output/'inputs.json');spec=read(p.PROTOCOL)
    if ref!=frozen['reference'] or verify_images(output)!=frozen['images']:raise ValueError('Frozen inputs changed')
    rows=p.population()[0];new=[];donors,_=p.donors_and_groups();queries={q['id']:q for q in spec['queries']}
    termination=read(SOURCE/'termination-0.json')
    for shard in range(2):
        path=output/f'readout-{shard}.jsonl';ident=output/f'identity-{shard}.json';records=read_lines(path)
        identity=read(ident)
        if read(output/f'finished-{shard}.json')!=dict(rows=len(records),identity_sha256=sha(ident),readout_sha256=sha(path)) or identity!=dict(inputs_sha256=sha(output/'inputs.json'),source_sha256=sha(Path(__file__)),protocol_sha256=sha(p.PROTOCOL),assignment=[r['base_pair_id'] for r in rows[shard::2]]):raise ValueError('Readout receipt changed')
        index_rows(records,new_keys(identity['assignment'],spec,(ENDPOINT,)))
        for row in records:validate_new_row(row,ref['targets'],frozen['images']['assets'],donors,queries,'pilot')
        if read(output/f'termination-{shard}.json')!=termination:raise ValueError('Termination changed')
        new.extend(records)
    for row in rows:
        for q in spec['queries']:load_target(ref['targets'][f"{row['base_pair_id']}|{q['id']}"],ref['reader'],row,q['query'],termination['eos'],termination['pad'])
    result=summarize(old,new,spec);save_once(output/'comparison.json',result);return result


def rollout_job(args):
    return dict(name='rollout',gpu=0,command=[sys.executable,str(Path(__file__)),'rollout','--output',str(args.output),
        '--base',str(args.base),'--reader',str(args.reader),'--official-source',str(args.official_source)])


def main(args):
    if args.output.resolve()!=RUN/'context-layout-v1':raise ValueError('Wrong isolated output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json',dict(commit=commit,plan_sha256=sha(PLAN),protocol_sha256=sha(p.PROTOCOL),
            checkpoint_sha256=CHECKPOINT_SHA,base=str(args.base),reader=str(args.reader),official_source=str(args.official_source),
            ids=[r['base_pair_id'] for r in p.population()[0]],new_pngs=65,new_rows=1536,combined_rows=6912,gpu_hours_cap=1,campaign_cap=16))
        write_json(args.output/'status.json',dict(status='source_audit',time=time.time(),commit=commit))
        _,ref=previous_inputs();save_once(args.output/'reference.json',ref)
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
        group([dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,str(Path(__file__)),'evaluate','--output',str(args.output),
            '--reader',str(args.reader),'--shard',str(i)]) for i in range(2)])
        result=report(args.output);cost=accrued_seconds(args.output)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),rows=result['combined_rows'],gpu_hours=cost/3600,campaign_gpu_hours=(previous+cost)/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','rollout','evaluate'))
    for name in ('output','base','reader','official-source'):parser.add_argument('--'+name,type=Path,required=name in ('output','reader'))
    parser.add_argument('--shard',type=int,choices=(0,1));args=parser.parse_args()
    {'run':main,'rollout':rollout,'evaluate':evaluate}[args.phase](args)
