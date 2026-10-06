"""Frozen three-format readout of the paired layout-augmentation continuation."""
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
from scripts.inspire import run_context_layout_aug as a
from scripts.inspire.run_writer_readout import execute,accrued_seconds
from scripts.experiments.prefeval_route_functional import read,read_lines,sha,save_once,new_keys,index_rows,load_target,validate_new_row,KEYS
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.reporting.context_coverage_report import paired_interval
l=a.l;p=a.p;RUN=a.RUN
SOURCE=RUN/'context-layout-aug-v1'
PREVIOUS=a.PREVIOUS
STYLES=('canonical','markdown','xml')
ENDPOINTS=tuple(arm+'-'+style for arm in a.ARMS for style in STYLES)


def remaining_seconds(output):
    _,previous,train=a.budget(SOURCE)
    current=accrued_seconds(output)
    return min(2.25*3600-current,3*3600-train-current,16*3600-previous-train-current),previous+train,current


def old_keys(ids,spec):
    return p.expected_keys(ids,spec)|new_keys(ids,spec,(l.e.ENDPOINT,l.ENDPOINT))


def previous_inputs():
    if read(SOURCE/'status.json')['status']!='ready_for_frozen_readout':raise ValueError('Both trained endpoints required')
    if a.verify_source()!=read(SOURCE/'source.json'):raise ValueError('Training source changed')
    training={arm:a.verify_training(SOURCE,arm) for arm in a.ARMS}
    if training!=read(SOURCE/'training-audit.json') or a.verify_parity(SOURCE)!=read(SOURCE/'native-parity.json'):raise ValueError('Training/parity changed')
    # Parent reports were authenticated in their frozen checkout by a.verify_source.
    # Reuse their stored absolute input identities rather than relocating them.
    old,_=l.e.previous_inputs();ref=read(PREVIOUS/'inputs.json')['reference'];files=dict(ref['files'])
    for source in (l.SOURCE,PREVIOUS):
        for shard in range(2):
            old.extend(read_lines(source/f'readout-{shard}.jsonl'))
            for name in (f'readout-{shard}.jsonl',f'identity-{shard}.json',f'finished-{shard}.json'):
                path=source/name;files[str(path)]=sha(path)
        for name in ('comparison.json','inputs.json'):
            path=source/name;files[str(path)]=sha(path)
    for arm in a.ARMS:
        for name in ('manifest.json','optimization.jsonl','checkpoint-final.pt','complete.json','resume.pt','migration.json','resume-step258.pt'):
            path=SOURCE/arm/name;files[str(path)]=sha(path)
    for path in (SOURCE/'training-audit.json',SOURCE/'native-parity.json',SOURCE/'status.json',a.PLAN):files[str(path)]=sha(path)
    ids=[r['base_pair_id'] for r in p.population()[0]];index_rows(old,old_keys(ids,read(p.PROTOCOL)))
    if len(old)!=6912:raise ValueError('Incomplete parent readout')
    return old,dict(ref,files=files,training=training)


def image_binding(arm,ref):
    if arm not in a.ARMS:raise ValueError('Foreign arm')
    return dict(arm=arm,checkpoint_sha256=ref['training'][arm]['checkpoint_sha256'],split='pilot',steps=28,cfg=1,
        noise_chains=2,styles=list(STYLES),inter_turns=0,state='only reopened uint8 RGB PNG; fresh Gaussian each write',
        source_sha256=sha(Path(__file__)),formatter_sha256=sha(ROOT/'scripts/inspire/run_context_layout_aug.py'),plan_sha256=sha(a.PLAN))


def rollout(args):
    import torch
    from scripts.experiments.prefeval_k1_writer import load_pipe
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0);torch.set_num_threads(1)
    ref=read(args.output/'reference.json')
    for path,h in ref['files'].items():
        if sha(Path(path))!=h:raise ValueError('Frozen training source changed')
    binding=image_binding(args.arm,ref);args.checkpoint=SOURCE/args.arm/'checkpoint-final.pt';args.device='cuda:0'
    if sha(args.checkpoint)!=binding['checkpoint_sha256']:raise ValueError('Wrong fixed384 endpoint')
    pipe=load_pipe(args);pipe.unet.requires_grad_(False)
    images=args.output/'images'/args.arm;images.mkdir(parents=True,exist_ok=True);save_once(images/'manifest.json',binding)
    for row in p.population()[0]:
        pid=row['base_pair_id']
        for style in STYLES:
            for chain in range(2):
                folder=images/style/pid.replace(':','_')/f'seed-{chain}';folder.mkdir(parents=True,exist_ok=True)
                png=folder/'prefix-00.png';done=folder/'complete.json'
                if done.exists():
                    if read(done)!=dict(binding=binding,style=style,png_hashes={png.name:sha(png)}):raise ValueError('Partial image identity changed')
                    continue
                trace=l.native_write(pipe,row,chain,a.evaluation_text(row['history'][:2],style),png,args.device)
                (folder/'writes.jsonl').write_text(json.dumps(trace,ensure_ascii=False)+'\n',encoding='utf8')
                save_once(done,dict(binding=binding,style=style,png_hashes={png.name:trace['output_png_sha256']}))
                print(json.dumps(dict(arm=args.arm,completed=pid,style=style,chain=chain)),flush=True)


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    ref=read(output/'reference.json');assets={};files={}
    for arm in a.ARMS:
        images=output/'images'/arm;binding=image_binding(arm,ref)
        if read(images/'manifest.json')!=binding or len(list(images.glob('*/*/seed-*/complete.json')))!=192:raise ValueError('Image binding/denominator mismatch')
        files[str(images/'manifest.json')]=sha(images/'manifest.json')
        for row in p.population()[0]:
            pid=row['base_pair_id']
            for style in STYLES:
                for chain in range(2):
                    folder=images/style/pid.replace(':','_')/f'seed-{chain}';png=folder/'prefix-00.png';h=sha(png)
                    if read(folder/'complete.json')!=dict(binding=binding,style=style,png_hashes={png.name:h}):raise ValueError('PNG receipt changed')
                    trace=dict(position=0,source_png_sha256=None,output_png_sha256=h,noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),event=a.evaluation_text(row['history'][:2],style))
                    if read_lines(folder/'writes.jsonl')!=[trace]:raise ValueError('History/noise/style changed')
                    with Image.open(png) as image:
                        if image.mode!='RGB' or image.size!=(1024,1024):raise ValueError('Invalid PNG')
                        image.load()
                    assets[f'{arm}-{style}|{pid}|{chain}']=dict(path=str(png),sha256=h,complete_sha256=sha(folder/'complete.json'))
    return dict(files=files,assets=assets)


def summarize(old,new,spec):
    groups=p.strata();ids=[r['base_pair_id'] for rs in groups.values() for r in rs]
    table=index_rows(old,old_keys(ids,spec));table.update(index_rows(new,new_keys(ids,spec,ENDPOINTS)))
    if (len(old),len(new),len(table))!=(6912,9216,16128):raise ValueError('Incomplete/foreign full denominator')
    for pid in ids:
        for q in spec['queries']:
            target=table[pid,q['id'],'text','text',0]
            for key in new_keys([pid],{'queries':[q]},ENDPOINTS):
                if any(table[key][k]!=target[k] for k in ('teacher_target','target_ids','teacher_logits_sha256')):raise ValueError('Unpaired teacher distribution')
    result=dict(new_rows=len(new),reused_rows=len(old),combined_rows=len(table),
        metric='Paired trained-history layout augmentation; XML format held out from training; no new-history claim',strata={})
    for group,rows in groups.items():
        ids=[r['base_pair_id'] for r in rows];out=dict(independent_n=len(ids),families={})
        for family in ('recall','application','neutral'):
            qs=[q['id'] for q in spec['queries'] if q['family']==family]
            def vals(endpoint,control):
                ns=[0] if control in ('blank','text') else [0,1]
                return [sum(table[pid,q,endpoint,control,n]['kl'] for q in qs for n in ns)/(len(qs)*len(ns)) for pid in ids]
            mean=lambda x:sum(x)/len(x)
            blank=vals('blank','blank');f=dict(blank_kl=mean(blank),text_self_consistency_kl=mean(vals('text','text')),formats={})
            for style in STYLES:
                g=dict(endpoints={})
                for arm in a.ARMS:
                    endpoint=arm+'-'+style;mem,wrong=vals(endpoint,'memory'),vals(endpoint,'mismatch')
                    g['endpoints'][arm]=dict(memory_kl=mean(mem),mismatch_kl=mean(wrong),
                        mismatch_minus_memory=paired_interval([x-y for x,y in zip(wrong,mem)]),blank_minus_memory=paired_interval([x-y for x,y in zip(blank,mem)]))
                cm,am=vals('canonical-'+style,'memory'),vals('augmented-'+style,'memory')
                cw,aw=vals('canonical-'+style,'mismatch'),vals('augmented-'+style,'mismatch')
                g['control_minus_augmented_memory']=paired_interval([x-y for x,y in zip(cm,am)])
                g['augmented_minus_control_specificity']=paired_interval([(w-z)-(x-y) for w,z,x,y in zip(aw,am,cw,cm)])
                if style!='xml':
                    endpoint=l.e.ENDPOINT if style=='canonical' else l.ENDPOINT
                    g['parent256_memory_kl']=mean(vals(endpoint,'memory'));g['parent256_mismatch_kl']=mean(vals(endpoint,'mismatch'))
                f['formats'][style]=g
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
    expected = new_keys(identity['assignment'], spec, ENDPOINTS); keys = [tuple(r[k] for k in KEYS) for r in records]
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
                for key in sorted(new_keys([pid], {'queries': [q]}, ENDPOINTS)):
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
    if ref!=frozen['reference'] or verify_images(output)!=frozen['images']:raise ValueError('Frozen evaluation changed')
    rows=p.population()[0];donors,_=p.donors_and_groups();queries={q['id']:q for q in spec['queries']};new=[]
    termination=read(PREVIOUS/'termination-0.json')
    for shard in range(2):
        path=output/f'readout-{shard}.jsonl';ident=output/f'identity-{shard}.json';records=read_lines(path);identity=read(ident)
        if read(output/f'finished-{shard}.json')!=dict(rows=len(records),identity_sha256=sha(ident),readout_sha256=sha(path)) or identity!=dict(inputs_sha256=sha(output/'inputs.json'),source_sha256=sha(Path(__file__)),protocol_sha256=sha(p.PROTOCOL),assignment=[r['base_pair_id'] for r in rows[shard::2]]):raise ValueError('Readout receipt changed')
        index_rows(records,new_keys(identity['assignment'],spec,ENDPOINTS))
        for row in records:validate_new_row(row,ref['targets'],frozen['images']['assets'],donors,queries,'pilot')
        if read(output/f'termination-{shard}.json')!=termination:raise ValueError('Reader termination changed')
        new.extend(records)
    for row in rows:
        for q in spec['queries']:load_target(ref['targets'][f"{row['base_pair_id']}|{q['id']}"],ref['reader'],row,q['query'],termination['eos'],termination['pad'])
    result=summarize(old,new,spec);save_once(output/'comparison.json',result);return result


def rollout_jobs(args):
    return [dict(name='rollout-'+arm,gpu=i,command=[sys.executable,str(Path(__file__)),'rollout','--arm',arm,
        '--output',str(args.output),'--base',str(args.base),'--reader',str(args.reader),'--official-source',str(args.official_source)]) for i,arm in enumerate(a.ARMS)]


def main(args):
    if args.output.resolve()!=RUN/'context-layout-aug-readout-v1':raise ValueError('Wrong isolated output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty execution tree')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json',dict(commit=commit,plan_sha256=sha(a.PLAN),protocol_sha256=sha(p.PROTOCOL),
            source=str(SOURCE),new_pngs=384,new_rows=9216,combined_rows=16128,gpu_hours_cap=2.25,iteration_cap=3,campaign_cap=16,
            base=str(args.base),reader=str(args.reader),official_source=str(args.official_source),styles=list(STYLES)))
        write_json(args.output/'status.json',dict(status='source_audit',time=time.time(),commit=commit))
        _,ref=previous_inputs();save_once(args.output/'reference.json',ref)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [x for x in RUN.glob('*/active-owner/owner.json') if x!=claim/'owner.json']:raise ValueError('Another owner')
        left,previous,_=remaining_seconds(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(jobs):
            if time.monotonic()+30>=deadline:raise ValueError('Insufficient stage time')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:f.result()
        write_json(args.output/'status.json',dict(status='rollout',time=time.time(),commit=commit))
        group(rollout_jobs(args));save_once(args.output/'inputs.json',dict(reference=ref,images=verify_images(args.output)))
        write_json(args.output/'status.json',dict(status='readout',time=time.time(),commit=commit))
        group([dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,str(Path(__file__)),'evaluate','--output',str(args.output),'--reader',str(args.reader),'--shard',str(i)]) for i in range(2)])
        result=report(args.output);cost=accrued_seconds(args.output);train=accrued_seconds(SOURCE)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),rows=result['combined_rows'],gpu_hours=cost/3600,iteration_gpu_hours=(train+cost)/3600,campaign_gpu_hours=(previous+cost)/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','rollout','evaluate'))
    for name in ('output','base','reader','official-source'):parser.add_argument('--'+name,type=Path,required=name in ('output','reader'))
    parser.add_argument('--arm',choices=a.ARMS);parser.add_argument('--shard',type=int,choices=(0,1));args=parser.parse_args()
    {'run':main,'rollout':rollout,'evaluate':evaluate}[args.phase](args)
