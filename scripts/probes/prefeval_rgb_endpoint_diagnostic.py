"""Fixed endpoint factorial diagnosis. No training or artifact replacement."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.eval.prefeval_rgb import load_reader,read_png,load_overlay,append,mcq_score
from scripts.experiments.build_prefeval_rgb_teachers import file_sha,validate_vae
from vision_memory.prefeval.rgb_protocol import digest,text_prefix_v2
from vision_memory.repro import canonical_tensor_sha256 as tensor_sha,configure_strict_cuda_determinism
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer,score_short_answer
from vision_memory.reader.open_eos import assistant_termination_contract,qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT

REPORT=ROOT/'reports/prefeval-rgb-20260917'
CONDITIONS=('float_cpu','png_cpu','float_cuda','png_cuda')

def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def recovery_queries(overlay,sid):
    return [dict(q,partition=partition,query_id=f'{partition}:{q["form"]}:{q["scope"]}')
        for partition,key in [('training','training'),('heldout','qualification')]
        for q in overlay['teachers'][sid][key] if q['kind']=='recovery']

def application_queries(m,o,sid):
    state=m['targets'][sid]['state'];result=[]
    for scope,value in sorted(state.items()):
        if value is None:continue
        candidates=[]
        for j in o['reference_jobs']:
            if j['kind']!='official_mcq' or j['split']!='train' or j['state']!={scope:value}:continue
            gid=j['groups'][0];rid=m['groups'][gid]['representative'];r=m['records'][rid]
            assert r['topic']==scope and r['preference']==value and gid in m['pilot_train']
            candidates.append((rid,j,r))
        if not candidates:raise ValueError(f'No current-value MCQ lineage: {sid}/{scope}')
        candidates.sort(key=lambda x:x[0]);rid,j,r=candidates[0]
        result.append(dict(scope=scope,record=rid,semantic_group=r['semantic_group'],value=value,
            query=j['query'],target_index=r['target_index'],question=r['question'],options=r['options'],
            option_permutation=r['option_permutation'],candidates=[c[0] for c in candidates],
            text_prefix=text_prefix_v2(state),source_job=j['id']))
    return result

def register(source):
    m=load_json(REPORT/'registered/manifest.json');o=load_overlay(REPORT/'reader-format-v2.json',m)
    targets={}
    for sid in sorted(m['sentinel_targets']):
        old=load_json(source/sid/'result.json')
        for name,h in old['artifacts'].items():assert file_sha(source/sid/name)==h
        z=torch.load(source/sid/'latent.pt',map_location='cpu',weights_only=True)
        targets[sid]=dict(artifacts=old['artifacts'],result_sha=file_sha(source/sid/'result.json'),
            latent_tensor_sha=tensor_sha(z),latent_stride=list(z.stride()),predecessor=old['predecessor'],
            state=m['targets'][sid]['state'],changed_scope=m['targets'][sid]['changed_scope'],
            recovery=recovery_queries(o,sid),mcq=application_queries(m,o,sid))
    assert len(targets)==40 and sum(len(t['recovery']) for t in targets.values())==440
    assert sum(len(t['mcq']) for t in targets.values())==84
    p=dict(schema='prefeval.endpoint-diagnostic.v1',plan='prefeval-rgb-endpoint-factorial-diagnosis-04',
        endpoint_code='191d24ac01b936acd7c98ac44d2dc5ab2c26eada',manifest_digest=digest(m),
        overlay_digest=digest(o),overlay_file_sha=file_sha(REPORT/'reader-format-v2.json'),
        policy_digest=digest(load_json(REPORT/'visual-recovery-allocation-v1.json')),
        conditions=list(CONDITIONS),max_new_tokens=128,targets=targets,
        budgets=dict(recovery_generations=1760,recovery_ce=1760,mcq_generations=252,
                     optimizer_updates=0,writer_updates=0,conditional_divergence_cells=8),
        query_holdout_meaning='excluded from teacher gradients; already observed diagnostics, not fresh research holdouts',
        layout='Match all four RGB inputs to archived PNG strides; verify archived preprocessing anchor unchanged',
        mcq_tie_break='lexicographically first representative record among exact current-value pilot-train materialized jobs')
    path=REPORT/'endpoint-diagnostic-v1.json'
    if path.exists():assert load_json(path)==p
    else:path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(digest=digest(p),budgets=p['budgets'])))

def describe(x):
    return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device),sha=tensor_sha(x))

def difference(a,b):
    d=(a.detach().float().cpu()-b.detach().float().cpu()).abs()
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),different=int(torch.count_nonzero(d)))

def matched_images(decoded,png,device):
    if not torch.equal(decoded.detach().cpu().mul(255).round().clamp(0,255).byte(),png.mul(255).round().byte()):
        raise ValueError('Endpoint reconstruction differs from archived PNG pixels; archive untouched')
    result={}
    for condition in CONDITIONS:
        values=decoded if condition.startswith('float') else png
        destination=device if condition.endswith('cuda') else 'cpu'
        x=torch.empty_strided(png.shape,png.stride(),device=destination,dtype=torch.float32)
        x.copy_(values);result[condition]=x
    for representation in ('float','png'):
        assert torch.equal(result[representation+'_cpu'],result[representation+'_cuda'].cpu())
    assert len({tuple(x.stride()) for x in result.values()})==1
    return result

class CaptureProcessor:
    """Observe existing processor calls; forward every argument unchanged."""
    def __init__(self,processor):self.processor=processor;self.last=None
    def __getattr__(self,name):return getattr(self.processor,name)
    def __call__(self,**kwargs):
        batch=self.processor(**kwargs)
        self.last={k:v.detach().clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        self.resized=kwargs['images'][0].detach().clone()
        return batch

def compare_prefix(generation_batch,ce_batch,n):
    reports={}
    for key,value in generation_batch.items():
        other=ce_batch[key]
        if key in ('input_ids','attention_mask','mm_token_type_ids'):other=other[:,:-n]
        if not torch.equal(value.cpu(),other.cpu()):raise ValueError(f'CE/generation prefix mismatch: {key}')
        reports[key]=describe(value)
    return reports

def score_generation(generated,q,mcq=False):
    s=mcq_score(generated['raw'],q['target_index']) if mcq else score_short_answer(generated['raw'],q['target'])
    s['strict_correct']=bool(s['strict_correct'] and generated['eos_reached']);return s

def generate(reader,processor,image,query,device):
    # This boundary intentionally has no target/state/answer argument.
    return generate_short_answer(model=reader,processor=processor,image=image,query=query,device=device,
        max_new_tokens=128,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)

def ce_details(ce,generated):
    logits=ce.target_logits[0].float();gold=ce.target_ids[0]
    values=logits.gather(-1,gold[:,None]).squeeze(-1);nll=torch.logsumexp(logits,-1)-values
    top=logits.topk(2,dim=-1);alt=torch.where(top.indices[:,0]==gold,top.values[:,1],top.values[:,0])
    ids=gold.tolist();actual=generated['generated_token_ids']
    first=next((i for i in range(max(len(ids),len(actual))) if i>=len(ids) or i>=len(actual) or ids[i]!=actual[i]),None)
    return dict(answer_ce=float(ce.answer_loss),eos_ce=float(ce.eos_loss),answer_token_count=ce.answer_token_count,
        gold_token_ids=ids,per_token_nll=nll.tolist(),gold_minus_alternative_margin=(values-alt).tolist(),
        gold_argmax_correct=logits.argmax(-1).eq(gold).tolist(),first_divergent_token=first,
        generation_token_exact=actual==ids)

def main(args):
    if args.register:return register(args.source)
    configure_strict_cuda_determinism(0)
    p=load_json(args.registration);m=load_json(REPORT/'registered/manifest.json')
    o=load_overlay(REPORT/'reader-format-v2.json',m)
    assert p['manifest_digest']==digest(m) and p['overlay_digest']==digest(o)
    assert p['policy_digest']==digest(load_json(REPORT/'visual-recovery-allocation-v1.json'))
    from scripts.experiments.refine_historical_writer_targets import snapshot_bindings,M
    assert args.base.resolve()==M/'DreamLite-base-a9a0f15-20260907' and args.reader.resolve()==M/'Qwen3-VL-4B-Instruct'
    snapshots=snapshot_bindings()
    from diffusers import AutoencoderTiny
    vae=AutoencoderTiny.from_pretrained(str(args.base),subfolder='vae',local_files_only=True,torch_dtype=torch.float32).to(args.device)
    vae.eval().requires_grad_(False);validate_vae(vae)
    original_processor,reader=load_reader(args.reader,args.device);processor=CaptureProcessor(original_processor)
    termination=assistant_termination_contract(reader,processor)
    versions=[(v,int(v._version)) for model in (vae,reader) for v in model.parameters()]
    assignment=sorted(p['targets'])[args.shard::args.shards]
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(registration_digest=digest(p),snapshots=snapshots,assignment=assignment,
        code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),optimizer_updates=0)
    (args.output/f'identity-{args.shard}.json').write_text(json.dumps(identity,indent=2))
    started=time.monotonic()
    with torch.no_grad():
        for sid in assignment:
            t=p['targets'][sid];source=args.source/sid;out=args.output/sid
            if out.exists():raise ValueError('Diagnostic output already exists; no silent replay')
            out.mkdir();old=load_json(source/'result.json')
            assert file_sha(source/'result.json')==t['result_sha']
            for name,h in t['artifacts'].items():assert file_sha(source/name)==h
            z=torch.load(source/'latent.pt',map_location=args.device,weights_only=True)
            assert not z.requires_grad and tensor_sha(z)==t['latent_tensor_sha'] and list(z.stride())==t['latent_stride']
            decoded=decode_model_latents_unit_interval(vae,z,clamp=True)[0]
            png=read_png(source/'memory.png');images=matched_images(decoded,png,args.device)
            image_meta=dict(endpoint=describe(z),decoded_native=describe(decoded),archived_png=describe(png),
                conditions={k:describe(v) for k,v in images.items()},quantization=difference(images['float_cpu'],images['png_cpu']))
            processed={};rows=[]
            archived={(r['query']['scope'],r['query']['form']):r['generation'] for r in old['rows'] if r['query']['kind']=='recovery'}
            for condition,image in images.items():
                for q in t['recovery']:
                    generated=generate(reader,processor,image,q['query'],args.device)
                    g_batch=processor.last
                    if condition not in processed:
                        processed[condition]=dict(resized=processor.resized.cpu(),pixel_values=g_batch['pixel_values'].cpu())
                        torch.save(processed[condition],out/f'{condition}-processor.pt')
                    ce=qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=q['query'],target=q['target'],
                        device=args.device,termination=termination,lambda_eos=1.,require_image_grad=False,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    prefixes=compare_prefix(g_batch,processor.last,len(ce.target_ids[0]))
                    anchor=None
                    if condition=='png_cpu' and q['partition']=='heldout':
                        a=archived[(q['scope'],q['form'])]
                        anchor=dict(tokens_equal=generated['generated_token_ids']==a['generated_token_ids'],
                                    prefix_equal=generated['input_token_ids']==a['input_token_ids'],raw_equal=generated['raw']==a['raw'])
                    row=dict(target=sid,capacity=len(t['state']),condition=condition,query=q,generation=generated,
                        score=score_generation(generated,q),ce=ce_details(ce,generated),prefixes=prefixes,archive_replay=anchor,
                        endpoint_sha=t['artifacts']['latent.pt'],png_sha=t['artifacts']['memory.png'])
                    append(out/'recovery.jsonl',row);rows.append(row)
                    del ce,g_batch
                print(json.dumps(dict(target=sid,condition=condition,recovery_rows=len(rows))),flush=True)
            pairs=[('float_cpu','png_cpu'),('float_cuda','png_cuda'),('float_cpu','float_cuda'),('png_cpu','png_cuda')]
            image_meta['paired_differences']={a+'/'+b:{stage:difference(processed[a][stage],processed[b][stage])
                for stage in ('resized','pixel_values')} for a,b in pairs}
            image_meta['processor_artifacts']={f'{c}-processor.pt':file_sha(out/f'{c}-processor.pt') for c in CONDITIONS}
            (out/'images.json').write_text(json.dumps(image_meta,indent=2))
            blank=torch.empty_strided(png.shape,png.stride(),dtype=torch.float32).fill_(128/255)
            for q in t['mcq']:
                for condition in ('png','text','blank'):
                    image=png if condition=='png' else blank
                    query=q['text_prefix']+q['query'] if condition=='text' else q['query']
                    generated=generate(reader,processor,image,query,args.device)
                    append(out/'mcq.jsonl',dict(target=sid,capacity=len(t['state']),condition=condition,query=q,
                        generation=generated,score=score_generation(generated,q,True),png_sha=t['artifacts']['memory.png']))
            assert tensor_sha(z)==t['latent_tensor_sha'] and not z.requires_grad and z.grad is None
            for name,h in t['artifacts'].items():assert file_sha(source/name)==h
            for param,version in versions:
                assert int(param._version)==version and not param.requires_grad and param.grad is None
            receipt=dict(target=sid,recovery=len(rows),mcq=3*len(t['mcq']),optimizer_updates=0,
                replay_mismatches=sum(r['archive_replay'] is not None and not all(r['archive_replay'].values()) for r in rows),
                artifacts={f:file_sha(out/f) for f in ('images.json','recovery.jsonl','mcq.jsonl') if (out/f).exists()})
            (out/'complete.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    (args.output/f'complete-{args.shard}.json').write_text(json.dumps(dict(targets=assignment,seconds=time.monotonic()-started)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true')
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--registration',type=Path,default=REPORT/'endpoint-diagnostic-v1.json')
    ap.add_argument('--output',type=Path);ap.add_argument('--base',type=Path);ap.add_argument('--reader',type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0')
    main(ap.parse_args())
