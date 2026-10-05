"""Read-only paired functional audit of already frozen visual memories."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

ENDPOINTS = ('original/history_hard', 'original/prompt_matching',
             'diverse/history_hard/teachers', 'diverse/prompt_matching/teachers',
             'diverse/history_hard/selected', 'diverse/prompt_matching/selected')


def validate_protocol(spec):
    from scripts.experiments.prefeval_context_coverage import RECALL, NEUTRAL, VALIDATION
    queries = spec['queries']
    if len(queries) != 12 or len({q['id'] for q in queries}) != 12 or len({q['query'] for q in queries}) != 12:
        raise ValueError('Expected 12 unique diagnostic queries')
    if any(sum(q['family'] == f for q in queries) != 4 for f in ('recall','application','neutral')):
        raise ValueError('Unbalanced diagnostic query families')
    if set(q['query'] for q in queries) & (set(RECALL) | set(NEUTRAL) | {q for _,q in VALIDATION}):
        raise ValueError('Diagnostic queries overlap training or selection templates')


def expected_keys(ids, queries):
    return {(pid,q['id'],endpoint,control) for pid in ids for q in queries
            for endpoint in ENDPOINTS for control in ('memory','mismatch')} | {
                (pid,q['id'],'blank','blank') for pid in ids for q in queries}


def main(args):
    import torch
    from collections import defaultdict
    from scripts.experiments.prefeval_k1_data import load_training_records, sha
    from scripts.experiments.prefeval_multitarget_bank import select_rows
    from scripts.experiments.prefeval_prompt_matching import get_history_target, supervision_binding
    from scripts.experiments.prefeval_k1_teacher import save_json
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    from types import SimpleNamespace

    spec=json.loads(args.protocol.read_text())
    validate_protocol(spec)
    if args.shards != 2 or args.shard not in range(args.shards):
        raise ValueError('Use the frozen two-shard assignment')
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    all_rows=select_rows(load_training_records('pilot',False),args.ids_file)
    rows=all_rows[args.shard::args.shards]
    if len(all_rows)!=16:
        raise ValueError('Audit must retain all 16 registered histories')
    peers=defaultdict(list)
    for row in all_rows: peers[row['topic']].append(row['base_pair_id'])
    bank={r['base_pair_id']:r for r in all_rows}
    args.output.mkdir(parents=True,exist_ok=True)
    assets={}
    def endpoint_dir(endpoint,pid):
        parts=endpoint.split('/')
        return ((args.reference/parts[1]/'teachers') if parts[0]=='original'
                else args.pilot/parts[1]/parts[2])/pid.replace(':','_')
    for endpoint in ENDPOINTS:
        for row in all_rows:
            pid=row['base_pair_id'];folder=endpoint_dir(endpoint,pid)
            done=json.loads((folder/'complete.json').read_text())
            png=folder/'memory.png'
            if sha(png)!=done['png_sha256']:
                raise ValueError('Source PNG hash changed')
            assets[endpoint+'|'+pid]={'path':str(png),'sha256':done['png_sha256'],
                                      'complete_sha256':sha(folder/'complete.json')}
    first=json.loads((endpoint_dir('diverse/prompt_matching/teachers',rows[0]['base_pair_id'])/'complete.json').read_text())
    binding=first['binding']
    actual=supervision_binding(SimpleNamespace(supervision='prompt_matching',temperature=1.,reader=args.reader,
        teacher_max_new_tokens=512,teacher_cache=None,boundary_recovery_from=None),rows)
    for key in ('reader_path','reader_weights_sha256','reader_config_sha256','target_pipeline_sha256','reader_objective_sha256'):
        if binding[key]!=actual[key]:raise ValueError('Reader/target implementation changed: '+key)
    identity={'schema':'dreamlite.context-readout-shard.v1','protocol_sha256':sha(args.protocol),
              'source_sha256':sha(Path(__file__)),'ids_sha256':sha(args.ids_file),'shard':args.shard,
              'assignment':[r['base_pair_id'] for r in rows], 'assets':assets,
              'reader':actual,'no_training':True,'all_targets':'same history teacher exact continuations'}
    ident=args.output/f'identity-{args.shard}.json'
    if ident.exists() and json.loads(ident.read_text())!=identity:raise ValueError('Audit resume identity changed')
    save_json(ident,identity)
    dest=args.output/f'readout-{args.shard}.jsonl'
    completed={}
    if dest.exists():
        for line in dest.read_text().splitlines():
            value=json.loads(line);key=tuple(value[k] for k in ('pair_id','query_id','endpoint','control'))
            if key in completed:raise ValueError('Duplicate audit record')
            completed[key]=value
    expected=expected_keys(identity['assignment'],spec['queries'])
    if not set(completed)<=expected:raise ValueError('Unexpected audit keys')
    processor,reader=load_reader(args.reader,args.device)
    termination=assistant_termination_contract(reader,processor)
    gray=torch.full((3,1024,1024),128/255,dtype=torch.float32,device=args.device)
    with torch.no_grad():
        for row in rows:
            pid=row['base_pair_id'];same_topic=peers[row['topic']]
            donor=same_topic[(same_topic.index(pid)+1)%len(same_topic)]
            if donor==pid:raise ValueError('Mismatch donor equals source')
            for query in spec['queries']:
                target,path=get_history_target(cache_root=args.output/'targets',binding=binding,row=row,
                    query=query['query'],model=reader,processor=processor,reference=gray,device=args.device,
                    assistant_end_token_id=termination['assistant_end_token_id'])
                ids=torch.tensor([target['target_ids']]);teacher_logits=target['logits'].to(args.device)
                for endpoint,control in [('blank','blank')]+[(e,c) for e in ENDPOINTS for c in ('memory','mismatch')]:
                    key=(pid,query['id'],endpoint,control)
                    if key in completed:continue
                    source_pid=donor if control=='mismatch' else pid
                    asset=None if control=='blank' else assets[endpoint+'|'+source_pid]
                    pixels=gray if asset is None else read_png(asset['path']).to(args.device)
                    output=qwen3vl_continuation_logits(model=reader,processor=processor,image=pixels,
                        query=query['query'],target_ids=ids,device=args.device,require_image_grad=False,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                    kl=float(soft_target_kl_divergence(output.target_logits,teacher_logits))
                    if not math.isfinite(kl):raise ValueError('Nonfinite readout KL')
                    record={'pair_id':pid,'query_id':query['id'],'family':query['family'],'endpoint':endpoint,
                            'control':control,'kl':kl,'teacher_target':str(path),'teacher_logits_sha256':target['logits_sha256'],
                            'target_tokens':len(target['target_ids']),'target_ids':target['target_ids'],
                            'png':asset,'donor_pair_id':donor if control=='mismatch' else None}
                    append(dest,record);completed[key]=record
                del teacher_logits,target
            print(json.dumps({'readout_complete':pid,'shard':args.shard,'rows':len(completed)}),flush=True)
    if set(completed)!=expected:raise ValueError('Incomplete audit denominator')
    save_json(args.output/f'finished-{args.shard}.json',{'status':'completed','rows':len(completed),
        'identity_sha256':sha(ident),'readout_sha256':sha(dest),'time':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','reference','reader','ids-file','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
