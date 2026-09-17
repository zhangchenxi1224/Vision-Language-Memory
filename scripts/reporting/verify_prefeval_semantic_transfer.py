"""Independent coverage, trace and raw-output reconstruction for the paired trial."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments.prefeval_semantic_transfer import DATA,REPORT,application_view,training_jobs
from scripts.probes.prefeval_rgb_endpoint_diagnostic import load_json,file_sha,score_generation
from scripts.reporting.verify_prefeval_rgb_endpoint_diagnostic import exact_index
from vision_memory.prefeval.rgb_protocol import digest,recovery_summary
from vision_memory.repro import canonical_tensor_sha256 as tensor_sha

def raw_score(row,q,mcq=False):
    g=row['generation'];assert len(g['generated_token_ids'])==g['generated_token_count']
    assert any(x in g['eos_token_ids'] for x in g['generated_token_ids'])==g['eos_reached']
    s=score_generation(g,q,mcq);assert s==row['score'];return s['strict_correct']

def feasibility(output):
    reg=load_json(DATA/'registration.json');p=load_json(DATA/'training-payload.json');assert digest(p)==reg['training_digest']
    expected={(sid,condition,application_view(c)['id']):application_view(c) for sid,t in p['targets'].items()
              for c in t['applications'] for condition in ('text','blank')}
    rows=[];seconds=0
    for i in range(4):
        folder=output/'feasibility'/f'shard-{i}';done=load_json(folder/'complete.json')
        assert done['optimizer_updates']==0 and file_sha(folder/'blank.png')==done['blank_sha']
        local=[json.loads(s) for s in (folder/'reads.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(local)==done['reads'];seconds+=done['seconds'];rows.extend(local)
    ix=exact_index(rows,lambda r:(r['target'],r['condition'],r['query']['id']),expected)
    counts=defaultdict(lambda:[0,0]);failures=[]
    for key,row in ix.items():
        q=expected[key];assert row['query']==q;ok=raw_score(row,q);condition=key[1]
        counts[condition][0]+=ok;counts[condition][1]+=1
        if not ok:failures.append(dict(target=key[0],condition=condition,query_id=q['id'],expected=q['target'],raw=row['generation']['raw']))
    assert len(rows)==672 and counts['text'][1]==counts['blank'][1]==336
    result=dict(registration_digest=digest(reg),text=counts['text'],blank=counts['blank'],passed=counts['text'][0]>=303,
        failures=failures,seconds=seconds,optimizer_updates=0,reads=672)
    (output/'feasibility-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2));return result

def final(output,source):
    reg=load_json(DATA/'registration.json');p=load_json(DATA/'training-payload.json');e=load_json(DATA/'evaluation-payload.json')
    assert digest(p)==reg['training_digest'] and digest(e)==reg['evaluation_digest']
    arm_receipts={};updates=forwards=0;cost=defaultdict(float);tokens=defaultdict(int)
    for sid,t in p['targets'].items():
        for name,h in t['initial']['artifacts'].items():assert file_sha(source/sid/name)==h
        paired=[]
        for arm in ('A','B'):
            folder=output/'training'/arm/sid;done=load_json(folder/'complete.json');arm_receipts[(arm,sid)]=done
            assert done['inherited_updates']==256 and done['additional_updates']==256
            for name,h in done['artifacts'].items():assert file_sha(folder/name)==h
            z=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True);paired.append(z)
            assert tensor_sha(z)==t['initial']['latent_tensor_sha']==done['initial_tensor_sha']
            endpoint=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            assert torch.isfinite(endpoint).all() and tensor_sha(endpoint)==done['final_tensor_sha']
            trace=[json.loads(s) for s in (folder/'optimization.jsonl').read_text().splitlines()]
            assert [r['step'] for r in trace]==list(range(256))
            n=ntokens=0
            for draw,row in zip(p['schedule'],trace):
                jobs=training_jobs(t,draw,arm);assert len(row['losses'])==len(jobs)
                for loss,(q,w) in zip(row['losses'],jobs):
                    assert loss['query_id']==q['id'] and loss['query_digest']==digest(q) and loss['weight']==w
                    assert loss['scope']==q['scope'] and loss['kind']==q['kind'];n+=1;ntokens+=loss['processed_input_tokens']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            assert len(opt['state'])==1 and int(next(iter(opt['state'].values()))['step'])==256
            assert n==len(t['state'])*256==done['slot_forwards']
            assert ntokens==done['processed_input_tokens_including_template_vision_and_target']
            updates+=256;forwards+=n;tokens[arm]+=ntokens;cost[arm]+=done['seconds']
        assert torch.equal(*paired)
    assert updates==20480 and forwards==45056
    expected={};rows=[]
    for sid,t in e['targets'].items():
        for condition in ('A','B','parent','text','blank'):
            panels=(['recovery_training','qualification','application_training','mcq','application_reserved'] if condition in ('A','B')
                    else ['application_training','application_reserved'] if condition=='parent' else ['application_reserved'])
            for panel in panels:
                for q in t[panel]:expected[(sid,condition,panel,q['id'])]=q
    for i in range(4):
        folder=output/'evaluation'/f'shard-{i}';done=load_json(folder/'complete.json')
        local=[json.loads(s) for s in (folder/'reads.jsonl').read_text(encoding='utf-8').splitlines()];assert len(local)==done['reads'];rows.extend(local)
    ix=exact_index(rows,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected)
    assert len(rows)==3408
    counts=defaultdict(lambda:[0,0]);groupcounts=defaultdict(lambda:[0,0]);complete={};read_correct={}
    for key,r in ix.items():
        sid,condition,panel,_=key;q=expected[key];assert r['query']==q;ok=raw_score(r,q,panel=='mcq');read_correct[key]=ok
        t=e['targets'][sid];capacity=len(t['state'])
        if condition in ('A','B'):assert r['png_sha']==arm_receipts[(condition,sid)]['artifacts']['memory.png']
        elif condition=='parent':assert r['png_sha']==t['parent']['artifacts']['memory.png']
        for suffix in ('all','K1' if capacity==1 else 'multi','changed' if q['scope']==t['changed_scope'] else 'untouched'):
            k=f'{condition}/{panel}/{suffix}';counts[k][0]+=ok;counts[k][1]+=1
            if panel=='qualification':
                k=f'{condition}/qualification_{q["kind"]}/{suffix}';counts[k][0]+=ok;counts[k][1]+=1
        if panel in ('mcq','application_training','application_reserved'):
            value=t['state'][q['scope']];assert value is not None
            sourceq=next(x for x in t['mcq'] if x['scope']==q['scope']);gid=sourceq['semantic_group']
            k=f'{condition}/{panel}/{gid}';groupcounts[k][0]+=ok;groupcounts[k][1]+=1
    for arm in ('A','B'):
        bycapacity=defaultdict(lambda:[0,0]);aux=0;states={}
        for sid,t in e['targets'].items():
            reads=[ix[(sid,arm,'qualification',q['id'])] for q in t['qualification']]
            translated=[dict(query=r['query'],score=r['score'],png_sha256=r['png_sha']) for r in reads]
            summary=recovery_summary(t['state'],t['qualification'],translated,arm_receipts[(arm,sid)]['artifacts']['memory.png'])
            k='K'+str(len(t['state']));bycapacity[k][0]+=summary['recovery_complete'];bycapacity[k][1]+=1
            states[sid]=summary;aux+=summary['all_registered_checks_pass']
        complete[arm]=dict(capacity=dict(bycapacity),states=states,all_aux_conjunction_states=aux)
        slots=defaultdict(lambda:[0,0]);clear_states=[]
        for sid,t in e['targets'].items():
            all_slots={}
            for scope,value in t['state'].items():
                queries=[q for q in t['qualification'] if q['scope']==scope and q['kind']=='recovery']
                assert len(queries)==2
                ok=all(read_correct[(sid,arm,'qualification',q['id'])] for q in queries);all_slots[scope]=ok
                for label in ('all','active' if value is not None else 'cleared','changed' if scope==t['changed_scope'] else 'untouched'):
                    slots[label][0]+=ok;slots[label][1]+=1
            if any(v is None for v in t['state'].values()):
                clear_states.append(dict(target=sid,slots=all_slots,selective_clear_complete=all(all_slots.values())))
        manifest=load_json(REPORT/'registered/manifest.json');assert digest(manifest)==reg['manifest_digest']
        chains=[]
        for episode in manifest['episodes']:
            if episode['panel']!='train-k4' or not all(x['target_state_id'] in states for x in episode['transitions']):continue
            steps=[dict(ordinal=x['ordinal'],operation=x['operation'],target=x['target_state_id'],
                        recovery_complete=states[x['target_state_id']]['recovery_complete']) for x in episode['transitions']]
            chains.append(dict(episode=episode['id'],steps=steps,complete=all(x['recovery_complete'] for x in steps)))
        assert len(chains)==4
        complete[arm].update(complete_slots=dict(slots),selective_clear_states=clear_states,
                            offline_teacher_chains=chains,chain_note='Independent endpoint teachers, not recurrent Writer rollouts')
    macro={}
    for condition in ('A','B','parent','text','blank'):
        for panel in ('mcq','application_training','application_reserved'):
            vals=[a/b for k,(a,b) in groupcounts.items() if k.startswith(f'{condition}/{panel}/')]
            if vals:macro[f'{condition}/{panel}']=sum(vals)/len(vals)
    paired=defaultdict(Counter)
    for sid,t in e['targets'].items():
        for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
            for q in t[panel]:
                a=read_correct[(sid,'A',panel,q['id'])];b=read_correct[(sid,'B',panel,q['id'])]
                paired[panel][f'{int(a)}->{int(b)}']+=1
    contrasts=[]
    for c in e['overwrite_contrasts']:
        for condition in ('A','B','parent','text','blank'):
            for panel in ('application_training','application_reserved'):
                if condition in ('text','blank') and panel!='application_reserved':continue
                before=e['targets'][c['before_state']];after=e['targets'][c['after_state']]
                for qa in [q for q in before[panel] if q['scope']==c['scope']]:
                    qb=next(q for q in after[panel] if q['case_id']==qa['case_id'] and q['scope']==qa['scope'])
                    assert qa['query']==qb['query'] and qa['target']!=qb['target']
                    a=read_correct[(c['before_state'],condition,panel,qa['id'])];b=read_correct[(c['after_state'],condition,panel,qb['id'])]
                    contrasts.append(dict(contrast=c,condition=condition,panel=panel,case_id=qa['case_id'],before_correct=a,after_correct=b,both_correct=a and b))
    criteria=dict(recovery=all(complete['B']['capacity'][k][0]>=threshold[0] for k,threshold in reg['progression']['recovery'].items()),
        mcq=counts['B/mcq/all'][0]>=68,reserved=counts['B/application_reserved/all'][0]>=135,
        mcq_macro_gain=macro['B/mcq']-macro['A/mcq']>=.10,
        reserved_macro_gain=macro['B/application_reserved']-macro['A/application_reserved']>=.10)
    result=dict(registration_digest=digest(reg),updates=updates,slot_forwards=forwards,final_reads=len(rows),counts=dict(counts),
        complete=complete,semantic_group_counts=dict(groupcounts),macro=macro,paired={k:dict(v) for k,v in paired.items()},
        overwrite_contrasts=contrasts,progression_criteria=criteria,progression_passed=all(criteria.values()),
        optimization_seconds=dict(cost),processed_input_tokens=dict(tokens),writer_updates=0)
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('complete','semantic_group_counts','overwrite_contrasts')},indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['feasibility','final'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path)
    a=ap.parse_args();feasibility(a.output) if a.mode=='feasibility' else final(a.output,a.source)
