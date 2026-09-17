"""Independent coverage, trace and raw-output reconstruction for the paired trial."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments.prefeval_semantic_transfer import DATA,REPORT,application_view,training_jobs,ranking_load,ranking_candidates
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
    (output/'feasibility-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2));return result

def ranking_calibration(output):
    reg,p=ranking_load();metadata=load_json(DATA/'authoring-source.json')
    assert digest(metadata)==reg['source_files']['authoring-source.json']
    groups={v['id']:v['source_groups'][0] for v in metadata['values']}
    assert all(len(v['source_groups'])==1 for v in metadata['values'])
    expected={}
    for sid,t in p['targets'].items():
        for c in t['applications']:
            for rotation in range(4):
                q,choices,gold=ranking_candidates(c,rotation)
                for condition in ('text','blank'):expected[(sid,condition,q['id'])]=(q,choices,gold)
    all_rows=[];frames={};actual=tokens=0;seconds=0;model_identities=[]
    for i in range(4):
        folder=output/'calibration'/f'shard-{i}';done=load_json(folder/'complete.json');ident=load_json(folder/'identity.json')
        assert ident['registration_digest']==digest(reg)
        assert ident['code']==(output/'ranking-calibration-code-commit.txt').read_text().strip()
        model_identities.append(digest(dict(snapshots=ident['snapshots'],template=ident['chat_template_digest'])))
        assert ident['assignment']==sorted(p['targets'])[i::4]
        assert done['optimizer_updates']==done['generated_answers']==0
        assert file_sha(folder/'scores.jsonl')==done['scores_sha'] and file_sha(folder/'blank.png')==done['blank_sha']
        rows=[json.loads(s) for s in (folder/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==done['decisions'];cache={};local_forwards=local_tokens=0
        frame=ident['prompt_frame'];assert frame.count('__REGISTERED_QUERY__')==1
        for r in rows:
            sid=r['target'];assert sid in ident['assignment'];q,choices,gold=expected[(sid,r['condition'],r['query']['id'])]
            query=p['targets'][sid]['text_prefix']+q['query'] if r['condition']=='text' else q['query']
            assert r['query']==q and r['choices']==choices and r['gold_index']==gold and r['reader_query']==query
            key=digest(dict(query=query,choices=choices,png=done['blank_sha']));assert key==r['input_identity']
            score=r['score'];assert score['model_prompt']==frame.replace('__REGISTERED_QUERY__',query)
            assert len(score['scores'])==len(score['candidate_token_counts'])==len(score['processor_calls'])==4
            assert all(torch.isfinite(torch.tensor(score['scores']))) and min(score['candidate_token_counts'])>0
            for call,choice in zip(score['processor_calls'],choices):
                assert call['text']==score['model_prompt']+choice and call['input_tokens']>0
            if key in cache:
                assert r['reused'] and r['actual_candidate_forwards']==0 and score==cache[key]
            else:
                assert not r['reused'] and r['actual_candidate_forwards']==4;cache[key]=score;local_forwards+=4
                local_tokens+=sum(x['input_tokens'] for x in score['processor_calls'])
            frames[sid]=frame
        assert local_forwards==done['candidate_forwards'] and local_tokens==done['processed_input_tokens']
        actual+=local_forwards;tokens+=local_tokens;seconds+=done['seconds'];all_rows.extend(rows)
    assert len(set(frames.values()))==1
    assert len(set(model_identities))==1
    ix=exact_index(all_rows,lambda r:(r['target'],r['condition'],r['query']['id']),expected);assert len(ix)==2688
    counts=defaultdict(lambda:[0,0]);gc=defaultdict(lambda:[0,0]);ties=Counter();margins={};correct={}
    for key,r in ix.items():
        sid,condition,_=key;q=r['query'];scores=r['score']['scores'];gold=r['gold_index']
        margin=scores[gold]-max(x for j,x in enumerate(scores) if j!=gold);ok=margin>0
        assert margin==r['gold_margin'] and ok==r['unique_correct'];correct[key]=ok;margins[str(key)]=margin
        ties[condition]+=margin==0
        for suffix in ('all','K'+str(len(p['targets'][sid]['state'])),'rotation'+str(q['rotation']),'value:'+q['value_id']):
            k=f'{condition}/{suffix}';counts[k][0]+=ok;counts[k][1]+=1
        k=f'{condition}/{groups[q["value_id"]]}';gc[k][0]+=ok;gc[k][1]+=1
    macro={condition:sum(a/b for k,(a,b) in gc.items() if k.startswith(condition+'/'))/len(groups) for condition in ('text','blank')}
    contrasts=[]
    for c in metadata['overwrite_contrasts']:
        bycondition={};details=[]
        before=p['targets'][c['before_state']];after=p['targets'][c['after_state']]
        for condition in ('text','blank'):
            successes=0
            for ca in [x for x in before['applications'] if x['scope']==c['scope']]:
                cb=next(x for x in after['applications'] if x['id']==ca['id'] and x['scope']==ca['scope'])
                for rotation in range(4):
                    qa,oa,ga=ranking_candidates(ca,rotation);qb,ob,gb=ranking_candidates(cb,rotation)
                    assert qa['query']==qb['query'] and oa==ob and ga!=gb
                    ka=(c['before_state'],condition,qa['id']);kb=(c['after_state'],condition,qb['id'])
                    if condition=='blank':assert ix[ka]['score']==ix[kb]['score']
                    both=correct[ka] and correct[kb];successes+=both
                    details.append(dict(condition=condition,scenario=ca['scenario'],rotation=rotation,
                                        before_correct=correct[ka],after_correct=correct[kb],both_correct=both))
            bycondition[condition]=[successes,16]
        contrasts.append(dict(contrast=c,counts=bycondition,pairs=details))
    criteria=dict(text=counts['text/all'][0]>=1210,macro_gain=macro['text']-macro['blank']>=.10,
                  contrasts=all(c['counts']['text'][0]>=12 for c in contrasts))
    result=dict(registration_digest=digest(reg),decisions=len(ix),actual_candidate_forwards=actual,
        max_candidate_forwards=10752,counts=dict(counts),semantic_group_counts=dict(gc),macro=macro,contrasts=contrasts,
        criteria=criteria,passed=all(criteria.values()),ties=dict(ties),seconds=seconds,processed_input_tokens=tokens,
        optimizer_updates=0,generated_answers=0,shard_receipts={str(i):file_sha(output/'calibration'/f'shard-{i}'/'complete.json') for i in range(4)})
    assert actual<=10752
    (output/'calibration-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('contrasts','counts','semantic_group_counts')},indent=2))
    print(json.dumps({k:v for k,v in counts.items() if k.endswith('/all')}));return result

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
    counts=defaultdict(lambda:[0,0]);groupcounts=defaultdict(lambda:[0,0]);stratified_groups=defaultdict(lambda:[0,0]);complete={};read_correct={}
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
            k=f'{condition}/{panel}/{"K1" if capacity==1 else "multi"}/{gid}'
            stratified_groups[k][0]+=ok;stratified_groups[k][1]+=1
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
            for stratum in ('K1','multi'):
                vals=[a/b for k,(a,b) in stratified_groups.items() if k.startswith(f'{condition}/{panel}/{stratum}/')]
                if vals:macro[f'{condition}/{panel}/{stratum}']=sum(vals)/len(vals)
    paired=defaultdict(Counter)
    for sid,t in e['targets'].items():
        for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
            for q in t[panel]:
                a=read_correct[(sid,'A',panel,q['id'])];b=read_correct[(sid,'B',panel,q['id'])]
                paired[panel][f'{int(a)}->{int(b)}']+=1
    parent_pairs=defaultdict(Counter);parent_counts=defaultdict(lambda:[0,0])
    for sid,t in e['targets'].items():
        old_path=source/sid/'result.json';assert file_sha(old_path)==t['parent']['result_sha']
        old=load_json(old_path)
        historical={('qualification',r['query']['query']):r for r in old['rows']}
        mcq_path=source.parent/'endpoint-diagnostic-v1'/sid/'mcq.jsonl'
        for r in [json.loads(s) for s in mcq_path.read_text(encoding='utf-8').splitlines()]:
            if r['condition']=='png':historical[('mcq',r['query']['query'])]=r
        for panel in ('qualification','mcq','application_training','application_reserved'):
            for q in t[panel]:
                if panel in ('qualification','mcq'):
                    r=historical[(panel,q['query'])];assert r.get('png_sha',r.get('png_sha256'))==t['parent']['artifacts']['memory.png']
                    for field in ('target','target_index'):
                        if field in q:assert r['query'][field]==q[field]
                    parent_ok=raw_score(r,r['query'],panel=='mcq')
                else:parent_ok=read_correct[(sid,'parent',panel,q['id'])]
                for label in ('all','K1' if len(t['state'])==1 else 'multi'):
                    k=f'{panel}/{label}';parent_counts[k][0]+=parent_ok;parent_counts[k][1]+=1
                for arm in ('A','B'):
                    ok=read_correct[(sid,arm,panel,q['id'])]
                    parent_pairs[f'{arm}/{panel}'][f'{int(parent_ok)}->{int(ok)}']+=1
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
        parent_counts=dict(parent_counts),paired_parent_to_arm={k:dict(v) for k,v in parent_pairs.items()},
        reused_parent_qualification_reads=432,reused_parent_mcq_reads=84,
        overwrite_contrasts=contrasts,progression_criteria=criteria,progression_passed=all(criteria.values()),
        optimization_seconds=dict(cost),processed_input_tokens=dict(tokens),writer_updates=0)
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('complete','semantic_group_counts','overwrite_contrasts')},indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['feasibility','final','ranking-calibration'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path)
    a=ap.parse_args()
    if a.mode=='ranking-calibration':ranking_calibration(a.output)
    elif a.mode=='feasibility':feasibility(a.output)
    else:final(a.output,a.source)
