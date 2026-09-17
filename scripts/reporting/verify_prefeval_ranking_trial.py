"""Plan07 verification. Legacy Plan05/06 verifiers remain unchanged."""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score
from scripts.reporting.verify_prefeval_rgb_endpoint_diagnostic import exact_index
from scripts.experiments.prefeval_semantic_transfer import DATA,REPORT,trial_load,training_jobs,ranking_candidates
from scripts.probes.prefeval_rgb_endpoint_diagnostic import load_json,file_sha
from vision_memory.prefeval.rgb_protocol import digest,recovery_summary
from vision_memory.repro import canonical_tensor_sha256 as tensor_sha

def jsonlines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]

def trial(output,source):
    reg,p=trial_load();e=load_json(DATA/'evaluation-payload.json')
    code=(output/'trial-code-commit.txt').read_text().strip()
    assert (output/'instance.txt').read_text().strip()==reg['instance']
    forwards=Counter();train_tokens=Counter();identities=[]
    for i in range(4):
        ident=load_json(output/f'train-identity-{i}.json');identities.append(digest(ident['snapshots']))
        assert ident['code']==code and ident['registration_digest']==digest(reg)
        assert ident['assignment']==sorted(p['targets'])[i::4] and ident['gate_sha']==reg['calibration_verified_sha']
        assert load_json(output/f'train-complete-{i}.json')['targets']==ident['assignment']
    for sid,t in p['targets'].items():
        for arm in ('A','B'):
            folder=output/'training'/arm/sid;done=load_json(folder/'complete.json')
            assert done['protocol']=='ranking-learning-trial-v1' and done['gate_sha']==reg['calibration_verified_sha']
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert ck['next_step']==128 and ck['registration_digest']==digest(reg)
            assert tensor_sha(ck['latent'])==done['final_tensor_sha']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            assert ck['optimizer']['param_groups']==opt['param_groups']
            for key,state in opt['state'].items():
                for field,value in state.items():assert torch.equal(value,ck['optimizer']['state'][key][field])
            n=tok=0
            for draw,row in zip(p['schedule'][:128],jsonlines(folder/'optimization.jsonl')):
                for loss,(q,w) in zip(row['losses'],training_jobs(t,draw,arm)):
                    application=q['kind']=='application';calls=4 if application else 1
                    assert loss['reader_forwards']==calls;n+=calls;tok+=loss['processed_input_tokens']
                    assert math.isfinite(loss['slot_loss']) and loss['processed_input_tokens']>0
                    if application:
                        case=next(c for c in t['applications'] if c['scope']==q['scope'] and c['scenario']==q['scenario'])
                        _,choices,gold=ranking_candidates(case,q['rotation'])
                        assert loss['choices']==choices and loss['gold_index']==gold and loss['loss_family']=='listwise_action_ranking'
                        scores=torch.tensor(loss['candidate_scores']);expected=float(torch.logsumexp(scores,0)-scores[gold])
                        assert math.isclose(expected,loss['slot_loss'],rel_tol=2e-5,abs_tol=2e-5)
                        assert len(loss['candidate_token_counts'])==4 and min(loss['candidate_token_counts'])>0
                    else:
                        assert loss['loss_family']=='answer_plus_eos'
                        assert math.isclose(loss['slot_loss'],loss['answer_ce']+loss['eos_ce'],rel_tol=2e-5,abs_tol=2e-5)
            assert n==done['reader_forwards'];forwards[arm]+=n;train_tokens[arm]+=tok
    assert dict(forwards)=={'A':11264,'B':27392}
    # The independent legacy-style reconstruction checks all 3408 generations,
    # unchanged exact scoring, whole states, parent comparisons, traces and hashes.
    generation=trial_generation(output,source)
    expected={};scenarios={}
    for filename in ('training-scenarios.json','reserved-scenarios.json'):
        for cases in load_json(DATA/filename).values():
            for c in cases:scenarios[(c['id'],c['scope'],c['value_id'])]=c
    for sid,t in e['targets'].items():
        for condition in ('A','B','parent','text','blank'):
            for panel in (('application_training','application_reserved') if condition in ('A','B','parent') else ('application_reserved',)):
                for q in t[panel]:
                    case=scenarios[(q['case_id'],q['scope'],q['value_id'])]
                    rq,choices,gold=ranking_candidates(case,q['rotation']);assert rq==q
                    expected[(sid,condition,panel,q['id'])]=(q,choices,gold)
                    if condition in ('A','B','parent') and panel=='application_training' and q['value_id'] in reg['contrast_values']:
                        for rotation in (1,2,3):
                            rq,rc,rg=ranking_candidates(case,rotation)
                            expected[(sid,condition,'application_training_rotations',rq['id'])]=(rq,rc,rg)
    assert len(expected)==2712
    rows=[];actual=tokens=0;seconds=0;blank_hashes=set()
    for i in range(4):
        folder=output/'evaluation'/f'shard-{i}';done=load_json(folder/'complete.json');ident=load_json(folder/'identity.json')
        assert file_sha(folder/'reads.jsonl')==done['reads_sha'] and file_sha(folder/'ranking.jsonl')==done['ranking_sha']
        assert file_sha(folder/'blank.png')==done['blank_sha']
        assert ident['frozen_endpoints']=={f'{arm}/{sid}':file_sha(output/'training'/arm/sid/'complete.json') for sid in p['targets'] for arm in ('A','B')}
        assert ident['code']==code and ident['registration_digest']==digest(reg)
        assert ident['assignment']==sorted(p['targets'])[i::4];identities.append(digest(ident['snapshots']))
        frame=ident['prompt_frame'];assert frame.count('__REGISTERED_QUERY__')==1
        blank=file_sha(folder/'blank.png');blank_hashes.add(blank);cache={};local_calls=local_tokens=0
        for r in jsonlines(folder/'reads.jsonl'):
            t=e['targets'][r['target']];query=t['text_prefix']+r['query']['query'] if r['condition']=='text' else r['query']['query']
            proof=r['processor_input'];assert r['reader_query']==query and proof['text']==frame.replace('__REGISTERED_QUERY__',query)
            assert proof['input_ids_digest']==digest([r['generation']['input_token_ids']])
            assert proof['input_tokens']==len(r['generation']['input_token_ids']);local_tokens+=proof['input_tokens']
        local=jsonlines(folder/'ranking.jsonl');assert len(local)==done['ranking_decisions']
        for r in local:
            sid=r['target'];assert sid in ident['assignment'];condition=r['condition']
            q,choices,gold=expected[(sid,condition,r['panel'],r['query']['id'])]
            t=e['targets'][sid];query=t['text_prefix']+q['query'] if condition=='text' else q['query']
            assert r['query']==q and r['choices']==choices and r['gold_index']==gold and r['reader_query']==query
            png=(load_json(output/'training'/condition/sid/'complete.json')['artifacts']['memory.png'] if condition in ('A','B')
                 else t['parent']['artifacts']['memory.png'] if condition=='parent' else blank)
            assert r['png_sha']==png
            key=digest(dict(query=query,choices=choices,png=png));assert key==r['input_identity']
            score=r['score'];assert score['model_prompt']==frame.replace('__REGISTERED_QUERY__',query)
            assert len(score['scores'])==len(score['candidate_token_counts'])==len(score['processor_calls'])==4
            assert all(math.isfinite(x) for x in score['scores']) and min(score['candidate_token_counts'])>0
            for call,choice in zip(score['processor_calls'],choices):assert call['text']==score['model_prompt']+choice and call['input_tokens']>0
            if key in cache:assert r['reused'] and r['actual_candidate_forwards']==0 and score==cache[key]
            else:
                assert not r['reused'] and r['actual_candidate_forwards']==4;cache[key]=score;local_calls+=4
                local_tokens+=sum(x['input_tokens'] for x in score['processor_calls'])
            margin=score['scores'][gold]-max(x for j,x in enumerate(score['scores']) if j!=gold)
            assert margin==r['gold_margin'] and (margin>0)==r['unique_correct']
        assert local_calls==done['ranking_candidate_forwards'] and local_tokens==done['processed_input_tokens']
        actual+=local_calls;tokens+=local_tokens;seconds+=done['seconds'];rows.extend(local)
    assert len(set(identities))==1 and len(blank_hashes)==1 and actual<=10848
    ix=exact_index(rows,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected)
    counts=defaultdict(lambda:[0,0]);gc=defaultdict(lambda:[0,0]);pairs=defaultdict(Counter);contrasts=[]
    for (sid,condition,panel,_),r in ix.items():
        t=e['targets'][sid];q=r['query'];ok=r['unique_correct']
        gid=next(x['semantic_group'] for x in t['mcq'] if x['scope']==q['scope'])
        for suffix in ('all','K1' if len(t['state'])==1 else 'multi','changed' if q['scope']==t['changed_scope'] else 'untouched','gold_position_'+str(r['gold_index'])):
            k=f'{condition}/{panel}/{suffix}';counts[k][0]+=ok;counts[k][1]+=1
        k=f'{condition}/{panel}/{gid}';gc[k][0]+=ok;gc[k][1]+=1
        if condition in ('A','parent'):
            for arm in (('B',) if condition=='A' else ('A','B')):
                other=ix[(sid,arm,panel,q['id'])]['unique_correct']
                pairs[f'{condition}->{arm}/{panel}'][f'{int(ok)}->{int(other)}']+=1
    macro={}
    for condition in ('A','B','parent','text','blank'):
        for panel in ('application_training','application_reserved','application_training_rotations'):
            vals=[a/b for k,(a,b) in gc.items() if k.startswith(f'{condition}/{panel}/')]
            if vals:macro[f'{condition}/{panel}']=sum(vals)/len(vals)
    # Reuse existing calibration decisions for text/blank, including rotations.
    historical={}
    for i in range(4):
        for r in jsonlines(source.parent.parent/'semantic-ranking-v1-run/calibration'/f'shard-{i}'/'scores.jsonl'):
            historical[(r['target'],r['condition'],r['query']['id'])]=r
    def ranked(sid,condition,panel,q):
        if condition in ('text','blank') and panel=='application_training':return historical[(sid,condition,q['id'])]['unique_correct']
        actual_panel='application_training_rotations' if panel=='application_training' and q['rotation'] else panel
        return ix[(sid,condition,actual_panel,q['id'])]['unique_correct']
    for c in e['overwrite_contrasts']:
        for condition in ('A','B','parent','text','blank'):
            for panel,filename in (('application_training','training-scenarios.json'),('application_reserved','reserved-scenarios.json')):
                scenario=load_json(DATA/filename)
                for ca,cb in zip(scenario[c['before']],scenario[c['after']]):
                    for rotation in (range(4) if panel=='application_training' else (0,)):
                        qa,oa,ga=ranking_candidates(ca,rotation);qb,ob,gb=ranking_candidates(cb,rotation)
                        assert qa['query']==qb['query'] and oa==ob and ga!=gb
                        before=ranked(c['before_state'],condition,panel,qa);after=ranked(c['after_state'],condition,panel,qb)
                        contrasts.append(dict(contrast=c,condition=condition,panel=panel,case_id=ca['id'],rotation=rotation,
                            before_correct=before,after_correct=after,both_correct=before and after))
    consistency=[]
    for sid,t in p['targets'].items():
        for case in t['applications']:
            if case['value_id'] not in reg['contrast_values']:continue
            for condition in ('A','B','parent','text','blank'):
                outcomes=[ranked(sid,condition,'application_training',ranking_candidates(case,r)[0]) for r in range(4)]
                consistency.append(dict(target=sid,value_id=case['value_id'],case_id=case['id'],condition=condition,
                    rotation_correct=outcomes,all_four_correct=all(outcomes)))
    reserved=[r for r in contrasts if r['condition']=='B' and r['panel']=='application_reserved']
    assert len(reserved)==8
    criteria=dict(recovery=generation['strict_generation_diagnostic_criteria']['recovery'],
        mcq=generation['counts']['B/mcq/all'][0]>=68,
        mcq_macro_gain=generation['macro']['B/mcq']-generation['macro']['A/mcq']>=.10,
        reserved_ranking=counts['B/application_reserved/all'][0]>=135,
        reserved_ranking_macro_gain=macro['B/application_reserved']-macro['A/application_reserved']>=.10,
        reserved_contrasts=sum(r['both_correct'] for r in reserved)>=6,
        every_reserved_contrast=all(any(r['both_correct'] for r in reserved if r['contrast']==c) for c in e['overwrite_contrasts']))
    result=dict(registration_digest=digest(reg),updates=10240,gradient_forwards=dict(forwards),gradient_forwards_total=sum(forwards.values()),
        training_tokens=dict(train_tokens),generation_verified_sha=file_sha(output/'generation-verified.json'),generations=3408,
        ranking_decisions=2712,actual_candidate_forwards=actual,evaluation_processed_tokens=tokens,evaluation_seconds=seconds,
        counts=dict(counts),macro=macro,group_counts=dict(gc),paired={k:dict(v) for k,v in pairs.items()},
        overwrite_contrasts=contrasts,rotation_consistency=consistency,adoption_targets=criteria,
        adoption_targets_met=all(criteria.values()),writer_updates=0,note='Exploratory independent latent teachers; no usable shared Writer claim')
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('group_counts','overwrite_contrasts','rotation_consistency')},indent=2))
    return result

def trial_generation(output,source):
    reg,p=trial_load();e=load_json(DATA/'evaluation-payload.json')
    assert digest(p)==reg['training_digest'] and digest(e)==reg['evaluation_digest']
    arm_receipts={};updates=forwards=0;cost=defaultdict(float);tokens=defaultdict(int)
    for sid,t in p['targets'].items():
        for name,h in t['initial']['artifacts'].items():assert file_sha(source/sid/name)==h
        paired=[]
        for arm in ('A','B'):
            folder=output/'training'/arm/sid;done=load_json(folder/'complete.json');arm_receipts[(arm,sid)]=done
            assert done['inherited_updates']==256 and done['additional_updates']==128
            for name,h in done['artifacts'].items():assert file_sha(folder/name)==h
            z=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True);paired.append(z)
            assert tensor_sha(z)==t['initial']['latent_tensor_sha']==done['initial_tensor_sha']
            endpoint=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            assert torch.isfinite(endpoint).all() and tensor_sha(endpoint)==done['final_tensor_sha']
            trace=[json.loads(s) for s in (folder/'optimization.jsonl').read_text().splitlines()]
            assert [r['step'] for r in trace]==list(range(128))
            n=ntokens=0
            for draw,row in zip(p['schedule'][:128],trace):
                jobs=training_jobs(t,draw,arm);assert len(row['losses'])==len(jobs)
                for loss,(q,w) in zip(row['losses'],jobs):
                    assert loss['query_id']==q['id'] and loss['query_digest']==digest(q) and loss['weight']==w
                    assert loss['scope']==q['scope'] and loss['kind']==q['kind'];n+=1;ntokens+=loss['processed_input_tokens']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            assert len(opt['state'])==1 and int(next(iter(opt['state'].values()))['step'])==128
            assert n==len(t['state'])*128==done['slot_forwards']
            assert ntokens==done['processed_input_tokens_including_template_vision_and_target']
            updates+=128;forwards+=n;tokens[arm]+=ntokens;cost[arm]+=done['seconds']
        assert torch.equal(*paired)
    assert updates==10240 and forwards==22528
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
        overwrite_contrasts=contrasts,strict_generation_diagnostic_criteria=criteria,strict_generation_diagnostic_passed=all(criteria.values()),
        optimization_seconds=dict(cost),processed_input_tokens=dict(tokens),writer_updates=0)
    (output/'generation-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('complete','semantic_group_counts','overwrite_contrasts')},indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    a=ap.parse_args();trial(a.output,a.source)
