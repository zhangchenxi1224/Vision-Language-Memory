"""Reconstruct Plan08 fixed-budget and same-PNG joint outcomes from raw records."""
import argparse,json,sys,math
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments import prefeval_joint_consolidation as j
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score
from scripts.reporting.verify_prefeval_rgb_endpoint_diagnostic import exact_index
s=j.s;digest=j.digest;tensor_sha=j.tensor_sha
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def add(counts,key,ok):counts[key][0]+=int(ok);counts[key][1]+=1
def verify(output,source):
    reg,p=j.load();e=s.load_json(s.DATA/'evaluation-payload.json');assert digest(e)==reg['evaluation_digest']
    for name,h in reg['parent_verified'].items():assert s.file_sha(source/name)==h
    for name,h in reg['parent_evaluation_files'].items():assert s.file_sha(source/name)==h
    code=(output/'joint-code-commit.txt').read_text().strip();snapshots=[];forward=Counter();tokens=Counter();cost=Counter();gradstats=defaultdict(Counter)
    for i in range(4):
        ident=s.load_json(output/f'train-identity-{i}.json');assert ident['code']==code and ident['registration_digest']==digest(reg)
        assert ident['actual_host']==reg['runtime']['actual'] and ident['assignment']==sorted(p['targets'])[i::4]
        assert s.load_json(output/f'train-complete-{i}.json')['targets']==ident['assignment'];snapshots.append(digest(ident['snapshots']))
    for sid,t in p['targets'].items():
        src=source/'training/B'/sid;initial=reg['endpoint_receipts'][sid]
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        for name,h in initial['artifacts'].items():assert s.file_sha(src/name)==h
        pair=[]
        for arm in ('C','J'):
            folder=output/'training'/arm/sid;done=s.load_json(folder/'complete.json')
            assert done['target']==sid and done['arm']==arm and done['inherited_updates']==[256,128] and done['additional_updates']==64
            for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
            z=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True);pair.append(z)
            assert tensor_sha(z)==initial['final_tensor_sha']==done['initial_tensor_sha']
            final=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True);assert torch.isfinite(final).all() and tensor_sha(final)==done['final_tensor_sha']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True);ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert ck['next_step']==64 and ck['registration_digest']==digest(reg) and tensor_sha(ck['latent'])==done['final_tensor_sha']
            assert opt['param_groups']==ck['optimizer']['param_groups'] and len(opt['state'])==1
            for key,state in opt['state'].items():
                assert int(state['step'])==64
                for field,v in state.items():assert torch.equal(v,ck['optimizer']['state'][key][field])
            for k,v in reg['optimizer'].items():assert opt['param_groups'][0][k]==v
            trace=rows(folder/'optimization.jsonl');assert [r['step'] for r in trace]==list(range(64));nf=nt=0
            for step,row in enumerate(trace):
                rec,app=j.jobs(t,step,arm);expected=[(q,w,None,None) for q,w in rec]+app
                assert len(row['losses'])==len(expected) and row['measurement'].startswith('pre-update')
                for loss,(q,w,choices,gold) in zip(row['losses'],expected):
                    assert loss['query']==q and loss['weight']==w and math.isfinite(loss['loss'])
                    calls=4 if choices else 1;assert loss['reader_forwards']==calls;nf+=calls;nt+=loss['processed_input_tokens']
                    if choices:
                        assert loss['choices']==choices and loss['gold_index']==gold and len(loss['candidate_token_counts'])==4
                        scores=torch.tensor(loss['scores']);value=float(torch.logsumexp(scores,0)-scores[gold])
                    else:value=loss['answer_ce']+loss['eos_ce']
                    assert math.isclose(value,loss['loss'],rel_tol=2e-5,abs_tol=2e-5)
                assert row['weighted_objective']==sum(x['weight']*x['loss'] for x in row['losses'])
                g=row['gradient'];assert all(v is None or math.isfinite(v) for v in g.values())
                assert all(g[k]>=0 for k in ('recovery_norm','weighted_application_norm','displacement_norm'))
                if step==0:assert g['displacement_norm']>0
                if g['cosine'] is not None:assert -1.00001<=g['cosine']<=1.00001
                for key in ('dot','recovery_dot_displacement','weighted_application_dot_displacement'):gradstats[arm][key+'_negative']+=g[key]<0
                gradstats[arm]['updates']+=1
            assert nf==done['reader_forwards'] and nt==done['processed_input_tokens'];forward[arm]+=nf;tokens[arm]+=nt;cost[arm]+=done['seconds']
        assert torch.equal(*pair)
    assert forward=={'C':16896,'J':38400}
    expected_gen={};expected_rank={};cases={}
    for filename in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/filename).values():
            for c in cc:cases[(c['id'],c['scope'],c['value_id'])]=c
    for sid,t in e['targets'].items():
        for arm in ('C','J'):
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for q in t[panel]:expected_gen[sid,arm,panel,q['id']]=q
            for panel in ('application_training','application_reserved'):
                for q0 in t[panel]:
                    case=cases[q0['case_id'],q0['scope'],q0['value_id']]
                    for rotation in (range(4) if panel=='application_training' and q0['value_id'] in reg['contrast_values'] else (0,)):
                        q,ch,gi=s.ranking_candidates(case,rotation);pan='application_training_rotations' if rotation else panel
                        expected_rank[sid,arm,pan,q['id']]=(q,ch,gi)
    gens=[];ranks=[];ces=[];evaltokens=evalcalls=evaltime=0
    for i in range(4):
        folder=output/'evaluation'/f'shard-{i}';ident=s.load_json(folder/'identity.json');done=s.load_json(folder/'complete.json')
        assert ident['code']==code and ident['registration_digest']==digest(reg) and ident['actual_host']==reg['runtime']['actual']
        assert ident['assignment']==sorted(p['targets'])[i::4];snapshots.append(digest(ident['snapshots']))
        assert ident['frozen_endpoints']=={f'{arm}/{sid}':s.file_sha(output/'training'/arm/sid/'complete.json') for sid in p['targets'] for arm in ('C','J')}
        for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
        gg=rows(folder/'reads.jsonl');rr=rows(folder/'ranking.jsonl');cc=rows(folder/'recovery-ce.jsonl')
        assert len(gg)==done['generations'] and len(rr)==done['rankings'] and len(cc)==done['recovery_ce_forwards']
        frame=ident['prompt_frame'];assert frame.count('__REGISTERED_QUERY__')==1;localtokens=calls=0;cache={}
        for r in gg+rr+cc:
            sid=r['target'];arm=r['condition'];assert sid in ident['assignment']
            assert r['png_sha']==s.load_json(output/'training'/arm/sid/'complete.json')['artifacts']['memory.png']
        for r in gg:
            key=r['target'],r['condition'],r['panel'],r['query']['id'];q=expected_gen[key];assert r['query']==q and r['reader_query']==q['query']
            proof=r['processor_input'];assert proof['text']==frame.replace('__REGISTERED_QUERY__',q['query'])
            assert proof['input_ids_digest']==digest([r['generation']['input_token_ids']]) and proof['input_tokens']==len(r['generation']['input_token_ids'])
            localtokens+=proof['input_tokens'];raw_score(r,q,r['panel']=='mcq')
        for r in rr:
            q,ch,gi=expected_rank[r['target'],r['condition'],r['panel'],r['query']['id']]
            assert r['query']==q and r['choices']==ch and r['gold_index']==gi and r['reader_query']==q['query']
            key=digest(dict(query=q['query'],choices=ch,png=r['png_sha']));assert key==r['input_identity'];score=r['score']
            assert score['model_prompt']==frame.replace('__REGISTERED_QUERY__',q['query'])
            assert len(score['scores'])==len(score['candidate_token_counts'])==len(score['processor_calls'])==4
            assert all(math.isfinite(x) for x in score['scores']) and min(score['candidate_token_counts'])>0
            for call,choice in zip(score['processor_calls'],ch):assert call['text']==score['model_prompt']+choice and call['input_tokens']>0
            margin=score['scores'][gi]-max(x for k,x in enumerate(score['scores']) if k!=gi)
            assert r['gold_margin']==margin and r['unique_correct']==(margin>0)
            if key in cache:assert r['reused'] and r['actual_candidate_forwards']==0 and score==cache[key]
            else:
                assert not r['reused'] and r['actual_candidate_forwards']==4;cache[key]=score;calls+=4
                localtokens+=sum(x['input_tokens'] for x in score['processor_calls'])
        for r in cc:
            q=expected_gen[r['target'],r['condition'],'recovery_training',r['query']['id']];assert r['query']==q
            assert r['processor_input']['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+q['target']+ident['termination']['assistant_end_token_text']
            assert math.isfinite(r['loss']) and math.isclose(r['loss'],r['answer_ce']+r['eos_ce'],rel_tol=2e-5,abs_tol=2e-5)
            assert len(r['token_nll'])==len(r['target_ids'])==r['answer_tokens']+1
            assert r['target_ids'][-1]==ident['termination']['assistant_end_token_id']
            assert all(math.isfinite(x) for x in r['token_nll'])
            assert math.isclose(sum(r['token_nll'][:-1])/r['answer_tokens'],r['answer_ce'],rel_tol=2e-5,abs_tol=2e-5)
            assert r['token_nll'][-1]==r['eos_ce']
            localtokens+=r['processor_input']['input_tokens']
        assert localtokens==done['processed_input_tokens'] and calls==done['ranking_candidate_forwards']
        evaltokens+=localtokens;evalcalls+=calls;evaltime+=done['seconds'];gens+=gg;ranks+=rr;ces+=cc
    assert len(set(snapshots))==1
    assert snapshots[0]==digest(s.load_json(source/'evaluation/shard-0/identity.json')['snapshots'])
    gi=exact_index(gens,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected_gen)
    ri=exact_index(ranks,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected_rank)
    ci=exact_index(ces,lambda r:(r['target'],r['condition'],'recovery_training',r['query']['id']),{k:v for k,v in expected_gen.items() if k[2]=='recovery_training'})
    assert len(gi)==2568 and len(ri)==1584 and len(ci)==528 and evalcalls<=6336
    # Reuse old references; never run fresh parent/text/blank inference.
    for i in range(4):
        for r in rows(source/'evaluation'/f'shard-{i}'/'reads.jsonl'):
            if r['condition'] in ('A','B'):gi[r['target'],r['condition'],r['panel'],r['query']['id']]=r
        for r in rows(source/'evaluation'/f'shard-{i}'/'ranking.jsonl'):
            if r['condition'] in ('A','B'):ri[r['target'],r['condition'],r['panel'],r['query']['id']]=r
    counts=defaultdict(lambda:[0,0]);groups=defaultdict(lambda:[0,0]);correct={};rankok={}
    for metric,index in (('generation',gi),('ranking',ri)):
        for key,r in index.items():
            sid,arm,panel,_=key;q=r['query'];t=e['targets'][sid]
            ok=raw_score(r,q,panel=='mcq') if metric=='generation' else r['gold_margin']>0
            (correct if metric=='generation' else rankok)[key]=ok
            for label in ('all','K1' if len(t['state'])==1 else 'multi','changed' if q['scope']==t['changed_scope'] else 'untouched'):
                add(counts,f'{arm}/{metric}/{panel}/{label}',ok)
                if panel=='qualification':add(counts,f'{arm}/{metric}/qualification_{q["kind"]}/{label}',ok)
            if t['state'][q['scope']] is not None:
                gid=next(x['semantic_group'] for x in t['mcq'] if x['scope']==q['scope'])
                add(groups,f'{arm}/{metric}/{panel}/{gid}',ok)
    recovery={};joint={};slot_joint={};rotation={};contrast={}
    for arm in ('A','B','C','J'):
        capacities=defaultdict(lambda:[0,0]);slots=defaultdict(lambda:[0,0]);jointcaps=defaultdict(lambda:[0,0]);jointgroups=defaultdict(lambda:[0,0]);states={};jstates={};jslots={}
        for sid,t in e['targets'].items():
            qr=[gi[sid,arm,'qualification',q['id']] for q in t['qualification']]
            png=qr[0]['png_sha'];assert all(r['png_sha']==png for r in qr)
            rs=s.recovery_summary(t['state'],t['qualification'],[dict(query=r['query'],score=r['score'],png_sha256=r['png_sha']) for r in qr],png)
            states[sid]=rs;add(capacities,'K'+str(len(t['state'])),rs['recovery_complete']);this={}
            for scope,value in t['state'].items():
                qq=[q for q in t['qualification'] if q['scope']==scope and q['kind']=='recovery'];assert len(qq)==2
                rec=all(correct[sid,arm,'qualification',q['id']] for q in qq)
                for label in ('all','active' if value is not None else 'cleared','changed' if scope==t['changed_scope'] else 'untouched'):add(slots,label,rec)
                app=[q for q in t['application_reserved'] if q['scope']==scope];assert len(app)==(2 if value is not None else 0)
                assert all(ri[sid,arm,'application_reserved',q['id']]['png_sha']==png for q in app)
                this[scope]=rec and all(rankok[sid,arm,'application_reserved',q['id']] for q in app)
                jslots[f'{sid}/{scope}']=this[scope]
                if value is not None:
                    gid=next(q['semantic_group'] for q in t['mcq'] if q['scope']==scope);add(jointgroups,gid,this[scope])
            alljoint=all(this.values());mcq=all(correct[sid,arm,'mcq',q['id']] for q in t['mcq'])
            assert all(gi[sid,arm,'mcq',q['id']]['png_sha']==png for q in t['mcq'])
            jstates[sid]=dict(slots=this,complete=alljoint,with_mcq=alljoint and mcq,cleared_state=any(v is None for v in t['state'].values()))
            add(jointcaps,'K'+str(len(t['state'])),alljoint)
        manifest=s.load_json(s.REPORT/'registered/manifest.json');chains=[]
        for ep in manifest['episodes']:
            if ep['panel']!='train-k4' or not all(x['target_state_id'] in states for x in ep['transitions']):continue
            ids=[x['target_state_id'] for x in ep['transitions']]
            chains.append(dict(episode=ep['id'],recovery_complete=all(states[sid]['recovery_complete'] for sid in ids),joint_complete=all(jstates[sid]['complete'] for sid in ids)))
        assert len(chains)==4
        recovery[arm]=dict(capacity=dict(capacities),slots=dict(slots),states=states,offline_teacher_chains=chains,
            selective_clear=[sid for sid,t in e['targets'].items() if any(v is None for v in t['state'].values()) and states[sid]['recovery_complete']])
        joint[arm]=dict(capacity=dict(jointcaps),states=jstates,complete=sum(v['complete'] for v in jstates.values()),with_mcq=sum(v['with_mcq'] for v in jstates.values()),
            active_slot_semantic_groups=dict(jointgroups),active_slot_macro=sum(a/b for a,b in jointgroups.values())/len(jointgroups))
        slot_joint[arm]=jslots;rotation[arm]=[];contrast[arm]=[]
        for sid,t in p['targets'].items():
            for case in t['applications']:
                if case['value_id'] not in reg['contrast_values']:continue
                checks=[]
                for rot in range(4):
                    q,_,_=s.ranking_candidates(case,rot);pan='application_training_rotations' if rot else 'application_training'
                    checks.append(rankok[sid,arm,pan,q['id']])
                rotation[arm].append(dict(target=sid,value_id=case['value_id'],case_id=case['id'],correct=checks,all_four=all(checks)))
        for c in e['overwrite_contrasts']:
            before=e['targets'][c['before_state']];after=e['targets'][c['after_state']]
            for qa in [q for q in before['application_reserved'] if q['scope']==c['scope']]:
                qb=next(q for q in after['application_reserved'] if q['case_id']==qa['case_id'] and q['scope']==qa['scope'])
                assert qa['query']==qb['query'] and qa['target']!=qb['target']
                contrast[arm].append(dict(contrast=c,case_id=qa['case_id'],both_correct=rankok[c['before_state'],arm,'application_reserved',qa['id']] and rankok[c['after_state'],arm,'application_reserved',qb['id']]))
    paired={}
    for left,right in (('C','J'),('B','C'),('B','J')):
        table=defaultdict(Counter)
        for metric,index in (('generation',correct),('ranking',rankok)):
            for (sid,arm,panel,qid),ok in index.items():
                if arm==left:table[metric+'/'+panel][f'{int(ok)}->{int(index[sid,right,panel,qid])}']+=1
        for sid in e['targets']:
            for key in ('complete','with_mcq'):table['joint_state/'+key][f'{int(joint[left]["states"][sid][key])}->{int(joint[right]["states"][sid][key])}']+=1
        for key,ok in slot_joint[left].items():table['joint_slot'][f'{int(ok)}->{int(slot_joint[right][key])}']+=1
        paired[left+'->'+right]={k:dict(v) for k,v in table.items()}
    macro={}
    for key in groups:
        prefix=key.rsplit('/',1)[0]
        if prefix not in macro:
            values=[a/b for k,(a,b) in groups.items() if k.startswith(prefix+'/')];macro[prefix]=sum(values)/len(values)
    absolute={arm:dict(recovery=all(recovery[arm]['capacity'][k][0]>=v[0] for k,v in reg['progression']['recovery'].items()),
        original_mcq=counts[f'{arm}/generation/mcq/all'][0]>=68,transfer_ranking=counts[f'{arm}/ranking/application_reserved/all'][0]>=135,
        overwrite_pairs=sum(x['both_correct'] for x in contrast[arm])>=6,
        every_overwrite=all(any(x['both_correct'] for x in contrast[arm] if x['contrast']==c) for c in e['overwrite_contrasts'])) for arm in ('C','J')}
    ce_summary={arm:dict(mean_answer=sum(r['answer_ce'] for r in ces if r['condition']==arm)/264,
        mean_eos=sum(r['eos_ce'] for r in ces if r['condition']==arm)/264,cells=264) for arm in ('C','J')}
    result=dict(registration_digest=digest(reg),updates=5120,gradient_forwards=dict(forward),training_tokens=dict(tokens),training_seconds=dict(cost),
        generations=2568,rankings=1584,endpoint_ce_forwards=528,actual_candidate_forwards=evalcalls,evaluation_tokens=evaltokens,evaluation_seconds=evaltime,
        counts=dict(counts),groups=dict(groups),macro=macro,recovery=recovery,joint=joint,paired=paired,rotation=rotation,contrasts=contrast,
        absolute_targets=absolute,endpoint_ce=ce_summary,gradient_diagnostics={k:dict(v) for k,v in gradstats.items()},writer_updates=0,
        limitation='Observed gradient-excluded diagnostics, independent endpoint teachers; no recurrent Writer or usable-version claim')
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','recovery','joint','rotation','contrasts')},indent=2));return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    a=ap.parse_args();verify(a.output,a.source)
