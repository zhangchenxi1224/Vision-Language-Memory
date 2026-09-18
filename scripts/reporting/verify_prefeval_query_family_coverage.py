"""Independently reconstruct the fixed Plan09 U/V coverage experiment."""
import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments import prefeval_query_family_coverage as q
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score
from scripts.reporting.verify_prefeval_rgb_endpoint_diagnostic import exact_index
from vision_memory.reader.open_answer import normalize_short_answer
s=q.s;digest=q.digest;tensor_sha=q.tensor_sha

def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def add(d,k,v):d[k][0]+=int(v);d[k][1]+=1

def verify(output,source):
    reg,p=q.load();e=s.load_json(s.DATA/'evaluation-payload.json');assert digest(e)==reg['evaluation_digest']
    assert s.file_sha(source/'final-verified.json')==reg['parent_verified']
    code=(output/'coverage-code-commit.txt').read_text().strip();forward=Counter();tokens=Counter();seconds=Counter();snap=[]
    for i in range(4):
        ident=s.load_json(output/f'train-identity-{i}.json');assert ident['code']==code and ident['registration_digest']==digest(reg)
        assert ident['actual_host']==reg['runtime']['actual'] and ident['assignment']==sorted(p['targets'])[i::4]
        assert s.load_json(output/f'train-complete-{i}.json')['targets']==ident['assignment'];snap.append(digest(ident['snapshots']))
    for sid,t in p['targets'].items():
        src=source/'training/J'/sid;initial=reg['endpoint_receipts'][sid]
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        pair=[]
        for arm in ('U','V'):
            folder=output/'training'/arm/sid;done=s.load_json(folder/'complete.json')
            assert done['inherited_updates']==[256,128,64] and done['additional_updates']==64
            for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
            z=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True);pair.append(z)
            assert tensor_sha(z)==initial['final_tensor_sha']==done['initial_tensor_sha']
            final=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True);assert torch.isfinite(final).all() and tensor_sha(final)==done['final_tensor_sha']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True);ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert ck['next_step']==64 and ck['registration_digest']==digest(reg) and tensor_sha(ck['latent'])==done['final_tensor_sha']
            assert int(next(iter(opt['state'].values()))['step'])==64 and opt['param_groups']==ck['optimizer']['param_groups']
            for k,v in reg['optimizer'].items():assert opt['param_groups'][0][k]==v
            trace=rows(folder/'optimization.jsonl');assert [x['step'] for x in trace]==list(range(64));nf=nt=0
            for step,row in enumerate(trace):
                rec,app=q.jobs(t,step,arm);expected=[(x,w,None,None) for x,w in rec]+app
                assert len(row['losses'])==len(expected) and row['measurement'].startswith('pre-update')
                for loss,(query,w,ch,gold) in zip(row['losses'],expected):
                    assert loss['query']==query and loss['weight']==w and math.isfinite(loss['loss'])
                    calls=4 if ch else 1;assert loss['reader_forwards']==calls;nf+=calls;nt+=loss['processed_input_tokens']
                    if ch:
                        assert loss['choices']==ch and loss['gold_index']==gold and len(loss['candidate_token_counts'])==4
                        score=torch.tensor(loss['scores']);value=float(torch.logsumexp(score,0)-score[gold])
                    else:value=loss['answer_ce']+loss['eos_ce']
                    assert math.isclose(value,loss['loss'],rel_tol=2e-5,abs_tol=2e-5)
                assert math.isclose(row['weighted_objective'],sum(x['weight']*x['loss'] for x in row['losses']),rel_tol=1e-12)
                assert all(v is None or math.isfinite(v) for v in row['gradient'].values())
                if step==0:assert row['gradient']['displacement_norm']>0
            assert nf==done['reader_forwards'] and nt==done['processed_input_tokens'];forward[arm]+=nf;tokens[arm]+=nt;seconds[arm]+=done['seconds']
        assert torch.equal(*pair)
    assert forward=={'U':38400,'V':38400}
    cases={}
    for fn in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/fn).values():
            for c in cc:cases[(c['id'],c['scope'],c['value_id'])]=c
    expected_gen={};expected_rank={}
    for sid,t in e['targets'].items():
        train=p['targets'][sid]
        for arm in ('U','V'):
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for item in t[panel]:expected_gen[sid,arm,panel,item['id']]=item
            for scope,value in train['state'].items():
                for bank in ('question','instruction'):
                    for ix in range(4):
                        item=q.recovery_form(scope,value,bank,ix);expected_gen[sid,arm,'recovery_coverage',item['id']]=item
            for item0 in t['application_training']:
                item,_,_=q.xml_candidates(cases[(item0['case_id'],item0['scope'],item0['value_id'])],0);expected_gen[sid,arm,'application_xml',item['id']]=item
            for panel in ('application_training','application_reserved'):
                for item0 in t[panel]:
                    case=cases[(item0['case_id'],item0['scope'],item0['value_id'])]
                    for rot in (range(4) if panel=='application_training' and item0['value_id'] in reg['contrast_values'] else (0,)):
                        item,ch,gold=s.ranking_candidates(case,rot);pan='application_training_rotations' if rot else panel
                        expected_rank[sid,arm,pan,item['id']]=(item,ch,gold)
    gens=[];ranks=[];ces=[];ecalls=etokens=0;etime=0
    for i in range(4):
        folder=output/'evaluation'/f'shard-{i}';ident=s.load_json(folder/'identity.json');done=s.load_json(folder/'complete.json')
        assert ident['code']==code and ident['registration_digest']==digest(reg) and ident['actual_host']==reg['runtime']['actual']
        assert ident['assignment']==sorted(p['targets'])[i::4];snap.append(digest(ident['snapshots']))
        assert ident['frozen_endpoints']=={f'{a}/{sid}':s.file_sha(output/'training'/a/sid/'complete.json') for sid in p['targets'] for a in ('U','V')}
        for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
        gg,rr,cc=rows(folder/'reads.jsonl'),rows(folder/'ranking.jsonl'),rows(folder/'recovery-ce.jsonl')
        assert len(gg)==done['generations'] and len(rr)==done['rankings'] and len(cc)==done['recovery_ce_forwards'];frame=ident['prompt_frame'];local=0;calls=0;cache={}
        for r in gg:
            key=r['target'],r['condition'],r['panel'],r['query']['id'];item=expected_gen[key];assert r['query']==item and r['reader_query']==item['query']
            assert r['processor_input']['text']==frame.replace('__REGISTERED_QUERY__',item['query'])
            assert r['processor_input']['input_ids_digest']==digest([r['generation']['input_token_ids']])
            local+=r['processor_input']['input_tokens'];raw_score(r,item,r['panel'] in ('mcq','application_xml'))
            assert r['png_sha']==s.load_json(output/'training'/r['condition']/r['target']/'complete.json')['artifacts']['memory.png']
        for r in rr:
            item,ch,gold=expected_rank[r['target'],r['condition'],r['panel'],r['query']['id']]
            assert r['query']==item and r['choices']==ch and r['gold_index']==gold;key=digest(dict(query=item['query'],choices=ch,png=r['png_sha']));assert key==r['input_identity']
            score=r['score'];assert len(score['scores'])==len(score['processor_calls'])==4 and all(math.isfinite(x) for x in score['scores'])
            margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold);assert r['gold_margin']==margin and r['unique_correct']==(margin>0)
            if key in cache:assert r['reused'] and not r['actual_candidate_forwards'] and score==cache[key]
            else:assert not r['reused'] and r['actual_candidate_forwards']==4;cache[key]=score;calls+=4;local+=sum(x['input_tokens'] for x in score['processor_calls'])
        for r in cc:
            item=expected_gen[r['target'],r['condition'],'recovery_training',r['query']['id']];assert r['query']==item
            assert r['processor_input']['text']==frame.replace('__REGISTERED_QUERY__',item['query'])+item['target']+ident['termination']['assistant_end_token_text']
            assert len(r['token_nll'])==len(r['target_ids'])==r['answer_tokens']+1 and all(math.isfinite(x) for x in r['token_nll'])
            assert math.isclose(r['loss'],r['answer_ce']+r['eos_ce'],rel_tol=2e-5,abs_tol=2e-5);local+=r['processor_input']['input_tokens']
        assert local==done['processed_input_tokens'] and calls==done['ranking_candidate_forwards'];etokens+=local;ecalls+=calls;etime+=done['seconds'];gens+=gg;ranks+=rr;ces+=cc
    assert len(set(snap))==1 and len(gens)==4648 and len(ranks)==1584 and len(ces)==528 and ecalls<=6336
    gi=exact_index(gens,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected_gen)
    ri=exact_index(ranks,lambda r:(r['target'],r['condition'],r['panel'],r['query']['id']),expected_rank)
    oldg={};oldr={}
    for i in range(4):
        for r in rows(source/'evaluation'/f'shard-{i}'/'reads.jsonl'):
            if r['condition']=='J':oldg[r['target'],'J',r['panel'],r['query']['id']]=r
        for r in rows(source/'evaluation'/f'shard-{i}'/'ranking.jsonl'):
            if r['condition']=='J':oldr[r['target'],'J',r['panel'],r['query']['id']]=r
    counts=defaultdict(lambda:[0,0]);correct={};rankok={}
    for metric,index in (('generation',{**oldg,**gi}),('ranking',{**oldr,**ri})):
        for key,r in index.items():
            sid,arm,panel,_=key;ok=raw_score(r,r['query'],panel in ('mcq','application_xml')) if metric=='generation' else r['gold_margin']>0
            (correct if metric=='generation' else rankok)[key]=ok;add(counts,f'{arm}/{metric}/{panel}/all',ok)
    recovery={};joint={};failure_classes={}
    for arm in ('J','U','V'):
        cap=defaultdict(lambda:[0,0]);jcap=defaultdict(lambda:[0,0]);states={};jstates={};fail=Counter()
        for sid,t in e['targets'].items():
            qr=[({**oldg,**gi})[sid,arm,'qualification',x['id']] for x in t['qualification']];png=qr[0]['png_sha']
            summary=s.recovery_summary(t['state'],t['qualification'],[dict(query=x['query'],score=x['score'],png_sha256=x['png_sha']) for x in qr],png)
            states[sid]=summary;add(cap,'K'+str(len(t['state'])),summary['recovery_complete']);slots={}
            values={normalize_short_answer(v) for v in t['state'].values() if v is not None}
            for scope,value in t['state'].items():
                forms=[x for x in t['qualification'] if x['scope']==scope and x['kind']=='recovery'];rec=all(correct[sid,arm,'qualification',x['id']] for x in forms)
                apps=[x for x in t['application_reserved'] if x['scope']==scope];slots[scope]=rec and all(rankok[sid,arm,'application_reserved',x['id']] for x in apps)
                for item in forms:
                    r=({**oldg,**gi})[sid,arm,'qualification',item['id']]
                    if correct[sid,arm,'qualification',item['id']]:continue
                    norm=r['score']['normalized'];raw=r['score']['raw'];expected=r['score']['normalized_expected']
                    if r['generation']['truncated']:kind='truncation'
                    elif r['score']['format_status'] not in ('exact_short_answer','normalized_match'):kind='formatting'
                    elif value is None and norm!='no active preference':kind='stale_value'
                    elif value is not None and norm=='no active preference':kind='incorrect_absence'
                    elif norm in values and norm!=expected:kind='wrong_slot_value'
                    else:kind='other_content_mismatch'
                    fail[kind]+=1
            complete=all(slots.values());mcq=all(correct[sid,arm,'mcq',x['id']] for x in t['mcq']);jstates[sid]=dict(slots=slots,complete=complete,with_mcq=complete and mcq);add(jcap,'K'+str(len(t['state'])),complete)
        manifest=s.load_json(s.REPORT/'registered/manifest.json');chains=[]
        for ep in manifest['episodes']:
            if ep['panel']=='train-k4' and all(x['target_state_id'] in states for x in ep['transitions']):
                ids=[x['target_state_id'] for x in ep['transitions']];chains.append(dict(episode=ep['id'],recovery_complete=all(states[x]['recovery_complete'] for x in ids),joint_complete=all(jstates[x]['complete'] for x in ids)))
        recovery[arm]=dict(capacity=dict(cap),states=states,selective_clear=[sid for sid,t in e['targets'].items() if any(v is None for v in t['state'].values()) and states[sid]['recovery_complete']],offline_teacher_chains=chains)
        joint[arm]=dict(capacity=dict(jcap),states=jstates,complete=sum(x['complete'] for x in jstates.values()),with_mcq=sum(x['with_mcq'] for x in jstates.values()))
        failure_classes[arm]=dict(fail)
    paired={}
    for left,right in (('J','U'),('J','V'),('U','V')):
        table=defaultdict(Counter)
        for metric,index in (('generation',correct),('ranking',rankok)):
            for (sid,arm,panel,i),ok in index.items():
                if arm==left and (sid,right,panel,i) in index:table[metric+'/'+panel][f'{int(ok)}->{int(index[sid,right,panel,i])}']+=1
        for sid in e['targets']:
            for key in ('complete','with_mcq'):table['joint_state/'+key][f'{int(joint[left]["states"][sid][key])}->{int(joint[right]["states"][sid][key])}']+=1
        paired[left+'->'+right]={k:dict(v) for k,v in table.items()}
    rotation={arm:[] for arm in ('J','U','V')};contrasts={arm:[] for arm in rotation}
    for arm in rotation:
        for sid,t in p['targets'].items():
            for case in t['applications']:
                if case['value_id'] not in reg['contrast_values']:continue
                checks=[]
                for rot in range(4):
                    item,_,_=s.ranking_candidates(case,rot);pan='application_training_rotations' if rot else 'application_training';checks.append(rankok[sid,arm,pan,item['id']])
                rotation[arm].append(dict(target=sid,value_id=case['value_id'],case_id=case['id'],correct=checks,all_four=all(checks)))
        for c in e['overwrite_contrasts']:
            before,after=e['targets'][c['before_state']],e['targets'][c['after_state']]
            for a in [x for x in before['application_reserved'] if x['scope']==c['scope']]:
                b=next(x for x in after['application_reserved'] if x['case_id']==a['case_id'] and x['scope']==a['scope'])
                contrasts[arm].append(dict(contrast=c,case_id=a['case_id'],both_correct=rankok[c['before_state'],arm,'application_reserved',a['id']] and rankok[c['after_state'],arm,'application_reserved',b['id']]))
    absolute={a:dict(recovery=all(recovery[a]['capacity'][k][0]>=v[0] for k,v in reg['progression']['recovery'].items()),
        original_mcq=counts[f'{a}/generation/mcq/all'][0]>=68,transfer_ranking=counts[f'{a}/ranking/application_reserved/all'][0]>=135,
        overwrite_pairs=sum(x['both_correct'] for x in contrasts[a])>=6,every_overwrite=all(any(x['both_correct'] for x in contrasts[a] if x['contrast']==c) for c in e['overwrite_contrasts'])) for a in ('U','V')}
    result=dict(registration_digest=digest(reg),updates=5120,gradient_forwards=dict(forward),training_tokens=dict(tokens),training_seconds=dict(seconds),
        generations=4648,rankings=1584,endpoint_ce_forwards=528,actual_candidate_forwards=ecalls,evaluation_tokens=etokens,evaluation_seconds=etime,
        counts=dict(counts),recovery=recovery,joint=joint,paired=paired,rotation=rotation,contrasts=contrasts,failure_classes=failure_classes,
        absolute_targets=absolute,writer_updates=0,limitation=reg['limitation'])
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('recovery','joint','rotation','contrasts')},indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();verify(a.output,a.source)
