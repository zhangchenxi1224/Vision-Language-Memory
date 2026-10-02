"""Reconstruct Plan12 exposures, mean/worst aggregation and fixed-PNG outcomes."""
import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments import prefeval_worst_query_consolidation as x
from scripts.reporting.verify_prefeval_compositional_evidence import expected
from scripts.reporting.summarize_prefeval_plan10_gates import summarize
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score
s=x.s


def rows(path):return [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()]
def close(a,b):assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-5),(a,b)
def key(row):return row['target'],row['condition'],row['panel'],row['query']['id']


def verify(a):
    reg,p,cases=x.load();ev=s.load_json(s.DATA/'evaluation-payload.json')
    assert s.file_sha(a.source/'final-verified.json')==reg['parent_verified']
    forward=Counter();token=Counter();seconds=Counter();worst=defaultdict(Counter);snapshots=set();identities={}
    for shard in range(4):
        ident=s.load_json(a.output/f'train-identity-{shard}.json')
        assert ident['actual_host']==reg['runtime']['actual'] and ident['registration_digest']==x.digest(reg)
        assert ident['assignment']==sorted(p['targets'])[shard::4]
        assert s.load_json(a.output/f'train-complete-{shard}.json')['targets']==ident['assignment']
        snapshots.add(x.digest(ident['snapshots']))
        for sid in ident['assignment']:identities[sid]=ident
    for sid,t in p['targets'].items():
        starts=[];identity=identities[sid];frame=identity['prompt_frame'];end=identity['termination']['assistant_end_token_text']
        for arm in ('M','W'):
            folder=a.output/'training'/arm/sid;done=s.load_json(folder/'complete.json')
            assert done['inherited_updates']==[256,128,64,64,64] and done['additional_updates']==32
            for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
            z=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True);starts.append(z)
            assert x.tensor_sha(z)==reg['endpoint_receipts'][sid]['final_tensor_sha']==done['initial_tensor_sha']
            final=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            assert torch.isfinite(final).all() and x.tensor_sha(final)==done['final_tensor_sha']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert ck['next_step']==32 and ck['registration_digest']==x.digest(reg) and x.tensor_sha(ck['latent'])==done['final_tensor_sha']
            assert int(next(iter(opt['state'].values()))['step'])==32
            for k,v in reg['optimizer'].items():
                value=opt['param_groups'][0][k];assert (list(value) if isinstance(value,tuple) else value)==v
            trace=rows(folder/'optimization.jsonl');assert len(trace)==32
            nf=nt=0
            for step,row in enumerate(trace):
                bank=x.recovery_bank(t);apps=x.applications(t,step,cases)
                assert row['step']==step and row['schedule']==x.schedule(step)
                rr=row['losses'][:len(bank)];aa=row['losses'][len(bank):]
                assert len(rr)==len(bank) and len(aa)==len(apps)
                assert len({(v['query']['scope'],v['query']['id']) for v in rr})==len(bank)
                for loss,(q,w) in zip(rr,bank):
                    assert loss['kind']=='recovery' and loss['query']==q and loss['mean_weight']==w
                    assert loss['reader_forwards']==1 and math.isfinite(loss['gradient_norm'])
                    close(loss['loss'],loss['answer_ce']+loss['eos_ce'])
                    assert loss['processor_input']['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+q['target']+end
                    assert loss['processed_input_tokens']==loss['processor_input']['input_tokens']
                    nf+=1;nt+=loss['processed_input_tokens']
                selected=min(rr,key=lambda v:(-v['loss'],v['query']['scope'],v['query']['id']))
                which=[selected['query']['scope'],selected['query']['id']]
                assert row['worst_query']==which;close(row['worst_recovery_loss'],selected['loss'])
                mean=sum(v['mean_weight']*v['loss'] for v in rr);close(row['mean_recovery_loss'],mean)
                value=mean if arm=='M' else .5*mean+.5*selected['loss'];close(row['recovery_objective'],value)
                worst[arm]['|'.join(which)]+=1
                for v in rr:
                    w=v['mean_weight'] if arm=='M' else .5*v['mean_weight']+.5*int([v['query']['scope'],v['query']['id']]==which)
                    close(v['effective_weight'],w)
                app=0.
                for loss,(q,w,choices,gold) in zip(aa,apps):
                    assert loss['kind']=='application' and loss['query']==q and loss['weight']==w
                    assert loss['choices']==choices and loss['gold_index']==gold and len(loss['candidate_token_counts'])==4
                    assert loss['reader_forwards']==4 and len(loss['processor_inputs'])==4
                    for choice,proof in zip(choices,loss['processor_inputs']):
                        assert proof['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+choice
                    assert loss['processed_input_tokens']==sum(v['input_tokens'] for v in loss['processor_inputs'])
                    score=torch.tensor(loss['scores']);close(loss['loss'],float(torch.logsumexp(score,0)-score[gold]))
                    app+=w*loss['loss'];nf+=4;nt+=loss['processed_input_tokens']
                close(row['application_objective'],app);close(row['weighted_objective'],value+app)
                assert all(math.isfinite(v) for v in row['gradient'].values())
            assert nf==done['reader_forwards'] and nt==done['processed_input_tokens']
            forward[arm]+=nf;token[arm]+=nt;seconds[arm]+=done['seconds']
        assert torch.equal(*starts)
    assert forward=={'M':41728,'W':41728}
    eg,er=expected(reg,p,cases,('M','W'),('M','W'))
    ec={k:v for k,v in eg.items() if k[2] in ('recovery_training','recovery_coverage') or (k[2]=='qualification' and v['kind']=='recovery')}
    actual_g={};actual_r={};actual_c={};png={};eval_tokens=eval_calls=0;eval_seconds=0.
    for arm in ('M','W'):
        for sid in p['targets']:png[sid,arm]=s.file_sha(a.output/'training'/arm/sid/'memory.png')
    for shard in range(4):
        folder=a.output/'evaluation'/f'shard-{shard}';ident=s.load_json(folder/'identity.json');done=s.load_json(folder/'complete.json')
        assert ident['registration_digest']==x.digest(reg) and ident['actual_host']==reg['runtime']['actual']
        assert ident['assignment']==sorted(p['targets'])[shard::4];snapshots.add(x.digest(ident['snapshots']))
        assert ident['frozen_endpoints']=={f'{arm}/{sid}':s.file_sha(a.output/'training'/arm/sid/'complete.json') for arm in ('M','W') for sid in p['targets']}
        for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
        frame=ident['prompt_frame'];local_tokens=0
        gg,rr,cc=rows(folder/'reads.jsonl'),rows(folder/'ranking.jsonl'),rows(folder/'recovery-ce.jsonl')
        assert len(gg)==done['generations'] and len(rr)==done['rankings'] and len(cc)==done['recovery_ce_forwards']
        for row in gg:
            k=key(row);assert k not in actual_g and k in eg;q=eg[k]
            assert row['query']==q and row['reader_query']==q['query'] and row['png_sha']==png[k[:2]]
            assert row['processor_input']['text']==row['generation']['chat_prompt']==frame.replace('__REGISTERED_QUERY__',q['query'])
            assert row['processor_input']['input_ids_digest']==x.digest([row['generation']['input_token_ids']])
            local_tokens+=row['processor_input']['input_tokens'];actual_g[k]=row
        for row in rr:
            k=key(row);assert k not in actual_r and k in er;q,choices,gold=er[k]
            assert row['query']==q and row['choices']==choices and row['gold_index']==gold and row['png_sha']==png[k[:2]]
            score=row['score'];assert len(score['scores'])==len(score['processor_calls'])==4 and all(math.isfinite(v) for v in score['scores'])
            for choice,proof in zip(choices,score['processor_calls']):
                assert proof['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+choice
                local_tokens+=proof['input_tokens']
            margin=score['scores'][gold]-max(v for i,v in enumerate(score['scores']) if i!=gold)
            assert row['gold_margin']==margin and row['unique_correct']==(margin>0);actual_r[k]=row
        for row in cc:
            k=key(row);assert k not in actual_c and k in ec;q=ec[k]
            assert row['query']==q and row['png_sha']==png[k[:2]]
            n=row['answer_tokens'];assert len(row['target_ids'])==len(row['token_nll'])==len(row['gold_token_margins'])==len(row['teacher_forced_correct'])==n+1
            assert row['target_ids'][-1]==ident['termination']['assistant_end_token_id']
            assert all(math.isfinite(v) for v in row['token_nll']+row['gold_token_margins'])
            close(row['answer_ce'],sum(row['token_nll'][:n])/n);close(row['eos_ce'],row['token_nll'][-1]);close(row['loss'],row['answer_ce']+row['eos_ce'])
            assert row['processor_input']['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+q['target']+ident['termination']['assistant_end_token_text']
            local_tokens+=row['processor_input']['input_tokens'];actual_c[k]=row
        assert local_tokens==done['processed_input_tokens'] and done['ranking_candidate_forwards']==4*len(rr)
        eval_tokens+=local_tokens;eval_calls+=4*len(rr);eval_seconds+=done['seconds']
    assert actual_g.keys()==eg.keys() and actual_r.keys()==er.keys() and actual_c.keys()==ec.keys()
    assert len(actual_g)==5320 and len(actual_r)==2256 and len(actual_c)==2288 and eval_calls==9024 and len(snapshots)==1
    outcomes={arm:summarize(a.output,arm,ev,reg['progression']) for arm in ('M','W')}
    outcomes['E']=summarize(a.source,'E',ev,reg['progression'])
    panel_scores={arm:defaultdict(list) for arm in ('M','W')}
    for k,row in actual_g.items():
        item=row['query'];mcq=k[2] in ('mcq','application_xml','attribute_xml')
        if k[2].endswith('xml'):item={**item,'target_index':'ABCD'.index(item['target'].removeprefix('<choice>').removesuffix('</choice>'))}
        panel_scores[k[1]][k[2]].append(raw_score(row,item,mcq))
    result=dict(registration_digest=x.digest(reg),updates=2560,gradient_forwards=dict(forward),training_tokens=dict(token),training_seconds=dict(seconds),
        generations=len(actual_g),rankings=len(actual_r),endpoint_ce=len(actual_c),candidate_forwards=eval_calls,evaluation_tokens=eval_tokens,evaluation_seconds=eval_seconds,
        outcomes=outcomes,generation_scores={arm:{k:[sum(v),len(v)] for k,v in panels.items()} for arm,panels in panel_scores.items()},
        worst_query_selections={arm:dict(v) for arm,v in worst.items()},writer_updates=0,needs_teacher_review=True,
        primary=dict(joint_W_minus_M=outcomes['W']['joint']-outcomes['M']['joint'],mcq_W_minus_E=outcomes['W']['mcq'][0]-outcomes['E']['mcq'][0]),
        absolute_targets={arm:all(outcomes[arm]['absolute_gates'].values()) for arm in ('M','W')})
    (a.output/'final-verified.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('outcomes','worst_query_selections')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('source','output'):ap.add_argument('--'+k,type=Path,required=True)
    verify(ap.parse_args())
