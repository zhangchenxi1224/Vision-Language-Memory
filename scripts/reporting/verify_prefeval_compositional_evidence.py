"""Exact record reconstruction for the archived R2 control and Plan11 E."""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments import prefeval_attribute_generalization as r2
from scripts.experiments import prefeval_compositional_evidence as e
from scripts.reporting.summarize_prefeval_plan10_gates import summarize
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score
s=e.s


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def expected(reg,p,cases,full_arms,new_arms):
    evaluation=s.load_json(s.DATA/'evaluation-payload.json')
    old={}
    for name in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/name).values():
            for c in cc:old[c['id'],c['scope'],c['value_id']]=c
    gens,ranks={},{}
    def add(dst,key,value):
        assert key not in dst
        dst[key]=value
    for sid,t in evaluation['targets'].items():
        for arm in full_arms:
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for q in t[panel]:add(gens,(sid,arm,panel,q['id']),q)
            for scope,value in p['targets'][sid]['state'].items():
                for bank in ('question','instruction'):
                    for ix in range(4):
                        q=e.q.recovery_form(scope,value,bank,ix);add(gens,(sid,arm,'recovery_coverage',q['id']),q)
            for q0 in t['application_training']:
                q,_,_=e.q.xml_candidates(old[q0['case_id'],q0['scope'],q0['value_id']],0)
                add(gens,(sid,arm,'application_xml',q['id']),q)
            for panel in ('application_training','application_reserved'):
                for q0 in t[panel]:
                    case=old[q0['case_id'],q0['scope'],q0['value_id']]
                    for rotation in (range(4) if panel=='application_training' and q0['value_id'] in reg['contrast_values'] else (0,)):
                        q,ch,gold=s.ranking_candidates(case,rotation)
                        pan='application_training_rotations' if rotation else panel
                        add(ranks,(sid,arm,pan,q['id']),(q,ch,gold))
        for arm in new_arms:
            for vid in sorted({q['value_id'] for q in t['application_training']}):
                for case in cases[vid]:
                    q,_,_=e.q.xml_candidates(case,0);add(gens,(sid,arm,'attribute_xml',q['id']),q)
                    q,ch,gold=s.ranking_candidates(case,0);add(ranks,(sid,arm,'attribute_full_action',q['id']),(q,ch,gold))
    return gens,ranks


def verify(a):
    module=r2 if a.mode=='r2' else e
    reg,p,cases=module.load()
    arms=('R','D') if a.mode=='r2' else ('E',)
    new_arms=arms if a.mode=='r2' else ('E','R')
    gens,ranks=expected(reg,p,cases,arms,new_arms)
    assert len(gens)==reg['budget']['generations'] and len(ranks)==reg['budget']['ranking_decisions']
    for sid,t in p['targets'].items():
        for arm in arms:
            folder=a.output/'training'/arm/sid
            done=s.load_json(folder/'complete.json')
            assert done['additional_updates']==64
            for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
            initial=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True)
            assert e.tensor_sha(initial)==reg['endpoint_receipts'][sid]['final_tensor_sha']
            final=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            assert torch.isfinite(final).all() and e.tensor_sha(final)==done['final_tensor_sha']
            opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            assert int(next(iter(opt['state'].values()))['step'])==64
            for k,v in reg['optimizer'].items():
                actual=opt['param_groups'][0][k]
                assert (list(actual) if isinstance(actual,tuple) else actual)==v
            trace=rows(folder/'optimization.jsonl');assert len(trace)==64
            nf=nt=0
            for step,row in enumerate(trace):
                rec,apps=module.jobs(t,step,arm,cases)
                jobs=[(q,w,None,None) for q,w in rec]+apps
                assert row['step']==step and row['schedule']==module.schedule(step) and len(row['losses'])==len(jobs)
                for loss,(q,w,ch,gold) in zip(row['losses'],jobs):
                    assert loss['query']==q and loss['weight']==w and math.isfinite(loss['loss'])
                    assert loss['reader_forwards']==(4 if ch else 1)
                    nf+=loss['reader_forwards'];nt+=loss['processed_input_tokens']
                    if ch:
                        assert loss['choices']==ch and loss['gold_index']==gold
                        score=torch.tensor(loss['scores'])
                        assert math.isclose(float(torch.logsumexp(score,0)-score[gold]),loss['loss'],abs_tol=2e-5)
                assert math.isclose(row['weighted_objective'],sum(v['weight']*v['loss'] for v in row['losses']),rel_tol=1e-12)
            assert nf==done['reader_forwards'] and nt==done['processed_input_tokens']
    actual_gen,actual_rank={},{}
    cached_png={}
    for shard in range(4):
        folder=a.output/'evaluation'/f'shard-{shard}'
        ident=s.load_json(folder/'identity.json')
        assert ident['actual_host']==reg['runtime']['actual'] and ident['registration_digest']==e.digest(reg)
        done=s.load_json(folder/'complete.json')
        for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
        for name,dst,exp in [('reads',actual_gen,gens),('ranking',actual_rank,ranks)]:
            for row in rows(folder/f'{name}.jsonl'):
                sid,arm,pan=row['target'],row['condition'],row['panel']
                key=sid,arm,pan,row['query']['id']
                assert key not in dst and key in exp
                root=a.control if a.mode!='r2' and arm=='R' else a.output
                png=root/'training'/arm/sid/'memory.png'
                if png not in cached_png:cached_png[png]=s.file_sha(png)
                assert row['png_sha']==cached_png[png]
                if name=='reads':
                    q=exp[key];assert row['query']==q and row['reader_query']==q['query']
                    mcq=pan in ('mcq','application_xml','attribute_xml')
                    item={**q,'target_index':'ABCD'.index(q['target'].removeprefix('<choice>').removesuffix('</choice>'))} if pan.endswith('xml') else q
                    raw_score(row,item,mcq)
                else:
                    q,ch,gold=exp[key]
                    assert row['query']==q and row['choices']==ch and row['gold_index']==gold
                    scores=row['score']['scores'];assert len(scores)==4 and all(math.isfinite(v) for v in scores)
                    margin=scores[gold]-max(v for i,v in enumerate(scores) if i!=gold)
                    assert margin==row['gold_margin'] and row['unique_correct']==(margin>0)
                dst[key]=row
    assert actual_gen.keys()==gens.keys() and actual_rank.keys()==ranks.keys()
    result=dict(mode=a.mode,registration_digest=e.digest(reg),generations=len(gens),rankings=len(ranks),
        exact_coverage=True,training_exposures_reconstructed=True,optimizer_steps=64,ranking_flags_recomputed=True,
        limitations=['R2 did not record per-component recovery CE or gradient norms in training; these cannot be reconstructed from its scalar losses.'],writer_updates=0)
    if a.mode!='r2':
        ev=s.load_json(s.DATA/'evaluation-payload.json')
        result['states']={arm:summarize(root,arm,ev,reg['progression']) for arm,root in [('E',a.output),('R',a.control),('V',a.source)]}
        gain=result['states']['E']['semantic_group_macro']-result['states']['R']['semantic_group_macro']
        result['comparison']=dict(gain=gain,minimum=.10)
        result['teacher_candidate']=all(result['states']['E']['absolute_gates'].values()) and gain>=.10
        result['new_cases']={}
        for arm in new_arms:
            xml,ranking,xml_pairs,rank_pairs=[],[],[],[]
            for sid,t in ev['targets'].items():
                for vid in sorted({q['value_id'] for q in t['application_training']}):
                    for first,second in ((0,1),(2,3)):
                        xx,rr=[],[]
                        for case in (cases[vid][first],cases[vid][second]):
                            q,_,gold=e.q.xml_candidates(case,0)
                            ok=raw_score(actual_gen[sid,arm,'attribute_xml',q['id']],{**q,'target_index':gold},True)
                            xx.append(ok);xml.append(ok)
                            q,_,gold=s.ranking_candidates(case,0)
                            scores=actual_rank[sid,arm,'attribute_full_action',q['id']]['score']['scores']
                            ok=scores[gold]>max(v for i,v in enumerate(scores) if i!=gold)
                            rr.append(ok);ranking.append(ok)
                        xml_pairs.append(all(xx));rank_pairs.append(all(rr))
            result['new_cases'][arm]={k:[sum(v),len(v)] for k,v in dict(xml=xml,ranking=ranking,xml_pairs=xml_pairs,ranking_pairs=rank_pairs).items()}
    filename='reuse-verification-addendum.json' if a.mode=='r2' else 'final-verified.json'
    (a.output/filename).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='states'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['r2','E'],default='E')
    for name in ('output','source','control'):ap.add_argument('--'+name,type=Path,required=True)
    verify(ap.parse_args())
