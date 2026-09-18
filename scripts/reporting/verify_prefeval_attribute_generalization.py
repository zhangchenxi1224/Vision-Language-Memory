"""Reconstruct the core Plan10 result and its preregistered gates."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.experiments import prefeval_attribute_generalization as x
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score

s=x.s;digest=x.digest;tensor_sha=x.tensor_sha
def rows(p):return [json.loads(v) for v in p.read_text(encoding='utf-8').splitlines()]
def ratio(vals):return sum(vals)/len(vals) if vals else 0.0

def verify(output,source):
    reg,p,cases=x.load();e=s.load_json(s.DATA/'evaluation-payload.json')
    assert s.file_sha(source/'final-verified.json')==reg['parent_verified']
    starts={};forwards=Counter()
    for shard in range(4):
        ident=s.load_json(output/f'train-identity-{shard}.json');assert ident['actual_host']==reg['runtime']['actual']
        assert ident['registration_digest']==digest(reg)
    for sid in p['targets']:
        for arm in ('R','D'):
            folder=output/'training'/arm/sid;done=s.load_json(folder/'complete.json')
            for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
            initial=torch.load(folder/'initial-latent.pt',map_location='cpu',weights_only=True)
            assert tensor_sha(initial)==done['initial_tensor_sha']==reg['endpoint_receipts'][sid]['final_tensor_sha']
            starts[sid,arm]=initial;forwards[arm]+=done['reader_forwards']
            trace=rows(folder/'optimization.jsonl');assert len(trace)==64
            for step,row in enumerate(trace):assert row['step']==step and row['schedule']==x.schedule(step)
        assert torch.equal(starts[sid,'R'],starts[sid,'D'])
    assert forwards=={'R':38400,'D':38400}
    gens=[];ranks=[];ces=[];candidate_forwards=0
    for shard in range(4):
        folder=output/'evaluation'/f'shard-{shard}';done=s.load_json(folder/'complete.json')
        for name,h in done['artifacts'].items():assert s.file_sha(folder/name)==h
        gg,rr,cc=rows(folder/'reads.jsonl'),rows(folder/'ranking.jsonl'),rows(folder/'recovery-ce.jsonl')
        assert len(gg)==done['generations'] and len(rr)==done['rankings'] and len(cc)==done['recovery_ce_forwards']
        for row in cc:
            assert len(row['target_ids'])==len(row['token_nll'])==row['answer_tokens']+1
            assert abs(row['loss']-row['answer_ce']-row['eos_ce'])<2e-5
        candidate_forwards+=done['ranking_candidate_forwards'];gens+=gg;ranks+=rr;ces+=cc
    assert len(gens)==5320 and len(ranks)==2256 and len(ces)==528 and candidate_forwards==9024
    g={(r['target'],r['condition'],r['panel'],r['query']['id']):r for r in gens}
    rank={(r['target'],r['condition'],r['panel'],r['query']['id']):r for r in ranks}
    counts=defaultdict(list);value_mcq=defaultdict(list);value_attribute=defaultdict(list)
    recovery={};joint={}
    for arm in ('R','D'):
        capacity=defaultdict(lambda:[0,0]);joint_states={}
        for sid,t in e['targets'].items():
            quals=[g[sid,arm,'qualification',item['id']] for item in t['qualification']]
            summary=s.recovery_summary(t['state'],t['qualification'],[dict(query=r['query'],score=r['score'],png_sha256=r['png_sha']) for r in quals],quals[0]['png_sha'])
            capacity['K'+str(len(t['state']))][1]+=1;capacity['K'+str(len(t['state']))][0]+=int(summary['recovery_complete'])
            slot_ok=[]
            for scope,value in t['state'].items():
                forms=[item for item in t['qualification'] if item['scope']==scope and item['kind']=='recovery']
                rec=all(raw_score(g[sid,arm,'qualification',item['id']],item,False) for item in forms)
                apps=[item for item in t['application_reserved'] if item['scope']==scope]
                app=all(rank[sid,arm,'application_reserved',item['id']]['unique_correct'] for item in apps)
                slot_ok.append(rec and app)
            mcq=[]
            for item in t['mcq']:
                ok=raw_score(g[sid,arm,'mcq',item['id']],item,True);mcq.append(ok);value_mcq[arm,item['value_id']].append(ok);counts[arm,'mcq'].append(ok)
            for item in t['application_reserved']:counts[arm,'reserved_ranking'].append(rank[sid,arm,'application_reserved',item['id']]['unique_correct'])
            for item0 in t['application_training']:
                for case in cases[item0['value_id']]:
                    query,_,_=x.q.xml_candidates(case,0);ok=raw_score(g[sid,arm,'attribute_xml',query['id']],{**query,'target_index':ord(query['target'][-10])-65},True)
                    value_attribute[arm,item0['value_id']].append(ok);counts[arm,'attribute_xml'].append(ok)
                    query,_,_=s.ranking_candidates(case,0);counts[arm,'attribute_ranking'].append(rank[sid,arm,'attribute_full_action',query['id']]['unique_correct'])
            joint_states[sid]=dict(complete=all(slot_ok),with_mcq=all(slot_ok) and all(mcq))
        recovery[arm]=dict(capacity=dict(capacity));joint[arm]=dict(complete=sum(v['complete'] for v in joint_states.values()),with_mcq=sum(v['with_mcq'] for v in joint_states.values()))
    source_data=s.load_json(x.DATA/'authoring-source.json');groups={v['id']:v['source_groups'][0] for v in source_data['values']}
    metrics={}
    for arm in ('R','D'):
        per_value={vid:ratio(vals) for (a,vid),vals in value_mcq.items() if a==arm}
        by_group=defaultdict(list)
        for vid,val in per_value.items():by_group[groups[vid]].append(val)
        metrics[arm]=dict(original_mcq_accuracy=ratio(counts[arm,'mcq']),semantic_group_macro=ratio([ratio(v) for v in by_group.values()]),
            reserved_ranking_accuracy=ratio(counts[arm,'reserved_ranking']),attribute_xml_accuracy=ratio(counts[arm,'attribute_xml']),
            attribute_ranking_accuracy=ratio(counts[arm,'attribute_ranking']),occurrences=len(counts[arm,'mcq']))
    absolute={arm:dict(recovery=all(recovery[arm]['capacity'][k][0]>=v[0] for k,v in reg['progression']['recovery'].items()),
        original_mcq=sum(counts[arm,'mcq'])>=reg['progression']['mcq_min'],reserved_ranking=sum(counts[arm,'reserved_ranking'])>=reg['progression']['reserved_ranking_min']) for arm in ('R','D')}
    gain=metrics['D']['semantic_group_macro']-metrics['R']['semantic_group_macro']
    result=dict(registration_digest=digest(reg),updates=5120,gradient_forwards=dict(forwards),generations=len(gens),rankings=len(ranks),
        candidate_forwards=candidate_forwards,metrics=metrics,recovery=recovery,joint=joint,absolute_targets=absolute,
        comparative=dict(gain=gain,minimum=reg['comparative_target']['minimum_gain'],passed=gain>=reg['comparative_target']['minimum_gain']),writer_updates=0,
        usable=absolute['D']['recovery'] and absolute['D']['original_mcq'] and absolute['D']['reserved_ranking'] and gain>=reg['comparative_target']['minimum_gain'],limitation=reg['limitation'])
    (output/'final-verified.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    (output/'results.md').write_text(f"# PrefEval attribute generalization\n\n- R semantic-group macro MCQ: {metrics['R']['semantic_group_macro']:.3f}\n- D semantic-group macro MCQ: {metrics['D']['semantic_group_macro']:.3f}\n- D-R gain: {gain:.3f}\n- D new attribute XML: {metrics['D']['attribute_xml_accuracy']:.3f}\n- D new attribute ranking: {metrics['D']['attribute_ranking_accuracy']:.3f}\n- Usable under registered gates: **{result['usable']}**\n\n{reg['limitation']}\n",encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2,ensure_ascii=False));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();verify(a.output,a.source)
