"""Paired outcomes and measured cost; no new inference or changed scoring."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.experiments import prefeval_compositional_evidence as e
from scripts.reporting.summarize_prefeval_plan10_gates import read_rows
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score


def score(row):
    item=row['query'];mcq=row['panel'] in ('mcq','application_xml','attribute_xml')
    if row['panel'].endswith('xml'):
        item={**item,'target_index':'ABCD'.index(item['target'].removeprefix('<choice>').removesuffix('</choice>'))}
    return raw_score(row,item,mcq)


def main():
    ap=argparse.ArgumentParser()
    for key in ('output','control','source'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();reg,p,cases=e.load()
    roots={'E':a.output,'R':a.control,'V':a.source}
    reads={arm:read_rows(root,'reads',arm) for arm,root in roots.items()}
    paired={}
    for old in ('R','V'):
        panels=defaultdict(lambda:dict(total=0,rescued=0,regressed=0,both_correct=0,both_wrong=0))
        groups=defaultdict(lambda:dict(total=0,E=0,control=0))
        for key,row in reads['E'].items():
            if key not in reads[old]:continue
            other=reads[old][key];assert row['query']==other['query']
            now,before=score(row),score(other);d=panels[row['panel']];d['total']+=1
            d['rescued']+=int(now and not before);d['regressed']+=int(before and not now)
            d['both_correct']+=int(now and before);d['both_wrong']+=int(not now and not before)
            if row['panel']=='mcq':
                d=groups[row['query']['semantic_group']];d['total']+=1;d['E']+=int(now);d['control']+=int(before)
        paired[old]=dict(panels=dict(panels),semantic_groups=dict(groups))
    current={name:{(r['target'],r['condition'],r['panel'],r['query']['id']):r
             for path in (a.output/'evaluation').glob(f'shard-*/{name}.jsonl')
             for r in map(json.loads,path.read_text().splitlines())} for name in ('reads','ranking')}
    contrasts=e.s.load_json(e.s.DATA/'evaluation-payload.json')['overwrite_contrasts']
    overwrite={}
    for arm in ('E','R'):
        out=[]
        for c in contrasts:
            xx,rr=[],[]
            for left,right in zip(cases[c['before']],cases[c['after']]):
                xok,rok=[],[]
                assert left['situation']==right['situation'] and left['proposals']==right['proposals'] and left['base_rotation']==right['base_rotation']
                for sid,case in ((c['before_state'],left),(c['after_state'],right)):
                    q,_,gold=e.q.xml_candidates(case,0)
                    row=current['reads'][sid,arm,'attribute_xml',q['id']]
                    xok.append(raw_score(row,{**q,'target_index':gold},True))
                    q,_,gold=e.s.ranking_candidates(case,0)
                    scores=current['ranking'][sid,arm,'attribute_full_action',q['id']]['score']['scores']
                    rok.append(scores[gold]>max(v for i,v in enumerate(scores) if i!=gold))
                xx.append(all(xok));rr.append(all(rok))
            out.append(dict(scope=c['scope'],xml=[sum(xx),len(xx)],ranking=[sum(rr),len(rr)]))
        overwrite[arm]=out
    done=[json.loads(path.read_text()) for path in (a.output/'training/E').glob('*/complete.json')]
    ev=[json.loads(path.read_text()) for path in (a.output/'evaluation').glob('shard-*/complete.json')]
    result=dict(paired=paired,new_overwrite=overwrite,
        training={k:sum(d[k] for d in done) for k in ('seconds','processed_input_tokens','reader_forwards')},
        evaluation={k:sum(d[k] for d in ev) for k in ('seconds','processed_input_tokens','ranking_candidate_forwards','recovery_ce_forwards')})
    (a.output/'paired-outcomes.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({**result,'paired':{arm:v['panels'] for arm,v in paired.items()}},indent=2))


if __name__=='__main__':main()
