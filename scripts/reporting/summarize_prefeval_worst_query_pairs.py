"""Plan12 paired outcomes and recovery diagnostics from frozen records only."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.experiments import prefeval_worst_query_consolidation as x
from scripts.reporting.summarize_prefeval_plan10_gates import read_rows
from scripts.reporting.summarize_prefeval_compositional_pairs import score
s=x.s


def ranked(row):
    scores=row['score']['scores'];gold=row['gold_index']
    return scores[gold]>max(v for i,v in enumerate(scores) if i!=gold)


def tally(values):return [sum(values),len(values)]


def main():
    ap=argparse.ArgumentParser()
    for key in ('output','source'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();reg,p,cases=x.load();ev=s.load_json(s.DATA/'evaluation-payload.json')
    roots={'M':a.output,'W':a.output,'E':a.source}
    reads={arm:read_rows(root,'reads',arm) for arm,root in roots.items()}
    ranks={arm:read_rows(root,'ranking',arm) for arm,root in roots.items()}
    paired={};panels={};cohorts={};failures=[];counterfactual={};overwrite={}
    for arm,rr in reads.items():
        pp=defaultdict(list);cc=defaultdict(list)
        for key,row in rr.items():
            sid,panel,qid=key;q=row['query'];ok=score(row);pp[panel].append(ok)
            if q.get('kind')=='recovery' or panel=='recovery_coverage':
                subset='excluded' if panel=='qualification' else 'training'
                changed='changed' if q['scope']==p['targets'][sid]['changed_scope'] else 'untouched'
                cc[f'{subset}:{changed}'].append(ok)
                if not ok:
                    raw=row['generation']['raw'];state=p['targets'][sid]['state']
                    other=[scope for scope,value in state.items() if scope!=q['scope'] and value is not None and raw.strip()==value]
                    failures.append(dict(arm=arm,state=sid,panel=panel,scope=q['scope'],query_id=qid,cohort=changed,
                        expected=q['target'],actual=raw,truncated=row['generation']['truncated'],exact_other_scope_matches=other))
        panels[arm]={k:tally(v) for k,v in pp.items()};cohorts[arm]={k:tally(v) for k,v in cc.items()}
        cf={'xml':[],'ranking':[]}
        for sid,t in p['targets'].items():
            for vid in sorted({c['value_id'] for c in t['applications']}):
                for pair in ((0,1),(2,3)):
                    xx=[];yy=[]
                    for index in pair:
                        case=cases[vid][index];q,_,_=x.q.xml_candidates(case,0)
                        xx.append(score(rr[sid,'attribute_xml',q['id']]))
                        q,_,_=s.ranking_candidates(case,0);yy.append(ranked(ranks[arm][sid,'attribute_full_action',q['id']]))
                    cf['xml'].append(all(xx));cf['ranking'].append(all(yy))
        counterfactual[arm]={k:tally(v) for k,v in cf.items()}
        oo=[]
        for contrast in ev['overwrite_contrasts']:
            xx=[];yy=[]
            for before,after in zip(cases[contrast['before']],cases[contrast['after']]):
                assert before['situation']==after['situation'] and before['proposals']==after['proposals'] and before['base_rotation']==after['base_rotation']
                px=[];py=[]
                for sid,case in ((contrast['before_state'],before),(contrast['after_state'],after)):
                    q,_,_=x.q.xml_candidates(case,0);px.append(score(rr[sid,'attribute_xml',q['id']]))
                    q,_,_=s.ranking_candidates(case,0);py.append(ranked(ranks[arm][sid,'attribute_full_action',q['id']]))
                xx.append(all(px));yy.append(all(py))
            oo.append(dict(scope=contrast['scope'],xml=tally(xx),ranking=tally(yy)))
        overwrite[arm]=oo
    for now,before in (('M','E'),('W','E'),('W','M')):
        pp=defaultdict(lambda:dict(total=0,rescued=0,regressed=0,both_correct=0,both_wrong=0))
        for key,row in reads[now].items():
            other=reads[before][key];assert row['query']==other['query']
            u,v=score(row),score(other);d=pp[row['panel']];d['total']+=1
            d['rescued']+=int(u and not v);d['regressed']+=int(v and not u);d['both_correct']+=int(u and v);d['both_wrong']+=int(not u and not v)
        paired[f'{now}_vs_{before}']=dict(pp)
    ce=defaultdict(list)
    for path in (a.output/'evaluation').glob('shard-*/recovery-ce.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);subset='excluded' if row['panel']=='qualification' else 'training'
            ce[row['condition']+':'+subset].append(row)
    ce_summary={k:dict(queries=len(rows),mean_answer_ce=sum(r['answer_ce'] for r in rows)/len(rows),
        mean_eos_ce=sum(r['eos_ce'] for r in rows)/len(rows),max_total_ce=max(r['loss'] for r in rows),
        all_teacher_forced_tokens_correct=sum(all(r['teacher_forced_correct']) for r in rows),
        minimum_gold_token_margin=min(min(r['gold_token_margins']) for r in rows)) for k,rows in ce.items()}
    result=dict(generation_panels=panels,paired=paired,recovery_cohorts=cohorts,counterfactual_pairs=counterfactual,
        new_memory_overwrite=overwrite,endpoint_ce=ce_summary,writer_updates=0,
        diagnostic_note='Exact-other-slot and truncation flags are mechanical only. Remaining errors require wording/content review; scoring is unchanged.')
    for name,value in [('paired-diagnostics.json',result),('recovery-failures.json',failures)]:
        (a.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('generation_panels','paired')},indent=2))


if __name__=='__main__':main()
