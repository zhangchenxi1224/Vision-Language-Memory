"""Summarize frozen K1 readbacks with fixed denominators and pending judge visibility."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import official_mcq

def identity(row):
    return tuple(row[k] for k in ['pair_id','chain','prefix','control','family','task'])

def canonical(row):
    return json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def result(counts,expected):
    observed=len(counts)
    scored=sum(x['correct'] is not None for x in counts)
    correct=sum(x['correct'] is True for x in counts)
    if observed>expected:
        raise ValueError(f'Observed {observed} exceeds registered denominator {expected}')
    return {'expected':expected,'observed':observed,'scored':scored,'correct':correct,
        'accuracy':correct/expected if scored==expected else None,
        'observed_scored_accuracy':correct/scored if scored else None,
        'status':'complete' if scored==expected else 'incomplete_or_judge_pending',
        'parse_failures':sum(x.get('parse_failure',False) for x in counts),
        'generation_truncations':sum(x.get('truncated',False) for x in counts)}

def summarize(inputs,*,expected_ids,prefixes,chains,judge_dir=None):
    mcq=official_mcq(ROOT/'third_party/prefeval_reference')
    judges={}
    judge_models=set()
    if judge_dir:
        for path in Path(judge_dir).glob('*.json'):
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('status')=='complete':
                judges[canonical(record['input'])]=record['correct']
                judge_models.add((record['judge_model'],record['model_label']))
        if len(judge_models)>1:
            raise ValueError('Do not mix distinct judge models in a single accuracy')
    metrics=[]
    joint=[]
    sources=[]
    for label,directory in inputs:
        rows={}
        pngs={}
        for path in sorted(Path(directory).glob('readback-*.jsonl')):
            payload=path.read_bytes()
            sources.append({'label':label,'file':str(path),'sha256':hashlib.sha256(payload).hexdigest()})
            for line in payload.decode('utf-8').splitlines():
                row=json.loads(line)
                key=identity(row)
                if row['pair_id'] not in expected_ids:
                    raise ValueError('Readback has an unexpected sample ID')
                if row['prefix'] not in prefixes or not 0 <= row['chain'] < chains:
                    raise ValueError('Readback has an unexpected prefix or noise-chain index')
                endpoint=tuple(row[k] for k in ['pair_id','chain','prefix','control'])
                if row['control'] in ['memory','mismatch']:
                    if endpoint in pngs and pngs[endpoint]!=row['png_sha256']:
                        raise ValueError('Different questions did not read the same frozen PNG')
                    pngs[endpoint]=row['png_sha256']
                if key in rows:
                    if canonical(rows[key])!=canonical(row):
                        raise ValueError(f'Conflicting duplicate readback {label}/{key}')
                    continue
                rows[key]=row
        groups=defaultdict(list)
        scored={}
        for key,row in rows.items():
            score={'correct':None,'truncated':row['generated']['truncated']}
            if row['task']=='mcq':
                correct='ABCD'[row['option_order'].index(0)]
                predicted=mcq['extract_choice'](row['generated']['raw'])
                assert correct==row['correct_letter'] and row['correct']==(predicted==correct)
                score.update(correct=predicted==correct,parse_failure=predicted is None)
            else:
                score['correct']=judges.get(canonical(row))
            group=(row['prefix'],row['control'],row['family'],row['task'])
            groups[group].append(score)
            scored[key]=score['correct']
        # Explicit missing memory groups prevent an incomplete run from looking complete.
        for prefix in prefixes:
            for family in ['T1','T2','T3','O1','O2']:
                for task in ['free','mcq']:
                    groups.setdefault((prefix,'memory',family,task),[])
        for group,values in sorted(groups.items()):
            prefix,control,family,task=group
            n=len(expected_ids)*(1 if control in ['blank','text'] else chains)
            metrics.append({'label':label,'prefix':prefix,'control':control,'family':family,'task':task,**result(values,n)})
        for prefix in prefixes:
            for task in ['free','mcq']:
                pairs=[]
                for pid in expected_ids:
                    for chain in range(chains):
                        first=scored.get((pid,chain,prefix,'memory','O1',task))
                        second=scored.get((pid,chain,prefix,'memory','O2',task))
                        if first is not None and second is not None:
                            pairs.append({'correct':bool(first and second)})
                joint.append({'label':label,'prefix':prefix,'task':task,
                    'definition':'O1 and O2 both correct on the identical endpoint PNG',
                    **result(pairs,len(expected_ids)*chains)})
    return {'expected_ids':expected_ids,'prefixes':prefixes,'noise_chains':chains,
        'judge_models':[{'model':a,'label':b} for a,b in sorted(judge_models)],
        'metrics':metrics,'ood_joint':joint,'source_files':sources,
        'warning':'Observed-only scores are partial diagnostics. Missing or unjudged results are not silently dropped. Noise chains are repeated measures, not independent preference samples.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',required=True,help='label=directory containing readback shards')
    p.add_argument('--split',choices=['pilot','dev','official'],required=True)
    p.add_argument('--history-file',type=Path)
    p.add_argument('--prefixes',default='0')
    p.add_argument('--chains',type=int,default=1)
    p.add_argument('--judge-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from scripts.experiments.prefeval_k1_data import load_records
    value=summarize([x.split('=',1) for x in args.input],expected_ids=[r['base_pair_id'] for r in load_records(args.split,history_file=args.history_file)],
        prefixes=list(map(int,args.prefixes.split(','))),chains=args.chains,judge_dir=args.judge_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
