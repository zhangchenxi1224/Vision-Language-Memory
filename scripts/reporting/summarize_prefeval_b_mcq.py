"""Summarize a complete, explicitly specified B MCQ matrix using the upstream parser."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records,official_mcq,sha


def summarize(args):
    ids=[r['base_pair_id'] for r in load_records(args.split)]
    families=args.families.split(',')
    prefixes=list(map(int,args.prefixes.split(',')))
    controls=args.controls.split(',')
    assert 'memory' in controls and args.chains>=1
    mcq=official_mcq(args.upstream)
    rows={}
    sources={}
    for directory in args.input:
        for path in sorted(directory.glob('readback-*.jsonl')):
            sources[str(path)]=sha(path)
            for line in path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                assert row['task']=='mcq'
                key=tuple(row[k] for k in ['pair_id','chain','prefix','control','family'])
                assert key not in rows, f'Duplicate observation: {key}'
                parsed=mcq['extract_choice'](row['generated']['raw'])
                gold='ABCD'[row['option_order'].index(0)]
                assert parsed==row['predicted_letter'] and gold==row['correct_letter']
                assert row['correct']==(parsed==gold)
                assert row['parse_failure']==(parsed is None)
                rows[key]=row
    expected=set()
    for pid in ids:
        for chain in range(args.chains):
            for prefix in prefixes:
                for control in controls:
                    if control=='blank' and (prefix!=0 or chain!=0):
                        continue
                    if control=='text' and chain!=0:
                        continue
                    for family in families:
                        expected.add((pid,chain,prefix,control,family))
    assert set(rows)==expected, f'Incomplete/unexpected matrix: missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}'
    pngs={}
    groups=defaultdict(list)
    for key,row in rows.items():
        pid,chain,prefix,control,family=key
        groups[control,family,prefix].append(row)
        if control in ['memory','mismatch']:
            endpoint=pid,chain,prefix,control
            assert endpoint not in pngs or pngs[endpoint]==row['png_sha256'], 'Question forms used different PNGs'
            pngs[endpoint]=row['png_sha256']
        if control=='mismatch':
            matched=rows[pid,chain,prefix,'memory',family]
            assert row['option_order']==matched['option_order']
    def score(values):
        values=list(values)
        return {'correct':sum(values),'total':len(values)}
    metrics=[]
    for (control,family,prefix),values in sorted(groups.items()):
        metrics.append({'control':control,'family':family,'prefix':prefix,
            **score(r['correct'] for r in values),
            'parse_failures':sum(r['parse_failure'] for r in values),
            'truncations':sum(r['generated']['truncated'] for r in values)})
    def correct(pid,chain,prefix,family,control='memory'):
        return rows[pid,chain,prefix,control,family]['correct']
    paired=[]
    joint=[]
    retention=[]
    trajectories=[]
    for family in families:
        for prefix in prefixes:
            joint.append({'family':family,'prefix':prefix,'definition':'all_noise_chains_correct_per_preference',
                **score(all(correct(pid,c,prefix,family) for c in range(args.chains)) for pid in ids)})
            if 'mismatch' in controls:
                pairs=[(correct(pid,c,prefix,family),correct(pid,c,prefix,family,'mismatch')) for pid in ids for c in range(args.chains)]
                paired.append({'family':family,'prefix':prefix,'total':len(pairs),
                    'repair':sum(m and not w for m,w in pairs),'regression':sum(w and not m for m,w in pairs),
                    'both_correct':sum(m and w for m,w in pairs),'both_wrong':sum(not m and not w for m,w in pairs)})
            if prefix!=0 and 0 in prefixes:
                subset=[correct(pid,c,prefix,family) for pid in ids for c in range(args.chains) if correct(pid,c,0,family)]
                retention.append({'family':family,'prefix':prefix,'definition':'conditional_on_initial_correct_same_seed',**score(subset)})
        trajectories.append({'family':family,'prefixes':prefixes,
            'definition':'all_requested_prefixes_and_all_noise_chains_correct_per_preference',
            **score(all(correct(pid,c,p,family) for c in range(args.chains) for p in prefixes) for pid in ids)})
    ood=[]
    if {'O1','O2'}<=set(families):
        for prefix in prefixes:
            ood.append({'prefix':prefix,'definition':'O1_and_O2_on_same_image_per_chain',
                **score(all(correct(pid,c,prefix,f) for f in ['O1','O2']) for pid in ids for c in range(args.chains))})
            ood.append({'prefix':prefix,'definition':'O1_and_O2_all_noise_chains_per_preference',
                **score(all(correct(pid,c,prefix,f) for f in ['O1','O2'] for c in range(args.chains)) for pid in ids)})
    return {'split':args.split,'preferences':len(ids),'chains':args.chains,'prefixes':prefixes,'families':families,
        'observations':len(rows),'matrix_complete':True,'metrics':metrics,'matched_mismatched_pairs':paired,
        'noise_joint':joint,'ood_joint':ood,'conditional_retention':retention,'trajectory_joint':trajectories,
        'memory_failures':[{'pair_id':k[0],'chain':k[1],'prefix':k[2],'family':k[4]} for k,r in rows.items() if k[3]=='memory' and not r['correct']],
        'source_files':sources,'official_parser_sha256':sha(args.upstream/'utils/utils_mcq.py'),
        'note':'Seeds and question forms are repeated measurements. Conditional retention does not replace the full denominator. No free-answer judge is involved.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--split',choices=['pilot','train','dev'],required=True)
    p.add_argument('--prefixes',default='0')
    p.add_argument('--families',default='T1')
    p.add_argument('--chains',type=int,default=2)
    p.add_argument('--controls',default='memory,mismatch,blank,text')
    p.add_argument('--upstream',type=Path,default=ROOT/'third_party/prefeval_reference')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['split','preferences','observations','matrix_complete','metrics']}))
