"""Read-only T1 MCQ comparison of two already archived full-experiment stages."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def main(source, output):
    stages={}
    report={'source_experiment':'codex/dreamlite-prefeval-rgb-20260917 @ b41021a',
            'scope':'T1 MCQ only; existing archived outputs; no new inference or OOD analysis',
            'stages':{},'paired_matched_changes':{}}
    for stage,filename,expected_hash in [
        ('write','write-complete-evidence.jsonl.gz','5956ab0ac0c7b8800fe4d83ef6180b41f5958096b1eaedd1f8ff3360042747d9'),
        ('write-ackmix','ackmix-complete-evidence.jsonl.gz','4b54b4d23b868ec45968745ee3a46977d7582031276762d231bae2c43a18c785')]:
        path=source/filename
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==expected_hash
        index={}
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line)
                d=item['data']
                if not item['path'].startswith('evaluations/students/') or '/'+stage+'/' not in item['path']:
                    continue
                if not isinstance(d,dict) or 'rows' not in d:
                    continue
                arm=item['path'].split('/')[2]
                for row in d['rows']:
                    if row['task']!='mcq' or row['family']!='T1':
                        continue
                    key=(arm,d['split'],d['id'],d['condition'],d['seed'])
                    assert key not in index and isinstance(row['correct'],bool)
                    index[key]=row
        stats={}
        for arm in ['A','B']:
            for split,n in [('train',64),('dev',90)]:
                for control,seeds in [('matched',[0,1]),('mismatched',[0])]:
                    selected={k:r for k,r in index.items() if (k[0],k[1],k[3])==(arm,split,control)}
                    assert len(selected)==n*len(seeds)
                    assert len({k[2] for k in selected})==n and {k[4] for k in selected}==set(seeds)
                    stats[f'{arm}/{split}/{control}']={'expected':n*len(seeds),
                        'correct':sum(r['correct'] for r in selected.values()),
                        'per_seed':{str(seed):sum(r['correct'] for k,r in selected.items() if k[4]==seed) for seed in seeds}}
        assert len(index)==924
        stages[stage]=index
        report['stages'][stage]={'archive':filename,'sha256':digest,'conditions':924,'scores':stats}
    assert stages['write'].keys()==stages['write-ackmix'].keys()
    for arm in ['A','B']:
        for split in ['train','dev']:
            keys=[k for k in stages['write'] if (k[0],k[1],k[3])==(arm,split,'matched')]
            counts=collections.Counter()
            for k in keys:
                before=stages['write'][k]['correct']
                after=stages['write-ackmix'][k]['correct']
                counts['both_correct' if before and after else 'gained' if after else 'lost' if before else 'both_wrong']+=1
            report['paired_matched_changes'][f'{arm}/{split}']=dict(counts)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source,args.output)
