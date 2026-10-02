"""Pair complete, previously verified validation archives without changing the scorer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

GOLD_IDS={'ambient':[59614],'jazz':[73,9802],'no active preference':[2152,4541,21933],
          'green':[13250],'juice':[8613,558],'linen':[3732,268],'pasta':[79,14300]}


def read_rows(path,digest):
    with path.open('rb') as stream:
        actual=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            actual.update(block)
        if actual.hexdigest()!=digest:
            raise ValueError('Archive differs from its independently observed SHA256')
    with tarfile.open(path) as archive:
        rows=[json.loads(line) for line in archive.extractfile('generations.jsonl')]
    indexed={}
    for row in rows:
        key=(row['case'],row['condition'],row['noise_seed'],row['prompt_id'])
        if key in indexed:
            raise ValueError('Repeated validation cell')
        expected=GOLD_IDS[row['gold']]
        passed=row['generated_token_ids']==expected+[151645]
        if (row['scorer']['gold_token_ids']!=expected or passed!=bool(
                row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos'])):
            raise ValueError('Raw tokens contradict the strict scorer')
        indexed[key]=(row,passed)
    return indexed


def compare(registration,root):
    matrices={}
    for spec in registration['matrices']:
        before=read_rows(root/spec['before']['file'],spec['before']['sha256'])
        after=read_rows(root/spec['after']['file'],spec['after']['sha256'])
        if set(before)!=set(after) or len(before)!=spec['raw_rows']:
            raise ValueError('Unequal or incomplete paired matrix')
        changes=Counter()
        cases={}
        failures=Counter()
        for key,(previous,prior_pass) in before.items():
            current,current_pass=after[key]
            for field in ('query','gold','event_text'):
                if previous.get(field)!=current.get(field):
                    raise ValueError('Case semantics differ: '+field)
            if key[1]!='matched':
                if (previous['generated_token_ids']!=current['generated_token_ids']
                        or previous['image_sha256']!=current['image_sha256']):
                    raise ValueError('Fixed negative control changed')
                continue
            label=f'{int(prior_pass)}->{int(current_pass)}'
            changes[label]+=1
            case=cases.setdefault(key[0],{'gold':current['gold'],'changes':Counter(),'failures':Counter()})
            case['changes'][label]+=1
            if not current_pass:
                case['failures'][current['raw']]+=1
                failures[(current['gold'],current['raw'])]+=1
        if sum(changes.values())!=spec['matched_rows']:
            raise ValueError('Wrong matched coverage')
        matrices[spec['name']]={'raw_rows':len(before),'matched_rows':sum(changes.values()),
            'before_correct':changes['1->1']+changes['1->0'],
            'after_correct':changes['1->1']+changes['0->1'],'changes':dict(changes),'cases':cases,
            'after_failure_outputs':[{'gold':gold,'raw':raw,'count':n} for (gold,raw),n in sorted(failures.items())]}
    return {'registration':registration,'matrices':matrices,
        'scope':'All registered raw outputs paired by case/condition/noise/query. Exact tokens plus immediate EOS remain the only success criterion. Failures are preserved verbatim; no semantic rescoring or outcome selection.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(json.loads(args.registration.read_bytes()),args.root)
    args.output.write_bytes((json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8'))
    print(json.dumps({name:{k:v for k,v in value.items() if k not in ('cases','after_failure_outputs')}
                      for name,value in result['matrices'].items()}))
