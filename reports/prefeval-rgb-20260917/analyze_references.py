"""Reconstruct immutable strict scores; formatting diagnostics do not replace them."""
import json
from pathlib import Path
from collections import defaultdict, Counter
import re

HERE = Path(__file__).resolve().parent


def norm(text):
    return ' '.join(text.casefold().split()).rstrip('.?!')


def strip_quotes(text):
    text = text.strip()
    if len(text) >= 2 and (text[0],text[-1]) in [('"','"'),("'","'"),('“','”')]:
        text = text[1:-1]
    return text


def main():
    rows = [json.loads(s) for path in sorted((HERE/'references-v1').glob('shard-*/reads.jsonl'))
            for s in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2048 and len({r['key'] for r in rows}) == len(rows)
    counts = defaultdict(lambda: [0,0])
    failure_types = Counter()
    compact=[]
    for row in rows:
        job,result,score=row['job'],row['result'],row['score']
        key='/'.join((row['condition'],job['split'],job['kind']))
        counts[key][0]+=int(score['strict_correct']);counts[key][1]+=1
        diagnostic=None
        if row['condition']=='text' and not score['strict_correct']:
            raw=result['raw']
            if job['kind']=='official_mcq':
                letters=set(re.findall(r'<([ABCD])>',raw))
                diagnostic=('wrong_tag_but_correct_letter' if len(letters)==1 and
                    next(iter(letters))=='ABCD'[job['target_index']] else 'other_mcq_failure')
            else:
                prefix=job['scope'].replace('_',' ')+': '
                edited=raw[len(prefix):] if raw.lower().startswith(prefix.lower()) else raw
                diagnostic=('scope_prefix_or_quote_only' if norm(strip_quotes(edited))==norm(strip_quotes(job['target']))
                            else 'other_recovery_failure')
            failure_types[diagnostic]+=1
        compact.append({'key':row['key'],'condition':row['condition'],'split':job['split'],
            'kind':job['kind'],'semantic_groups':job['groups'],'state':job['state'],
            'query':job['query'],'target':job.get('target',job.get('target_index')),
            'raw':result['raw'],'eos_reached':result['eos_reached'],
            'strict_correct':score['strict_correct'],'diagnostic_only':diagnostic})
    summary={'status':'first_empirical_gate_failed','rows':len(rows),'strict_counts':dict(counts),
        'text_failure_diagnostics':dict(failure_types),
        'allocation_gate':{'official_mcq':.8,'derived_recovery':.9,'passed':False},
        'interpretation':'Most failures may be formatting; diagnostic categories do not amend strict scores. No teacher or shared Writer training launched.',
        'next_review':'Restore the exact upstream MCQ prompt example and remove ambiguity in scope/value and literal quotation formatting; preregister and rerun the same references before teacher training.'}
    (HERE/'references-v1-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (HERE/'references-v1-compact.jsonl').open('w',encoding='utf-8') as f:
        for r in compact:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
