"""Describe frozen raw failures without changing the registered strict gate."""
import json,re
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[2]
report=ROOT/'reports/prefeval-rgb-20260917'
output=report/'semantic-transfer-v1-run'
gate=json.loads((output/'feasibility-verified.json').read_text(encoding='utf-8'))
counts=defaultdict(Counter);examples=defaultdict(list)
for path in sorted((output/'feasibility').glob('shard-*/reads.jsonl')):
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);q=r['query'];g=r['generation'];raw=g['raw'];target=q['target']
        if r['score']['strict_correct']:kind='registered_strict_correct'
        elif re.fullmatch(r'[ABCD]\. '+re.escape(target),raw) and g['eos_reached']:
            kind='exact_correct_action_with_forbidden_label'
        else:kind='other_strict_failure'
        counts[r['condition']][kind]+=1
        if len(examples[(r['condition'],kind)])<8:
            examples[(r['condition'],kind)].append(dict(target=r['target'],query_id=q['id'],expected=target,raw=raw,eos=g['eos_reached']))
result=dict(gate=gate['passed'],registered_text=gate['text'],registered_blank=gate['blank'],
    diagnostic_only=True,note='Label-prefix categories describe failures; they do not rescore or pass the gate.',
    categories={k:dict(v) for k,v in counts.items()},examples={str(k):v for k,v in examples.items()})
(report/'semantic-feasibility-failure-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
