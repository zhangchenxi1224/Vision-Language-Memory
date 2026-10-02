"""Failure descriptions retain the original strict scores and all targets."""
import json
from pathlib import Path
import statistics
from collections import defaultdict,Counter
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from vision_memory.reader.open_answer import score_short_answer,normalize_short_answer

p=Path(__file__).resolve().parent
m=json.loads((p/'registered/manifest.json').read_text(encoding='utf-8'))
run=p/'visual-recovery-v1-run/sentinel-native-gray-r2'
counts=defaultdict(lambda:[0,0]);categories=Counter();failures=[];costs=[];losses=defaultdict(list)
for sid in m['sentinel_targets']:
    r=json.loads((run/sid/'result.json').read_text(encoding='utf-8'))
    trace=[json.loads(s) for s in (run/sid/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
    costs.append(r['seconds'])
    final3=[sum(x['weight']*(x['answer_ce']+x['eos_ce']) for x in t['losses']) for t in trace[-3:]]
    losses[str(r['capacity'])].append(dict(target=sid,last_three_update_losses=final3))
    for row in r['rows']:
        q=row['query']
        if q['kind']!='recovery':continue
        raw=row['generation']['raw'];correct=score_short_answer(raw,q['target'])['strict_correct'] and row['generation']['eos_reached']
        k=str(r['capacity']);counts[k][0]+=correct;counts[k][1]+=1
        if correct:continue
        normal=normalize_short_answer(raw)
        other=[scope for scope,value in r['state'].items() if scope!=q['scope'] and value is not None and normalize_short_answer(value)==normal]
        if other:category='exact_value_of_other_active_slot'
        elif normal==normalize_short_answer('no active preference'):category='incorrect_absence'
        elif q['target']=='no active preference':category='cleared_slot_not_absent'
        else:category='other_full_string_failure'
        categories[category]+=1
        failures.append(dict(target=sid,capacity=r['capacity'],scope=q['scope'],form=q['form'],expected=q['target'],raw=raw,
            category=category,matching_other_slots=other,eos=row['generation']['eos_reached']))
result=dict(recovery_read_counts=dict(counts),failure_categories=dict(categories),failures=failures,
    final_three_update_losses=dict(losses),latent_optimizer_updates=40*256,
    summed_target_optimization_and_read_seconds=sum(costs),
    cost_scope='post-model-load per-target optimization plus qualification, excludes startup; not billed GPU time',
    inference='Wrong-slot reads despite low training-view loss motivate a read-only train-vs-heldout / float-vs-PNG diagnostic; neither cause is established yet.')
(p/'visual-failure-analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('failures','final_three_update_losses')},indent=2))
