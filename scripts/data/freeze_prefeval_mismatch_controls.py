"""Outcome-independent donors: same split/topic, different registered semantic group."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vision_memory.prefeval.official_ab import records,sha

report=ROOT/'reports/prefeval-official-alignment-20260923'
manifest=ROOT/'reports/prefeval-rgb-20260917/registered/manifest.json'
registered=json.loads(manifest.read_text(encoding='utf-8'))['records']
rows=records(report)
donors={}
for row in rows:
    group=registered[row['id']]['semantic_group']
    eligible=[v for v in rows if v['split']==row['split']
              and registered[v['id']]['semantic_group']!=group
              and v['options'][0]!=row['options'][0]]
    same_topic=[v for v in eligible if v['benchmark']['topic']==row['benchmark']['topic']]
    candidates=same_topic or eligible
    donor=next((v for v in candidates if v['id']>row['id']),candidates[0])
    donors[row['id']]=dict(donor=donor['id'],semantic_group=group,
        donor_semantic_group=registered[donor['id']]['semantic_group'],
        same_topic=bool(same_topic),split=row['split'])
result=dict(status='fixed_before_student_evaluation',manifest_sha256=sha(manifest),
    policy='Prefer same topic and split; exclude identical semantic groups and identical correct option text. No outcomes used.',
    limit='Different groups need not be contradictory. This control tests memory dependence, not universal negative-answer correctness.',
    donors=donors)
(report/'pilot-mismatch-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(count=len(donors),same_topic=sum(v['same_topic'] for v in donors.values()))))
