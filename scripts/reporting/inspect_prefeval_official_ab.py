"""Compact read-only progress for the fixed official A/B run."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
result={}
for arm in ('A','B'):
    folder=root/'teachers'/arm
    completed=list(folder.glob('*/complete.json'))
    pending=[]
    for path in folder.glob('*/optimization.jsonl'):
        if (path.parent/'complete.json').exists():continue
        lines=path.read_text().splitlines()
        if lines:
            row=json.loads(lines[-1]);pending.append(dict(id=path.parent.name,step=row['step'],loss=row['loss']))
    result[arm]=dict(teacher_complete=len(completed),teacher_running=pending)
    for stage in ('write','retain'):
        fm=root/'writers'/arm/stage
        log=fm/'optimization.jsonl'
        if log.exists():
            row=json.loads(log.read_text().splitlines()[-1])
            result[arm][stage]=dict(step=row['step'],complete=(fm/'complete.json').exists())
result['failures']=[p.name for p in root.glob('failure-*.json')]
print(json.dumps(result,ensure_ascii=False))
