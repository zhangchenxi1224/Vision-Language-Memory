"""Summarize fixed RGB endpoints without inventing a natural-answer judge."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time


def summarize(root):
    cells=defaultdict(list); joint=defaultdict(list); paired={}; files=[]; pending=0
    complete_shards=[]
    for path in sorted((root/'evaluations').glob('*/*/*/*.json')):
        if path.name.startswith('complete-'):
            complete_shards.append(str(path.relative_to(root)));continue
        data=json.loads(path.read_text(encoding='utf-8'))
        phase,arm,stage=path.relative_to(root/'evaluations').parts[:3]
        files.append(str(path.relative_to(root)))
        key=(phase,arm,stage,data['split'],data['condition'],data['position'])
        mcq={row['family']:row for row in data['rows'] if row['task']=='mcq'}
        for family,row in mcq.items():
            cells[key+(family,)].append((bool(row['correct']),row['predicted'] is None))
            paired[(phase,arm,stage,data['id'],data['condition'],data['position'],data['seed'],family)]=bool(row['correct'])
        if {'O1','O2'}<=mcq.keys():
            joint[key].append(all(mcq[f]['correct'] for f in ('O1','O2')))
        pending+=sum(r['task']=='generation' and r['scoring']=='pending_official_judge' for r in data['rows'])
    table=[]
    for key,values in sorted(cells.items()):
        phase,arm,stage,split,condition,position,family=key
        table.append(dict(phase=phase,arm=arm,stage=stage,split=split,condition=condition,position=position,family=family,
            correct=sum(v[0] for v in values),n=len(values),accuracy=sum(v[0] for v in values)/len(values),
            format_failures=sum(v[1] for v in values)))
    ood=[]
    for key,values in sorted(joint.items()):
        phase,arm,stage,split,condition,position=key
        ood.append(dict(phase=phase,arm=arm,stage=stage,split=split,condition=condition,position=position,
            correct=sum(values),n=len(values),both_ood_correct=sum(values)/len(values)))
    # Seed0 provides a single paired observation per state for controls. Seed1
    # remains in the full matrix, not counted as another independent preference.
    comparisons=defaultdict(list)
    split_by_id={}
    for rel in files:
        d=json.loads((root/rel).read_text(encoding='utf-8'));split_by_id[d['id']]=d['split']
    for key,value in paired.items():
        phase,arm,stage,rid,condition,position,seed,family=key
        if phase!='students' or condition!='matched' or seed!=0:continue
        for control in ('blank','text','mismatched'):
            other=('students',arm,stage,rid,control,position,0,family) if control=='mismatched' else (
                'references','common',stage,rid,control,position,0,family)
            if other in paired:
                comparisons[(arm,stage,split_by_id[rid],position,family,control)].append((value,paired[other]))
    deltas=[]
    for key,values in sorted(comparisons.items()):
        arm,stage,split,position,family,control=key
        deltas.append(dict(arm=arm,stage=stage,split=split,position=position,family=family,control=control,n=len(values),
            matched_minus_control=sum(int(x)-int(y) for x,y in values)/len(values),
            matched_only_correct=sum(x and not y for x,y in values),control_only_correct=sum(y and not x for x,y in values)))
    return dict(created_unix=time.time(),source=str(root),status='partial_until_all_declared_shards_complete',
        mcq_matrix=table,same_png_ood_joint=ood,paired_seed0_controls=deltas,
        generation_pending_official_judge=pending,evaluation_files=len(files),completed_shards=complete_shards,
        limits=['No automatic method selection from OOD or MCQ proxy.',
                'Teacher results are training-content fit; dev students measure new-preference transfer.',
                'Matched matrix includes two correlated noise seeds; paired comparisons use seed0 only.',
                'No official generation accuracy until the external judge completes.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=summarize(a.run)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.write_text(text,encoding='utf-8')
    else:print(text)
