"""Compare already archived SFT/benchmark inputs without reading any scores."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vision_memory.prefeval.official_ab import records,writer_event


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,default=ROOT/'reports/prefeval-official-alignment-20260923')
    a=p.parse_args();histories={}
    for line in gzip.open(a.report/'reference-and-rollout-evidence.jsonl.gz','rt',encoding='utf-8'):
        row=json.loads(line)
        if row['path'].startswith('benchmark-history/'):
            histories[row['data']['id']]=row['data']['history']
    rows=[]
    for r in records(a.report):
        if r['split']!='train':continue
        history=histories[r['id']]
        rows.append(dict(id=r['id'],disclosure_same=r['history'][0]==history[0],ack_same=r['history'][1]==history[1],
            writer_initial_exchange_same=writer_event(r,0)==writer_event(dict(r,history=history),0)))
    result=dict(scope='all64 fixed train-content states; inputs only; no answers or OOD scores used',count=len(rows),
        same_counts={k:sum(row[k] for row in rows) for k in ('disclosure_same','ack_same','writer_initial_exchange_same')},records=rows)
    (a.report/'writer-input-condition-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
