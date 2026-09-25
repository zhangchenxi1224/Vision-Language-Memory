"""Freeze only explicitly reviewed neutral replies, before any Writer fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records,sha

V2='Thank you for telling me. I will keep your stated preference in mind.'


def main(args):
    approved=set(args.approved_reply)
    assert approved, 'An explicit review allowlist is required'
    summaries={}
    for split,expected in [('train',730),('dev',90)]:
        rows=load_records(split)
        assert len(rows)==expected
        items={}
        counts=Counter()
        for row in rows:
            pid=row['base_pair_id']
            path=args.input/(pid.replace(':','_')+'.json')
            source=json.loads(path.read_text())
            reply=source['generated']['raw']
            assert source['pair_id']==pid
            assert source['preference']==row['history'][0]['content']
            assert not source['generated']['truncated']
            assert reply in approved, f'Unreviewed reply: {pid}: {reply!r}'
            variants=[row['history'][1]['content'],reply,V2]
            assert V2 not in variants[:2]
            counts[reply]+=1
            items[pid]={'preference':source['preference'],'variants':variants,'raw_ack_sha256':sha(path)}
        artifact={'review_complete':True,'split':split,'count':expected,
            'review':'Exact reply allowlist reviewed for neutral acknowledgment; no MCQ output used.',
            'approved_replies':sorted(approved),'training_variants':[0,1] if split=='train' else [],
            'held_out_variant':2,'V1_protocol':'Frozen Reader, preference only, neutral acknowledgment, greedy, 48-token cap.',
            'V2_protocol':'Prospectively fixed neutral sentence; report only, no selection.',
            'items':items}
        target=args.output/f'variants-{split}.json'
        text=json.dumps(artifact,ensure_ascii=False,indent=2)+'\n'
        args.output.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert target.read_text(encoding='utf-8')==text, 'Frozen artifact differs'
        else:
            target.write_text(text,encoding='utf-8')
        summaries[split]={'count':expected,'replies':dict(counts),'sha256':sha(target)}
    summary={'reviewed_variants':summaries,'held_out_V2':V2}
    (args.output/'variants-review.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approved-reply',action='append',required=True)
    main(p.parse_args())
