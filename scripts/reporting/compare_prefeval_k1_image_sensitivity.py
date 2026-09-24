"""Compare complete T1 free answers under matched and mismatched saved PNGs."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records


def main():
    evidence=ROOT/'reports/prefeval-k1-l0-l2-20260924/evidence'
    sources={
        ('A','pilot'):'preretain-and-A-student-complete/student-A-write-T1-free/readback-0.jsonl',
        ('B','pilot'):'student-B-write-T1-complete/readback-0.jsonl',
        ('A','dev'):'student-A-dev-T1-free-complete/readback-0.jsonl',
        ('B','dev'):'student-B-dev-T1-complete/readback-0.jsonl',
    }
    result={'scope':'Complete write2048 T1 free outputs only; exact output sensitivity is not semantic memory accuracy',
            'unit':'Preference x noise chain; two chains are repeated measures, not independent preferences',
            'groups':{}}
    for (arm,split),name in sources.items():
        source=evidence/name
        rows=[r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines()) if r['task']=='free']
        expected={r['base_pair_id'] for r in load_records(split)}
        ix={(r['pair_id'],r['chain'],r['control']):r for r in rows}
        assert len(rows)==len(ix)==len(expected)*4
        assert set(ix)=={(pid,chain,c) for pid in expected for chain in [0,1] for c in ['memory','mismatch']}
        pairs=[]
        for pid in sorted(expected):
            for chain in [0,1]:
                m,w=ix[pid,chain,'memory'],ix[pid,chain,'mismatch']
                for key in ['reader_query','question','preference','family','prefix','max_new_tokens']:
                    assert m[key]==w[key]
                assert m['family']=='T1' and m['prefix']==0 and m['max_new_tokens']==300
                assert m['png_sha256']!=w['png_sha256'] and w['donor_pair_id']!=pid
                a,b=[r['generated']['generated_token_ids'] for r in [m,w]]
                assert a and b
                prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                pairs.append(dict(pair_id=pid,chain=chain,token_identical=a==b,
                    decoded_text_identical=m['generated']['raw']==w['generated']['raw'],
                    common_prefix_tokens=prefix,matched_tokens=len(a),mismatched_tokens=len(b),
                    matched_truncated=m['generated']['truncated'],mismatched_truncated=w['generated']['truncated']))
        changed=[p for p in pairs if not p['token_identical']]
        result['groups'][f'{arm}-{split}']={
            'preferences':len(expected),'paired_answers':len(pairs),
            'source':name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'token_identical_pairs':sum(p['token_identical'] for p in pairs),
            'decoded_text_identical_pairs':sum(p['decoded_text_identical'] for p in pairs),
            'changed_pairs':len(changed),
            'median_common_prefix_tokens_among_changed':statistics.median(p['common_prefix_tokens'] for p in changed) if changed else None,
            'matched_truncated':sum(p['matched_truncated'] for p in pairs),
            'mismatched_truncated':sum(p['mismatched_truncated'] for p in pairs),
            'pairs':pairs}
    baseline=evidence/'student-A-dev-T1-all-complete/shared-free-baselines.jsonl'
    base_rows=[json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines()]
    dev_ids={r['base_pair_id'] for r in load_records('dev')}
    assert len(base_rows)==180
    assert {(r['pair_id'],r['control']) for r in base_rows}=={(pid,c) for pid in dev_ids for c in ['blank','text']}
    assert all(r['task']=='free' and r['family']=='T1' and r['prefix']==0 and r['chain']==0 for r in base_rows)
    result['shared_dev_baselines']={'source':str(baseline.relative_to(evidence)),
        'source_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'controls':{}}
    for control in ['blank','text']:
        selected=[r for r in base_rows if r['control']==control]
        counts=[len(r['generated']['generated_token_ids']) for r in selected]
        result['shared_dev_baselines']['controls'][control]={'expected':90,'observed':len(selected),
            'truncated':sum(r['generated']['truncated'] for r in selected),'generated_token_range':[min(counts),max(counts)]}
    out=evidence/'write-T1-free-image-sensitivity.json'
    payload=json.dumps(result,indent=2)+'\n'
    if out.exists():assert out.read_text(encoding='utf-8')==payload, 'Existing result changed; inspect sources'
    else:out.write_text(payload,encoding='utf-8')
    print(json.dumps({key:{k:v for k,v in group.items() if k!='pairs'} for key,group in result['groups'].items()}))


if __name__=='__main__':main()
