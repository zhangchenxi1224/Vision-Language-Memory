"""Aggregate held-out-template KL per history before paired bootstrapping."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_context_readout import ENDPOINTS,expected_keys,validate_protocol
from scripts.reporting.context_coverage_report import paired_interval
from scripts.experiments.prefeval_k1_data import sha


def summarize(rows,ids,spec):
    validate_protocol(spec)
    table={}
    for row in rows:
        key=tuple(row[k] for k in ('pair_id','query_id','endpoint','control'))
        if key in table or not math.isfinite(row['kl']):raise ValueError('Duplicate or nonfinite audit row')
        table[key]=row
    if set(table)!=expected_keys(ids,spec['queries']):raise ValueError('Incomplete readout denominator')
    queries=defaultdict(list)
    for q in spec['queries']:queries[q['family']].append(q['id'])
    queries['all']=[q['id'] for q in spec['queries']]
    result={'scope':spec['scope'],'independent_n':len(ids),'rows':len(rows),'families':{},'contrasts':{},
            'metric':'Mean token KL on identical teacher-generated prefixes; not free-generation factual accuracy',
            'promotion':'none; no shared Writer/new-history evidence'}
    def values(endpoint,control,qids):
        return [sum(table[pid,q,endpoint,control]['kl'] for q in qids)/len(qids) for pid in ids]
    for family,qids in queries.items():
        blank=values('blank','blank',qids);out={}
        for endpoint in ENDPOINTS:
            memory=values(endpoint,'memory',qids);mismatch=values(endpoint,'mismatch',qids)
            out[endpoint]={'memory_kl':sum(memory)/len(ids),'mismatch_kl':sum(mismatch)/len(ids),
                'blank_minus_memory':paired_interval([b-m for b,m in zip(blank,memory)]),
                'mismatch_minus_memory':paired_interval([b-m for b,m in zip(mismatch,memory)])}
        result['families'][family]={'blank_kl':sum(blank)/len(ids),'endpoints':out}
        for mode in ('history_hard','prompt_matching'):
            original=values('original/'+mode,'memory',qids)
            final=values('diverse/'+mode+'/teachers','memory',qids)
            selected=values('diverse/'+mode+'/selected','memory',qids)
            result['contrasts'][family+'/'+mode+'/coverage_kl_reduction']=paired_interval([a-b for a,b in zip(original,final)])
            result['contrasts'][family+'/'+mode+'/selection_kl_reduction']=paired_interval([a-b for a,b in zip(final,selected)])
    return result


def main(args):
    ids=json.loads(args.ids_file.read_text())['ids'];spec=json.loads(args.protocol.read_text());rows=[]
    for shard in range(2):
        done=json.loads((args.run/f'finished-{shard}.json').read_text())
        identity=args.run/f'identity-{shard}.json';data=args.run/f'readout-{shard}.jsonl'
        if done['identity_sha256']!=sha(identity) or done['readout_sha256']!=sha(data):raise ValueError('Audit receipt changed')
        ident=json.loads(identity.read_text())
        if ident['protocol_sha256']!=sha(args.protocol) or ident['ids_sha256']!=sha(args.ids_file):raise ValueError('Protocol changed')
        part=[json.loads(s) for s in data.read_text().splitlines()]
        if len(part)!=done['rows']:raise ValueError('Receipt denominator mismatch')
        rows.extend(part)
    result=summarize(rows,ids,spec)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'completed','rows':result['rows'],'independent_n':result['independent_n'],'report':str(args.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','ids-file','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
