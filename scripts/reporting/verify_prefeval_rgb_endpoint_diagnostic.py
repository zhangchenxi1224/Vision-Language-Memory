"""Reconstruct fixed-endpoint diagnostic counts and paired effects from raw reads."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.probes.prefeval_rgb_endpoint_diagnostic import (load_json,file_sha,CONDITIONS,REPORT,
    recovery_queries,application_queries,score_generation)
from vision_memory.prefeval.rgb_protocol import digest
from vision_memory.reader.open_answer import normalize_short_answer

def exact_index(rows,key,expected):
    indexed={key(r):r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!=set(expected):raise ValueError('Missing, extra or duplicate diagnostic cell')
    return indexed

def main(registration,source,output,result_path):
    p=load_json(registration);m=load_json(REPORT/'registered/manifest.json');o=load_json(REPORT/'reader-format-v2.json')
    assert digest(m)==p['manifest_digest'] and digest(o)==p['overlay_digest']
    assert set(p['targets'])==set(m['sentinel_targets'])
    counts=defaultdict(lambda:[0,0]);complete=defaultdict(lambda:[0,0]);mcqcounts=defaultdict(lambda:[0,0])
    groups=defaultdict(lambda:[0,0]);pairs=defaultdict(Counter);attribution=[];anomalies=[];nreads=nmcq=nanchor=0
    mechanisms=Counter();failtypes=Counter();image_differences={}
    for sid,t in p['targets'].items():
        folder=output/sid;receipt=load_json(folder/'complete.json');old=load_json(source/sid/'result.json')
        assert receipt['optimizer_updates']==0 and file_sha(source/sid/'result.json')==t['result_sha']
        for name,h in t['artifacts'].items():assert file_sha(source/sid/name)==h
        for name,h in receipt['artifacts'].items():assert file_sha(folder/name)==h
        assert t['recovery']==recovery_queries(o,sid) and t['mcq']==application_queries(m,o,sid)
        rows=[json.loads(s) for s in (folder/'recovery.jsonl').read_text(encoding='utf-8').splitlines()]
        indexed=exact_index(rows,lambda r:(r['condition'],r['query']['query_id']),
            [(c,q['query_id']) for c in CONDITIONS for q in t['recovery']])
        archive={(r['query']['scope'],r['query']['form']):r['generation'] for r in old['rows'] if r['query']['kind']=='recovery'}
        k=len(t['state']);nreads+=len(rows)
        for (c,qid),r in indexed.items():
            q=next(q for q in t['recovery'] if q['query_id']==qid);assert r['query']==q and r['target']==sid
            g=r['generation'];assert len(g['generated_token_ids'])==g['generated_token_count']
            assert any(x in g['eos_token_ids'] for x in g['generated_token_ids'])==g['eos_reached']
            score=score_generation(g,q);assert score==r['score']
            ok=score['strict_correct'];part=q['partition'];scope=q['scope']
            for suffix in ('all',f'K{k}','changed' if scope==t['changed_scope'] else 'untouched',
                           'inactive' if t['state'][scope] is None else 'active'):
                key=f'{c}/{part}/{suffix}';counts[key][0]+=ok;counts[key][1]+=1
            ce=r['ce'];gold=ce['gold_token_ids']
            assert len(gold)==ce['answer_token_count']+1
            assert len(gold)==len(ce['per_token_nll'])==len(ce['gold_minus_alternative_margin'])==len(ce['gold_argmax_correct'])
            assert (g['generated_token_ids']==gold)==ce['generation_token_exact']
            if all(ce['gold_argmax_correct']) and not ce['generation_token_exact']:
                anomalies.append(dict(target=sid,condition=c,query_id=qid,first_divergent_token=ce['first_divergent_token']))
            if c=='png_cpu' and part=='heldout':
                a=archive[(scope,q['form'])];nanchor+=1
                assert g['generated_token_ids']==a['generated_token_ids'] and g['input_token_ids']==a['input_token_ids'] and g['raw']==a['raw'],f'Archived replay differs: {sid}/{qid}'
                if not ok:
                    other=[s for s,v in t['state'].items() if s!=scope and v is not None and normalize_short_answer(v)==normalize_short_answer(g['raw'])]
                    initial=load_json(source/t['predecessor']/'result.json')['state'] if t['predecessor'] else {}
                    stale=initial.get(scope) is not None and initial.get(scope)!=t['state'][scope] and normalize_short_answer(initial[scope])==normalize_short_answer(g['raw'])
                    category='truncated' if g['truncated'] else 'wrong_slot' if other else 'stale_value' if stale else 'incorrect_absence' if normalize_short_answer(g['raw'])=='no active preference' else 'other'
                    failtypes[category]+=1
                    flags={name:indexed[(cond,qid)]['score']['strict_correct'] for name,cond in
                        [('float_cpu_correct','float_cpu'),('png_cuda_correct','png_cuda'),('float_cuda_correct','float_cuda')]}
                    flags['png_training_all_correct']=all(indexed[('png_cpu',qq['query_id'])]['score']['strict_correct'] for qq in t['recovery'] if qq['scope']==scope and qq['partition']=='training')
                    mechanisms['+'.join(name for name,v in flags.items() if v) or 'none']+=1
                    attribution.append(dict(target=sid,query_id=qid,category=category,indicators=flags,raw=g['raw']))
        for c in CONDITIONS:
            for part in ('training','heldout'):
                qs=[q for q in t['recovery'] if q['partition']==part]
                all_ok=all(indexed[(c,q['query_id'])]['score']['strict_correct'] for q in qs)
                for suffix in ('all',f'K{k}'):
                    key=f'{c}/{part}/state/{suffix}';complete[key][0]+=all_ok;complete[key][1]+=1
                for scope in t['state']:
                    slot_ok=all(indexed[(c,q['query_id'])]['score']['strict_correct'] for q in qs if q['scope']==scope)
                    for suffix in ('all',f'K{k}'):
                        key=f'{c}/{part}/slot/{suffix}';complete[key][0]+=slot_ok;complete[key][1]+=1
        for a,b in [('float_cpu','png_cpu'),('float_cuda','png_cuda'),('float_cpu','float_cuda'),('png_cpu','png_cuda')]:
            for q in t['recovery']:
                ra,rb=[indexed[(c,q['query_id'])] for c in (a,b)]
                key=f'{a}->{b}/{q["partition"]}';v=pairs[key]
                v['total']+=1;v['token_changed']+=ra['generation']['generated_token_ids']!=rb['generation']['generated_token_ids']
                v[f'{int(ra["score"]["strict_correct"])}->{int(rb["score"]["strict_correct"])}']+=1
        images=load_json(folder/'images.json');image_differences[sid]=images['paired_differences']
        for name,h in images['processor_artifacts'].items():assert file_sha(folder/name)==h
        mcqs=[json.loads(s) for s in (folder/'mcq.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'mcq.jsonl').exists() else []
        mi=exact_index(mcqs,lambda r:(r['condition'],r['query']['scope']),[(c,q['scope']) for c in ('png','text','blank') for q in t['mcq']]);nmcq+=len(mcqs)
        for (c,scope),r in mi.items():
            q=next(q for q in t['mcq'] if q['scope']==scope);assert r['query']==q
            assert t['state'][scope] is not None and t['state'][scope]==q['value']
            g=r['generation'];s=score_generation(g,q,True);assert r['score']==s
            expected_query=q['text_prefix']+q['query'] if c=='text' else q['query']
            assert expected_query in g['chat_prompt']
            for suffix in ('all','K1' if k==1 else 'multi'):
                key=f'{c}/{suffix}';mcqcounts[key][0]+=s['strict_correct'];mcqcounts[key][1]+=1
                groupkey=f'{key}/{q["semantic_group"]}';groups[groupkey][0]+=s['strict_correct'];groups[groupkey][1]+=1
    assert (nreads,nmcq,nanchor)==(1760,252,176)
    summary=dict(registration_digest=digest(p),recovery_reads=nreads,mcq_reads=nmcq,archive_replays=nanchor,
        counts=dict(counts),complete=dict(complete),paired_effects={k:dict(v) for k,v in pairs.items()},
        original_failure_indicators=attribution,overlapping_indicator_intersections=dict(mechanisms),
        original_failure_types=dict(failtypes),mcq_counts=dict(mcqcounts),mcq_semantic_group_occurrences=dict(groups),
        tf_argmax_generation_mismatches=sorted(anomalies,key=lambda a:(a['target'],a['condition'],a['query_id'])),
        image_differences=image_differences,optimizer_updates=0,
        cost_seconds=sum(load_json(output/f'complete-{i}.json')['seconds'] for i in range(4)))
    result_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('original_failure_indicators','image_differences','mcq_semantic_group_occurrences')},indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--registration',type=Path,default=REPORT/'endpoint-diagnostic-v1.json')
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--result',type=Path,required=True)
    a=ap.parse_args();main(a.registration,a.source,a.output,a.result)
