"""Reconstruct reference scores from raw text/tokens, never stored correctness."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.experiments.register_prefeval_reader_v2 import original_jobs
from scripts.eval.prefeval_rgb import mcq_score, load_overlay
from vision_memory.reader.open_answer import score_short_answer
from vision_memory.prefeval.rgb_protocol import digest

def verify(manifest, directory, overlay=None):
    jobs = original_jobs(manifest) if overlay is None else overlay['reference_jobs']+overlay['supplement_jobs']
    expected={condition+':'+j['id']:(condition,j) for j in jobs for condition in ('blank','text')}
    seen=set(); counts=defaultdict(lambda:[0,0]); breakdown=defaultdict(lambda:[0,0]); files={}
    summaries=defaultdict(lambda:[0,0]); failures=[]; compact=[]
    for path in sorted(directory.glob('shard-*/reads.jsonl')):
        files[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
        identity=json.loads(path.with_name('identity.json').read_text())
        assert identity['manifest_sha']==digest(manifest)
        if overlay: assert identity['overlay_sha']==digest(overlay)
        for key,pair in json.loads(path.with_name('result.json').read_text())['counts'].items():
            summaries[key][0]+=pair[0];summaries[key][1]+=pair[1]
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);key=row['key'];assert key not in seen and key in expected
            seen.add(key);condition,job=expected[key]
            assert row['condition']==condition and row['job']==job, key
            result=row['result'];tokens=result['generated_token_ids'];eosids=result['eos_token_ids']
            assert eosids==[151645,151643], (key,eosids)
            eos=any(t in eosids for t in tokens)
            assert result['generated_token_count']==len(tokens)<=128
            assert result['eos_reached']==eos and result['truncated']==(not eos)
            assert result['prompt_token_count']==len(result['input_token_ids'])
            assert result['finish_reason']==('eos' if eos else 'token_limit' if len(tokens)>=128 else 'other_stop')
            score=mcq_score(result['raw'],job['target_index']) if job['kind']=='official_mcq' else score_short_answer(result['raw'],job['target'])
            correct=bool(score['strict_correct'] and eos)
            assert correct==row['score']['strict_correct'],key
            group='/'.join((condition,job['split'],job['kind']))
            counts[group][0]+=correct;counts[group][1]+=1
            for tag in ('family/'+job.get('family',job['kind']),
                        'capacity/'+str(len(job['state'])),
                        'status/'+('cleared' if job.get('scope') in job['state'] and job['state'][job['scope']] is None else 'active')):
                b=condition+'/'+tag;breakdown[b][0]+=correct;breakdown[b][1]+=1
            compact.append(dict(key=key,job=job,raw=result['raw'],eos=eos,correct=correct))
            if condition=='text' and not correct:failures.append(compact[-1])
    assert seen==set(expected),(len(seen),len(expected),set(expected)-seen)
    assert dict(counts)==dict(summaries),(dict(counts),dict(summaries))
    mcq=[sum(v[i] for k,v in counts.items() if k.startswith('text/') and k.endswith('/official_mcq')) for i in (0,1)]
    recovery=[sum(v[i] for k,v in counts.items() if k.startswith('text/') and k.endswith('/derived_recovery')) for i in (0,1)]
    families={k:v for k,v in breakdown.items() if k.startswith('text/family/') and k.split('/')[-1] not in ('official_mcq','derived_recovery')}
    gate=mcq[0]>=77 and mcq[1]==96 and recovery[0]>=836 and recovery[1]==928 and all(v[0]/v[1]>=.9 for v in families.values())
    return dict(verified=True,reads=len(seen),counts=dict(counts),breakdown=dict(breakdown),
        mcq=mcq,recovery=recovery,supplement_families=families,gate_passed=gate,files=files,
        failures=failures,scoring='raw reconstruction; unchanged full-string/XML plus observed EOS'),compact

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--overlay',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads((ROOT/'reports/prefeval-rgb-20260917/registered/manifest.json').read_text(encoding='utf-8'))
    overlay=load_overlay(a.overlay,m) if a.overlay else None
    report,compact=verify(m,a.directory,overlay)
    archive=a.directory.with_name(a.directory.name+'-evidence.tgz')
    if archive.exists():report['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    a.output.with_suffix('.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in compact),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','failures','breakdown')},indent=2))
