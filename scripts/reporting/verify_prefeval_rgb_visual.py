"""Reconstruct actual PNG/state outcomes, preserving every planned denominator."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from PIL import Image
import torch
from scripts.eval.prefeval_rgb import load_overlay,mcq_score
from scripts.experiments.prefeval_visual_policy import load_policy
from vision_memory.prefeval.rgb_protocol import digest,recovery_summary
from vision_memory.reader.open_answer import score_short_answer
from vision_memory.repro import canonical_tensor_sha256

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixel_sha(path):
    with Image.open(path) as im:
        assert im.mode=='RGB' and im.size==(1024,1024)
        return hashlib.sha256(np.asarray(im).tobytes()).hexdigest()

def score_rows(rows,expected,png):
    assert Counter(digest(r['query']) for r in rows)==Counter(digest(q) for q in expected)
    results=[]
    for row in rows:
        q=row['query'];g=row['generation'];tokens=g['generated_token_ids']
        assert g['eos_token_ids']==[151645,151643]
        eos=any(t in g['eos_token_ids'] for t in tokens)
        assert len(tokens)==g['generated_token_count']<=128 and eos==g['eos_reached']
        assert g['truncated']==(not eos)
        assert q['query'] in g['chat_prompt']
        assert 'Current memory values follow.' not in g['chat_prompt']
        assert row['png_sha256']==png
        s=mcq_score(g['raw'],q['target_index']) if q['kind']=='official_mcq' else score_short_answer(g['raw'],q['target'])
        s['strict_correct']=bool(s['strict_correct'] and eos)
        assert s['strict_correct']==row['score']['strict_correct']
        results.append({**row,'score':s})
    return results

def tally(counts,key,correct):
    counts[key][0]+=bool(correct);counts[key][1]+=1

def baseline(directory,o,policy):
    expected={(ep['id'],tr['ordinal']):(index,ep,tr) for index,ep in enumerate(o['baseline']['episodes']) for tr in ep['transitions']}
    found={};counts=defaultdict(lambda:[0,0]);episodes=defaultdict(list)
    for path in sorted(directory.glob('shard-*/writes.jsonl')):
        identity=read(path.with_name('identity.json'))
        assert identity['overlay_sha']==digest(o) and identity['qualification_policy_digest']==digest(policy)
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);key=(r['episode'],r['transition']['ordinal']);assert key not in found and key in expected
            index,ep,tr=expected[key];assert tr==r['transition'] and r['split']==ep['split']
            a=r['audit'];source=path.parent/r['source'];out=path.parent/r['output']
            assert a['source_file_sha']==sha(source) and a['source_pixels_sha']==pixel_sha(source)
            assert a['output_file_sha']==sha(out) and a['output_pixels_sha']==pixel_sha(out)
            assert a['trajectory_state_count']==29 and a['native_denoising_steps']==28
            seed=o['baseline']['seed']+index*100+tr['ordinal'];assert a['seed']==seed
            noise=torch.randn((1,4,128,128),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
            assert canonical_tensor_sha256(noise)==a['noise_sha']
            rows=score_rows(r['rows'],tr['queries'],sha(out))
            rec=[x for x in rows if x['query']['kind']=='recovery'];qs=[q for q in tr['queries'] if q['kind']=='recovery']
            s=recovery_summary(tr['state'],qs,rec,sha(out))
            group=('singleton/' if ep['id'].startswith('singleton:') else 'recurrent/')+ep['split']
            for row in rows:tally(counts,group+'/'+row['query']['kind'],row['score']['strict_correct'])
            tally(counts,group+'/complete_state',s['recovery_complete'])
            if not ep['id'].startswith('singleton:'):
                original=next(e for e in MANIFEST['episodes'] if e['id']==ep['id'])['transitions'][tr['ordinal']]
                for row in rec:
                    changed=row['query']['scope']==original['changed_scope']
                    tally(counts,'recurrent/'+('changed' if changed else 'untouched'),row['score']['strict_correct'])
                for scope in tr['state']:
                    tally(counts,'recurrent/complete_slot',all(x['score']['strict_correct'] for x in rec if x['query']['scope']==scope))
                if tr['operation']=='clear':tally(counts,'recurrent/selective_clear',s['recovery_complete'])
                if tr['operation']=='retain':tally(counts,'recurrent/post_clear_retain',s['recovery_complete'])
            found[key]=r;episodes[ep['id']].append((tr['ordinal'],s['recovery_complete']))
    missing=sorted(set(expected)-set(found))
    for ep in o['baseline']['episodes']:
        for tr in ep['transitions'][1:]:
            k=(ep['id'],tr['ordinal']);prev=(ep['id'],tr['ordinal']-1)
            if k in found and prev in found:
                assert found[k]['audit']['source_pixels_sha']==found[prev]['audit']['output_pixels_sha']
    per_ep={ep['id']:dict(complete=len(episodes[ep['id']])==len(ep['transitions']) and all(x[1] for x in episodes[ep['id']]),
        first_failure=next((i for i,ok in sorted(episodes[ep['id']]) if not ok),None)) for ep in o['baseline']['episodes']}
    return dict(expected_writes=124,observed_writes=len(found),missing=missing,counts=dict(counts),episodes=per_ep,
                whole_recurrent_episodes=[sum(v['complete'] for k,v in per_ep.items() if not k.startswith('singleton:')),4])

def sentinel(directory,m,o,policy):
    counts=defaultdict(lambda:[0,0]);aux=defaultdict(lambda:[0,0]);results={};technical=[]
    for sid in m['sentinel_targets']:
        target=m['targets'][sid];out=directory/sid;k=str(len(target['state']))
        if not (out/'result.json').exists():
            technical.append(sid);tally(counts,'K'+k,False);results[sid]=dict(recovery_complete=False,status='technical_or_blocked');continue
        r=read(out/'result.json');assert r['state']==target['state'] and r['qualification_policy_digest']==digest(policy)
        for name,value in r['artifacts'].items():assert sha(out/name)==value
        trace=[json.loads(s) for s in (out/'optimization.jsonl').read_text().splitlines()]
        assert [x['step'] for x in trace]==list(range(1,257))
        init=torch.load(out/'initial-latent.pt',map_location='cpu',weights_only=True)
        end=torch.load(out/'latent.pt',map_location='cpu',weights_only=True)
        assert end.dtype==torch.float32 and end.shape==(1,4,128,128) and torch.isfinite(end).all()
        assert canonical_tensor_sha256(init)==r['binding']['initialization_sha']
        if target['predecessor']:
            prev=torch.load(directory/target['predecessor']/'latent.pt',map_location='cpu',weights_only=True)
            assert torch.equal(init,prev)
        assert pixel_sha(out/'memory.png')
        rows=score_rows(r['rows'],o['teachers'][sid]['qualification'],sha(out/'memory.png'))
        s=recovery_summary(target['state'],o['teachers'][sid]['qualification'],rows,sha(out/'memory.png'))
        for field in s:assert s[field]==r[field]
        tally(counts,'K'+k,s['recovery_complete'])
        tally(counts,'active'+str(sum(v is not None for v in target['state'].values())),s['recovery_complete'])
        for family,pair in s['auxiliary_family_counts'].items():aux[family][0]+=pair[0];aux[family][1]+=pair[1]
        results[sid]=dict(**s,status='completed',png_sha256=sha(out/'memory.png'),predecessor=target['predecessor'],seconds=r['seconds'])
    chains={}
    selected=set(m['sentinel_targets'])
    for ep in m['episodes']:
        ids={t['target_state_id'] for t in ep['transitions']}
        if ep['panel']=='train-k4' and ids<=selected:
            chains[ep['id']]=all(results[sid]['recovery_complete'] for sid in ids)
    gate=not technical and all(counts['K'+k][0]>=threshold for k,threshold in policy['minimum_recovery_complete'].items())
    return dict(expected_states=40,completed=40-len(technical),technical_or_blocked=technical,
        complete_state_counts=dict(counts),auxiliary_counts=dict(aux),states=results,
        whole_teacher_chain_completeness=chains,allocation_gate_passed=gate)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=ROOT/'reports/prefeval-rgb-20260917'
    MANIFEST=read(root/'registered/manifest.json');o=load_overlay(root/'reader-format-v2.json',MANIFEST)
    policy=load_policy(root/'visual-recovery-allocation-v1.json',MANIFEST,o)
    result=dict(policy_digest=digest(policy),baseline=baseline(a.stage/'baseline',o,policy),
                sentinel=sentinel(a.stage/'sentinel',MANIFEST,o,policy))
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ('states','episodes')} if isinstance(v,dict) else v for k,v in result.items()},indent=2))
