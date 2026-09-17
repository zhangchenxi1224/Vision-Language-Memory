"""Prospective, immutable query overlay; no model outputs inform construction."""
from __future__ import annotations
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from vision_memory.prefeval.rgb_protocol import (digest, queries, queries_v2,
    scope_name, text_prefix_v2, event_wordings)
from scripts.eval.prefeval_rgb import mcq_query, mcq_score
from vision_memory.reader.open_answer import score_short_answer, normalize_short_answer

REPORT = ROOT/'reports/prefeval-rgb-20260917'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def original_jobs(m):
    jobs = []
    for gid in m['pilot_train'] + m['pilot_dev']:
        r = m['records'][m['groups'][gid]['representative']]
        jobs.append(dict(id='mcq:'+gid, groups=[gid], split=r['split'],
            state={r['topic']:r['preference']}, query=mcq_query(r),
            target_index=r['target_index'], kind='official_mcq'))
    seen = set()
    for ep in m['episodes']:
        if ep.get('evaluation_only'): continue
        for tr in ep['transitions']:
            for q in queries(tr['state']):
                key = digest([tr['state'],q['query']])
                if key in seen: continue
                seen.add(key)
                jobs.append({**q,'id':key,'groups':ep['semantic_groups'],'split':ep['split'],
                    'state':tr['state'],'kind':'derived_recovery'})
    return jobs

def qualification(target, m):
    result = queries_v2(target['state'])
    pilot = set(m['pilot_train'])
    for scope,value in sorted(target['state'].items()):
        result.append(dict(scope=scope,kind='active_status',query=
            f'Is any preference currently active for {scope_name(scope)}? Answer yes or no only.',
            target='yes' if value is not None else 'no'))
        if value is not None:
            same = {r['semantic_group'] for r in m['records'].values()
                    if r['topic']==scope and r['preference']==value}
            alternatives = sorted((r['preference'],rid,r['semantic_group']) for rid,r in m['records'].items()
                if r['semantic_group'] in pilot-same and r['topic']==scope and r['preference']!=value)
            if not alternatives: raise ValueError('No pilot-only nonequivalent foil')
            foil,rid,gid = alternatives[0]
            for proposed,label,kind in ((value,'yes','positive_content'),(foil,'no','negative_content')):
                result.append(dict(scope=scope,kind=kind,query=
                    f'Is this the complete current stored preference statement for {scope_name(scope)}? '
                    f'Statement: {proposed}\nAnswer yes or no only.',target=label,
                    foil_record=rid if label=='no' else None,foil_group=gid if label=='no' else None))
    return result

def build(m, upstream):
    source = upstream.read_text(encoding='utf-8')
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef)
             and n.name in ('get_mcq_question_format','format_options')]
    assert len(funcs)==2
    namespace = {}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(upstream),'exec'),namespace)
    template = namespace['get_mcq_question_format']
    old = original_jobs(m)
    assert len(old)==1024
    jobs = []
    for job in old:
        if job['kind']=='official_mcq':
            r = m['records'][m['groups'][job['groups'][0]]['representative']]
            query = r['question'] + template(r['options'])
        else:
            query = next(q['query'] for q in queries_v2(job['state'])
                         if (q['scope'],q['form'])==(job['scope'],job['form']))
        jobs.append({**job,'query':query,'v1_id':job['id'],
                     'query_sha':digest(query),'text_prefix':text_prefix_v2(job['state']),
                     'family':job['kind'],'occurrences':[]})
    by_id={j['id']:j for j in jobs}
    for ep in m['episodes']:
        if ep.get('evaluation_only'): continue
        for tr in ep['transitions']:
            for q in queries(tr['state']):
                by_id[digest([tr['state'],q['query']])]['occurrences'].append(
                    dict(episode=ep['id'],ordinal=tr['ordinal'],scope=q['scope'],form=q['form'],
                         capacity=len(tr['state']),active_k=tr['active_k'],operation=tr['operation']))
    teachers={sid:dict(training=queries_v2(t['state'],training=True),qualification=qualification(t,m))
              for sid,t in m['targets'].items()}
    supplemental={}
    for sid in m['sentinel_targets']:
        state=m['targets'][sid]['state']
        for phase in ('training','qualification'):
            for q in teachers[sid][phase]:
                if phase=='qualification' and q['kind']=='recovery': continue
                key='supplement:'+digest([state,q['query']])
                family='training_recovery' if phase=='training' else q['kind']
                item=supplemental.setdefault(key,{**q,'id':key,'split':'train','state':state,
                    'kind':'sentinel_supplement','family':family,'text_prefix':text_prefix_v2(state),
                    'occurrences':[]})
                item['occurrences'].append(dict(target_id=sid,phase=phase,scope=q['scope'],form=q.get('form')))
    assert len(supplemental)*2<=1920
    baseline=[]
    for gid in m['pilot_train']+m['pilot_dev']:
        r=m['records'][m['groups'][gid]['representative']]
        state={r['topic']:r['preference']}
        baseline.append(dict(id='singleton:'+gid,split=r['split'],transitions=[dict(
            ordinal=0,state=state,event=event_wordings(r['topic'],r['preference'],'set')[0],
            operation='set',queries=[dict(query=by_id['mcq:'+gid]['query'],kind='official_mcq',
                target_index=r['target_index'])]+queries_v2(state))]))
    for ep in [e for e in m['episodes'] if e['panel']=='dev-k4'][:4]:
        baseline.append(dict(id=ep['id'],split=ep['split'],transitions=[dict(
            ordinal=t['ordinal'],state=t['state'],event=t['event_wordings'][0],operation=t['operation'],
            queries=queries_v2(t['state'])) for t in ep['transitions']]))
    assert sum(len(e['transitions']) for e in baseline)==124
    return dict(schema='prefeval.reader-format.v2',parent_manifest_digest=digest(m),
        parent_manifest_file_sha256=sha(REPORT/'registered/manifest.json'),membership_sha=m['membership_sha'],
        upstream=dict(commit=m['upstream_commit'],file='utils/utils_mcq.py',file_sha256=sha(upstream),
            template_sha256=hashlib.sha256(template(['{A}','{B}','{C}','{D}']).encode()).hexdigest()),
        scoring=dict(mcq_source_sha=digest(inspect.getsource(mcq_score)),
            recovery_source_sha=digest(inspect.getsource(score_short_answer)),
            normalization_source_sha=digest(inspect.getsource(normalize_short_answer)),
            primary='unchanged normalized full-string equality (MCQ: strict XML), plus observed EOS'),
        generation=dict(max_new_tokens=128,do_sample=False,num_beams=1,reader_resize=256),
        reference_jobs=jobs,supplement_jobs=list(supplemental.values()),teachers=teachers,
        baseline=dict(seed=20260917,episodes=baseline),
        gates=dict(mcq=[77,96],recovery=[836,928],supplement_family_rate=.9))

if __name__=='__main__':
    m=json.loads((REPORT/'registered/manifest.json').read_text(encoding='utf-8'))
    overlay=build(m,ROOT/'.cache/prefeval-upstream/utils/utils_mcq.py')
    out=REPORT/'reader-format-v2.json'
    text=json.dumps(overlay,ensure_ascii=False,indent=2)+'\n'
    if out.exists() and out.read_text(encoding='utf-8')!=text:
        raise ValueError('Refuse to replace an existing protocol')
    out.write_text(text,encoding='utf-8',newline='\n')
    seal=dict(file_sha256=sha(out),canonical_digest=digest(overlay),
              reference_reads=len(overlay['reference_jobs'])*2,supplement_reads=len(overlay['supplement_jobs'])*2)
    (REPORT/'reader-format-v2-seal.json').write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(seal,indent=2))
