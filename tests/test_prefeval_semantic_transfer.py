import inspect
from collections import Counter
import torch
from scripts.experiments import prefeval_semantic_transfer as s
from scripts.probes.prefeval_rgb_endpoint_diagnostic import load_json

def data():return load_json(s.DATA/'training-payload.json'),load_json(s.DATA/'evaluation-payload.json')

def test_training_queries_exclude_evaluation_questions_and_reserved_probes():
    train,evaluation=data();train_queries=set();excluded=set()
    for sid,t in train['targets'].items():
        train_queries.update(q['query'] for q in t['recovery'])
        train_queries.update(s.application_view(c,r)['query'] for c in t['applications'] for r in range(4))
        assert 'mcq' not in t['initial'] and 'recovery' not in t['initial']
        e=evaluation['targets'][sid]
        excluded.update(q['query'] for q in e['qualification'])
        excluded.update(q['query'] for q in e['application_reserved'])
        excluded.update(q['query'] for q in e['mcq'])
    assert not train_queries & excluded
    assert all('<stored_value>' not in q for q in train_queries)
    source=inspect.getsource(s.train)
    assert 'evaluation-payload' not in source and 'reserved-scenarios' not in source

def test_scenario_permutations_preserve_full_selected_action():
    scenarios=load_json(s.DATA/'training-scenarios.json')
    for cases in scenarios.values():
        assert len(cases)==4
        for c in cases:
            views=[s.application_view(c,r) for r in range(4)]
            assert len({v['query'] for v in views})==4 and len({v['target'] for v in views})==1
            for v in views:assert all(action in v['query'] for action in c['proposals'])
            assert c['target']==c['proposals'][c['correct_option']] and c['target'] not in ('A','B','C','D')

def test_clause_audit_preserves_sources_and_unique_compatibility():
    source=load_json(s.DATA/'authoring-source.json');audit=load_json(s.DATA/'clause-option-audit.json')
    train=load_json(s.DATA/'training-scenarios.json');probe=load_json(s.DATA/'reserved-scenarios.json')
    assert len(audit)==28 and sum(map(len,train.values()))==112 and sum(map(len,probe.values()))==56
    for v in source['values']:
        a=audit[v['id']];assert a['source']==v['value'] and a['scope']==v['scope']
        assert a['clauses'] and all(len(o['clause_satisfaction'])==len(a['clauses']) and o['reason'] for o in a['options'])
        valid=[i for i,o in enumerate(a['options']) if all(o['clause_satisfaction'])]
        assert valid==[a['correct_option']]
        for c in train[v['id']]+probe[v['id']]:assert c['correct_option']==valid[0]

def test_overwrite_contrasts_require_image_dependent_answer_change():
    source=load_json(s.DATA/'authoring-source.json')
    assert len(source['overwrite_contrasts'])==4
    for filename in ('training-scenarios.json','reserved-scenarios.json'):
        payload=load_json(s.DATA/filename)
        for c in source['overwrite_contrasts']:
            for before,after in zip(payload[c['before']],payload[c['after']]):
                for rotation in range(4):
                    a=s.application_view(before,rotation);b=s.application_view(after,rotation)
                    assert a['query']==b['query'] and a['target']!=b['target']

def test_exact_schedule_and_equal_slot_forward_counts():
    p,_=data();totals=Counter()
    for sid,t in p['targets'].items():
        per_scope={scope:Counter() for scope in t['state']}
        for draw in p['schedule']:
            for arm in ('A','B'):
                jobs=s.training_jobs(t,draw,arm);totals[arm]+=len(jobs)
                assert len(jobs)==len(t['state']) and abs(sum(w for _,w in jobs)-1)<1e-12
                for q,w in jobs:
                    if len(t['state'])>1:
                        assert w==(.5 if q['scope']==t['changed_scope'] else .5/(len(t['state'])-1))
                    if arm=='A' or not draw['mixture_application'] or t['state'][q['scope']] is None:assert q['kind']=='recovery'
                    else:
                        assert q['kind']=='application';per_scope[q['scope']][q['scenario'],q['rotation']]+=1
        for scope,value in t['state'].items():
            if value is None:assert not per_scope[scope]
            else:assert len(per_scope[scope])==16 and set(per_scope[scope].values())=={8}
    assert totals=={'A':22528,'B':22528}

def test_fixed_population_and_read_budgets():
    p,e=data();assert len(p['targets'])==len(e['targets'])==40
    assert sum(len(t['applications']) for t in p['targets'].values())==336
    for key,total in [('recovery_training',264),('qualification',432),('application_training',336),('mcq',84),('application_reserved',168)]:
        assert sum(len(t[key]) for t in e['targets'].values())==total
    assert 2*(264+432+336+84+168)+(336+168)+2*168==3408

def test_identical_latents_can_receive_fresh_independent_optimizers():
    endpoint=torch.tensor([1.,2.]);a=torch.nn.Parameter(endpoint.clone());b=torch.nn.Parameter(endpoint.clone())
    oa=torch.optim.Adam([a],lr=.05);ob=torch.optim.Adam([b],lr=.05)
    assert torch.equal(a,b) and not oa.state and not ob.state
    a.sum().backward();oa.step();assert torch.equal(b,endpoint) and not torch.equal(a,b) and not ob.state

def test_counting_processor_preserves_live_autograd():
    class Fake:
        def __call__(self,**kwargs):return {'input_ids':torch.ones(1,12,dtype=torch.long),'pixels':kwargs['image']*2}
    x=torch.ones(3,requires_grad=True);p=s.TokenCountProcessor(Fake());batch=p(image=x)
    batch['pixels'].sum().backward();assert p.last_input_tokens==12 and torch.equal(x.grad,torch.full_like(x,2))

def test_ranking_candidates_follow_display_and_target_text():
    p,_=data()
    for t in p['targets'].values():
        for case in t['applications']:
            for rotation in range(4):
                q,choices,gold=s.ranking_candidates(case,rotation)
                assert choices[gold]==q['target']
                for label,choice in zip('ABCD',choices):assert f'{label}. {choice}\n' in q['query']
                assert 'gold_index' not in q['query'] and 'target_index' not in q['query']

def test_cumulative_processor_counts_every_candidate():
    class Fake:
        def __call__(self,**kwargs):return {'input_ids':torch.ones(1,len(kwargs['text'][0]),dtype=torch.long)}
    p=s.TokenCountProcessor(Fake());p.begin_capture()
    for text in ('a','bb','ccc','dddd'):p(text=[text])
    calls=p.end_capture()
    assert p.total_calls==4 and p.total_input_tokens==10 and p.last_input_tokens==4
    assert [x['input_tokens'] for x in calls]==[1,2,3,4]

def test_ranking_budget_and_protocol_isolation():
    p,_=data();counts={'A':0,'B':0}
    for t in p['targets'].values():
        for draw in p['schedule']:
            for arm in counts:
                counts[arm]+=sum(4 if q['kind']=='application' else 1 for q,_ in s.training_jobs(t,draw,arm))
    assert counts=={'A':22528,'B':54784} and sum(counts.values())==77312
    import inspect
    code=inspect.getsource(s.ranking_load)+inspect.getsource(s.ranking_calibrate)
    assert 'evaluation-payload' not in code and 'reserved-scenarios' not in code and 'generated_token' not in code
    assert 'feasibility-verified.json' in inspect.getsource(s.train)

def test_trial_fixed_128_budget_and_coverage():
    reg,p=s.trial_load();totals=Counter();logical=0
    for t in p['targets'].values():
        coverage={scope:Counter() for scope in t['state']}
        for draw in p['schedule'][:reg['steps']]:
            for arm in ('A','B'):
                jobs=s.training_jobs(t,draw,arm);logical+=len(jobs)
                assert abs(sum(w for _,w in jobs)-1)<1e-12
                for q,w in jobs:
                    totals[arm]+=4 if q['kind']=='application' else 1
                    if arm=='B' and q['kind']=='application':coverage[q['scope']][q['scenario'],q['rotation']]+=1
                    if t['state'][q['scope']] is None:assert q['kind']=='recovery'
        for scope,value in t['state'].items():
            if value is not None:assert len(coverage[scope])==16 and set(coverage[scope].values())=={4}
    assert totals=={'A':11264,'B':27392} and logical==22528
    assert reg['additional_updates']==40*2*128==10240
    occurrences=sum(c['value_id'] in reg['contrast_values'] for t in p['targets'].values() for c in t['applications'])
    assert occurrences==96 and 1848+occurrences*3*3==2712
    assert 'evaluation-payload' not in inspect.getsource(s.trial_train)
    assert 'reserved-scenarios' not in inspect.getsource(s.trial_load)
    assert "gate['passed']" in inspect.getsource(s.train)
    assert "assert not opt.state" in inspect.getsource(s.trial_train)
    assert "training-payload.json" in inspect.getsource(s.trial_evaluate)


def test_trial_cpu_attribution_preserves_every_contrast_occurrence():
    reg,_=s.trial_load();a=load_json(s.TRIAL_DATA/'calibration-attribution.json')
    assert s.digest(a)==reg['attribution_digest'] and a['model_calls']==0
    assert len(a['rows'])==768 and len({(r['target'],r['value_id']) for r in a['rows']})==24
    assert len(a['overwrite_contrasts'])==4
    for r in a['rows']:
        assert r['gold_margin']==r['scores'][r['gold_index']]-max(x for j,x in enumerate(r['scores']) if j!=r['gold_index'])
        assert r['correct']==(r['gold_margin']>0)
