import copy
from collections import Counter
import pytest
from scripts.experiments import author_prefeval_compositional_evidence as a
from scripts.experiments import prefeval_compositional_evidence as e
from scripts.experiments import prefeval_attribute_generalization as r


def test_component_edits_and_shared_contrast_questions():
    source,cases,audits=a.build()
    for contrast in source['contrasts']:
        for left,right in zip(cases[contrast['before']],cases[contrast['after']]):
            for rotation in range(4):
                for render in (e.q.xml_candidates,e.s.ranking_candidates):
                    l,lc,lg=render(left,rotation); rr,rc,rg=render(right,rotation)
                    assert l['query']==rr['query'] and lc==rc and lg!=rg
    broken=copy.deepcopy(cases)
    vid=next(iter(broken));broken[vid][1]['primitive_facts']=broken[vid][0]['primitive_facts']
    with pytest.raises(AssertionError):a.validate(source,broken,audits)


def test_only_application_content_changes_and_exact_draws():
    _,payload=r.q.load();_,cases,_=a.build()
    schedule=Counter((e.schedule(t)['case'],e.schedule(t)['format'],e.schedule(t)['rotation'])
                     for t in range(64) if e.schedule(t)['bank']==1)
    assert len(schedule)==32 and set(schedule.values())=={1}
    counts=Counter()
    for target in payload['targets'].values():
        for step in range(64):
            actual=e.jobs(target,step,'E',cases)
            assert actual==r.jobs(target,step,'D',cases)
            rec,apps=actual
            old_rec,old_apps=r.jobs(target,step,'R',cases)
            assert rec==old_rec
            assert [v[1] for v in apps]==[v[1] for v in old_apps]
            if not e.schedule(step)['bank']:assert apps==old_apps
            counts['recovery']+=len(rec);counts['application']+=4*len(apps)
    assert counts=={'recovery':16896,'application':21504}


def test_reject_value_specific_rotation_for_overwrite():
    source,cases,audits=a.build();contrast=source['contrasts'][0]
    cases[contrast['after']][0]['base_rotation']=(cases[contrast['after']][0]['base_rotation']+1)%4
    with pytest.raises(AssertionError):a.validate(source,cases,audits)
