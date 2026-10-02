from collections import Counter
from scripts.experiments import author_prefeval_attribute_scenarios as author
from scripts.experiments import prefeval_attribute_generalization as x


def test_attribute_cases_are_counterfactual_and_complete():
    source,cases,audit=author.build()
    assert len(cases)==28 and sum(map(len,cases.values()))==112
    for vid,items in cases.items():
        assert len(items)==4 and len({i['id'] for i in items})==4
        for left,right in ((items[0],items[1]),(items[2],items[3])):
            assert left['situation']==right['situation']
            assert left['correct_option']!=right['correct_option']
            assert left['proposals']!=right['proposals']
        assert audit[vid]['authoring_inputs'].startswith('sanitized preference')
    for contrast in source['overwrite_contrasts']:
        for before,after in zip(cases[contrast['before']],cases[contrast['after']]):
            assert before['situation']==after['situation']
            assert before['proposals']==after['proposals']
            assert before['correct_option']!=after['correct_option']


def test_training_schedule_is_balanced():
    specs=[x.schedule(i) for i in range(64)]
    assert Counter(v['bank'] for v in specs)=={0:32,1:32}
    assert Counter(v['format'] for v in specs)=={'full-action':32,'xml':32}
    assert Counter(v['case'] for v in specs)=={0:16,1:16,2:16,3:16}
    assert Counter(v['rotation'] for v in specs)=={0:16,1:16,2:16,3:16}
    assert len({(v['case'],v['bank'],v['format'],v['rotation']) for v in specs})==64
