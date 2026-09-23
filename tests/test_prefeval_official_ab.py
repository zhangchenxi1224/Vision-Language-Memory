import ast
from collections import Counter
from pathlib import Path
import pytest
from vision_memory.prefeval.official_ab import MCQ_TEMPLATE, training_item, writer_event

def test_official_mcq_template():
    source=Path('.cache/prefeval-upstream/utils/utils_mcq.py')
    if not source.exists(): pytest.skip('local upstream checkout unavailable')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'format_options','get_mcq_question_format'}]
    env={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),env)
    options=['one','two','three','four']
    assert MCQ_TEMPLATE.format(options=env['format_options'](options))==env['get_mcq_question_format'](options)

def test_balanced_forms_labels_and_dev_isolation():
    r=dict(id='sample',split='train',options=['correct','wrong1','wrong2','wrong3'],answer='official complete answer')
    forms=dict(T1='original',T2='paraphrase',T3='instruction',O1='never train one',O2='never train two')
    items=[training_item(r,forms,'B',step) for step in range(288)]
    assert set(Counter((x['family'],x['target']) for x in items).values())=={24}
    for x in items:
        gold='ABCD'.index(x['target'][8])
        assert x['order'][gold]==0
        assert 'never train' not in x['query']
    assert training_item(r,forms,'A',0)['target']==r['answer']
    with pytest.raises(ValueError): training_item(dict(r,split='dev'),forms,'A',0)

def test_writer_only_sees_current_exchange():
    r=dict(history=[dict(role='user',content='current disclosure'),dict(role='assistant',content='ack'),
                    dict(role='user',content='next event'),dict(role='assistant',content='next ack')],
           question='future question',answer='future target',options=['gold'])
    assert writer_event(r)=='user: current disclosure\nassistant: ack'
    assert writer_event(r,1)=='user: next event\nassistant: next ack'
