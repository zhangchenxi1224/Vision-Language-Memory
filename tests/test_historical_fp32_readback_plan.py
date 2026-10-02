import json
from pathlib import Path
from scripts.experiments.prepare_historical_fp32_readback import target_spec


def test_all_original_entities_and_questions_preserved_with_two_new_queries():
    panel = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/r11_open_eos_multiquestion_targets.json').read_text())
    targets = [target_spec(t) for t in panel['targets']]
    assert len({t['entity'] for t in targets}) == 16
    for old, new in zip(panel['targets'], targets, strict=True):
        questions = new['question_variants']
        assert questions['original_open'] == old['inputs']['original_open']
        assert questions['paraphrase_1'] == old['inputs']['paraphrase_open']
        assert questions['paraphrase_2'] == old['inputs']['new_rewrite_open']
        assert len(set(questions.values())) == 5
        assert new['gold'] == old['scorer_metadata']['gold']


def test_mixed_updates_survive_but_their_questions_and_answer_choices_do_not():
    target = {'target_index': 0, 'semantic_group_id': 'one', 'topic': 'music', 'stratum': 'music:clear',
              'scorer_metadata': {'gold': 'no active preference'},
              'inputs': {'original_open': 'What is the current music preference for the desk?',
                         'paraphrase_open': 'Question two', 'new_rewrite_open': 'Question three'},
              'source_prefix': [{'type': 'event', 'event_kind': 'set', 'event_text': 'Save ambient.'},
                                {'type': 'mixed', 'event_kind': 'clear', 'event_text': 'Clear music.',
                                 'query': {'text': 'DO_NOT_COPY', 'choices': ['ANSWER_LABEL']}},
                                {'type': 'query', 'query': {'text': 'DO_NOT_COPY'}}]}
    events = target_spec(target)['event_stream']
    assert events == [{'event_kind': 'set', 'event_text': 'Save ambient.'},
                      {'event_kind': 'clear', 'event_text': 'Clear music.'}]
    assert 'DO_NOT_COPY' not in json.dumps(events) and 'ANSWER_LABEL' not in json.dumps(events)
