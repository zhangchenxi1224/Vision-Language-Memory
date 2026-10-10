import json
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.experiments import prefeval_aug4_teacher as teacher
from scripts.experiments import prefeval_aug4_evaluate as evaluation


def row(pid='topic:0001', topic='topic'):
    return {'base_pair_id': pid, 'topic': topic,
            'writer_user_forms': {key: f'PRIVATE PREFERENCE {key}' for key in ('W0', 'W1', 'W2', 'W3', 'Wstar')},
            'teacher_question_forms': {key: f'NEW QUESTION {key}' for key in ('T1', 'T2', 'T3', 'Tstar')},
            'options': ['correct', 'wrong1', 'wrong2', 'wrong3']}


def test_teacher_new_questions_are_balanced_and_do_not_expose_preference():
    source = row()
    formatter = {'get_mcq_question_format': lambda options: ' | '.join(options)}
    counts = Counter()
    for step in range(288):
        query, target, order, family = teacher.query_target(source, step, formatter)
        position = order.index(0)
        counts[family, position] += 1
        assert query.startswith(source['teacher_question_forms'][family])
        assert 'PRIVATE PREFERENCE' not in query
        assert target == f'<choice>{"ABCD"[position]}</choice>'
        assert sorted(order) == [0, 1, 2, 3]
    assert set(counts.values()) == {24}
    assert len(counts) == 12


def test_no_legacy_forms_fallback(tmp_path):
    artifact = {'train': [row()]}
    artifact['train'][0]['forms'] = artifact['train'][0].pop('teacher_question_forms')
    path = tmp_path / 'data.json'
    path.write_text(json.dumps(artifact))
    with pytest.raises(KeyError):
        teacher.load_data(path, 'train')


def test_resume_refuses_changed_binding(tmp_path):
    path = tmp_path / 'identity.json'
    teacher.bound_json(path, {'data_sha256': 'a'})
    teacher.bound_json(path, {'data_sha256': 'a'})
    with pytest.raises(ValueError, match='binding differs'):
        teacher.bound_json(path, {'data_sha256': 'b'})


def test_shards_cover_exactly_all_preferences():
    rows = [row(f'topic:{i:04d}') for i in range(730)]
    assignments = [teacher.select_shard(rows, shard, 8) for shard in range(8)]
    assert sorted(len(x) for x in assignments) == [91] * 6 + [92] * 2
    assert len({r['base_pair_id'] for part in assignments for r in part}) == 730


def test_donors_are_same_topic_distinct_and_order_invariant():
    rows = [row('a:0', 'a'), row('a:1', 'a'), row('b:0', 'b'), row('b:1', 'b')]
    for i, source in enumerate(rows):
        source['writer_user_forms']['W0'] = f'Preference {i}'
    mapping = evaluation.donor_map(rows)
    assert mapping == evaluation.donor_map(list(reversed(rows)))
    assert all(pid != donor and pid[0] == donor[0] for pid, donor in mapping.items())
    with pytest.raises(ValueError, match='same-topic'):
        evaluation.donor_map(rows[:1])


def test_donors_skip_identical_preference_even_when_ids_differ():
    rows = [row('a:0', 'a'), row('a:1', 'a'), row('a:2', 'a')]
    rows[0]['writer_user_forms']['W0'] = 'I like cats.'
    rows[1]['writer_user_forms']['W0'] = ' I LIKE  cats. '
    rows[2]['writer_user_forms']['W0'] = 'I like dogs.'
    assert evaluation.donor_map(rows)['a:0'] == 'a:2'
    with pytest.raises(ValueError, match='distinct same-topic preference'):
        evaluation.donor_map(rows[:2])


def test_counterfactual_mismatch_uses_original_preference_not_an_unrelated_topic():
    opposite = row('topic:0001::opposite')
    opposite['origin_id'] = 'topic:0001'
    assert evaluation.mismatch_donors([opposite], 'opposites') == {'topic:0001::opposite': 'topic:0001'}


@pytest.mark.parametrize('kind,split,expected', [('teacher', 'train', 48), ('student', 'train', 24),
                                               ('student', 'diagnostics', 72)])
def test_read_protocol_covers_seed_positions_and_controls_without_duplicate_keys(kind, split, expected):
    args = SimpleNamespace(kind=kind, split=split, seeds=[0, 1], controls=['memory', 'mismatch', 'blank', 'text'])
    keys = list(evaluation.planned_read_keys([row()], args))
    assert len(keys) == len(set(keys)) == expected
    for key in keys:
        if key[4] in {'blank', 'text'}:
            assert key[1] == 0
    if split == 'diagnostics':
        assert {(key[2], key[3]) for key in keys} == {('Wstar', 'T1'), ('W0', 'Tstar'), ('Wstar', 'Tstar')}


def test_eval_rotation_and_writer_input_excludes_questions():
    source = row()
    for family in ('T1', 'T2', 'T3', 'Tstar'):
        for position in range(4):
            order, correct = evaluation.eval_order(source['base_pair_id'], family, position)
            assert order[correct] == 0 and correct == position
    text = evaluation.event_text(source, 'Wstar')
    assert text == 'user: PRIVATE PREFERENCE Wstar\nassistant: Understood.'
    assert 'QUESTION' not in text and 'correct' not in text


def test_official_is_final_endpoint_only_and_checkpoint_hash_is_checked(tmp_path):
    checkpoint = tmp_path / 'checkpoint-step-005840.pt'
    checkpoint.write_bytes(b'fixture')
    receipt = checkpoint.with_suffix('.ready.json')
    teacher.save_json(receipt, {'checkpoint_sha256': teacher.sha(checkpoint), 'step': 5840})
    args = SimpleNamespace(kind='student', checkpoint=checkpoint, split='official')
    with pytest.raises(ValueError, match='93440'):
        evaluation.checkpoint_identity(args)
    args.split = 'train'
    assert evaluation.checkpoint_identity(args)['optimizer_step'] == 5840
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        evaluation.checkpoint_identity(args)


def test_readback_recovers_only_interrupted_tail(tmp_path):
    path = tmp_path / 'readback.jsonl'
    path.write_bytes(b'{"ok":1}\n{"truncated":')
    assert evaluation.read_rows(path) == [{'ok': 1}]
    assert path.read_bytes() == b'{"ok":1}\n'
    path.write_bytes(b'invalid\n{"ok":1}\n')
    with pytest.raises(json.JSONDecodeError):
        evaluation.read_rows(path)


def test_teacher_cannot_optimize_or_read_heldout_targets():
    with pytest.raises(ValueError, match='train730'):
        evaluation.question_suites('teacher', 'official')
