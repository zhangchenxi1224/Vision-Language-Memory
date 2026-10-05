"""CPU tests for history-only targets, immutable identities, and paired-arm caches."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments import prefeval_prompt_matching as pm  # noqa: E402


def row():
    return {
        'base_pair_id': 'drink:0001',
        'history': [
            {'role': 'user', 'content': 'I prefer tea.'},
            {'role': 'assistant', 'content': 'I will remember your tea preference.'},
            {'role': 'user', 'content': 'FUTURE_USER_SENTINEL'},
            {'role': 'assistant', 'content': 'FUTURE_ASSISTANT_SENTINEL'},
        ],
        'target': {'content': 'PUBLISHED_ANSWER_SENTINEL'},
        'options': ['CORRECT_OPTION_SENTINEL', 'DISTRACTOR_SENTINEL'],
        'forms': {'T1': 'What drink would you suggest?', 'O1': 'HELDOUT_QUESTION_SENTINEL'},
        'query': {'content': 'FUTURE_QUERY_SENTINEL'},
    }


def args(tmp_path, supervision='prompt_matching', temperature=1.):
    reader = tmp_path / 'reader'
    reader.mkdir(exist_ok=True)
    (reader / 'config.json').write_text('{"model_type":"fake_reader"}')
    (reader / 'model-00001-of-00001.safetensors').write_bytes(b'fake weight identity for CPU tests')
    return SimpleNamespace(supervision=supervision, temperature=temperature,
                           teacher_cache=None, boundary_recovery_from=None,
                           teacher_max_new_tokens=32, reader=reader)


def generation():
    return {
        'generated_token_ids': [11, 12, 100], 'generated_token_count': 3,
        'eos_token_ids': [99, 100], 'eos_reached': True,
        'truncated': False, 'finish_reason': 'eos',
        'raw': 'teacher answer',
    }


@pytest.fixture
def target_environment(tmp_path, monkeypatch):
    calls = {'generation': [], 'scoring': []}
    receipt = generation()

    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        calls['generation'].append(kwargs)
        return deepcopy(receipt)

    def logits(**kwargs):
        assert not torch.is_grad_enabled()
        calls['scoring'].append(kwargs)
        # Nonconstant full-vocabulary logits, retained in their native model dtype.
        return SimpleNamespace(target_logits=torch.arange(384, dtype=torch.float16).reshape(1, 3, 128) / 20)

    monkeypatch.setattr(pm, 'generate_short_answer', generate)
    monkeypatch.setattr(pm, 'qwen3vl_continuation_logits', logits)
    sample = row()
    settings = args(tmp_path)
    binding = pm.supervision_binding(settings, [sample])
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    reference = torch.full((3, 4, 4), 128 / 255.)
    kwargs = dict(cache_root=tmp_path / 'cache', binding=binding, row=sample,
                  query=sample['forms']['T1'], model=model, processor=processor,
                  reference=reference, device='cpu', assistant_end_token_id=100)
    return SimpleNamespace(calls=calls, receipt=receipt, sample=sample, settings=settings,
                           binding=binding, kwargs=kwargs)


def test_history_query_contains_only_initial_actual_exchange_and_supplied_query():
    sample = row()
    query = sample['forms']['T1']
    text = pm.history_teacher_query(sample, query)
    assert 'user: I prefer tea.' in text
    assert 'assistant: I will remember your tea preference.' in text
    assert text.endswith('Current question:\n' + query)
    for forbidden in ('FUTURE_USER_SENTINEL', 'FUTURE_ASSISTANT_SENTINEL', 'PUBLISHED_ANSWER_SENTINEL',
                      'CORRECT_OPTION_SENTINEL', 'DISTRACTOR_SENTINEL', 'HELDOUT_QUESTION_SENTINEL',
                      'FUTURE_QUERY_SENTINEL'):
        assert forbidden not in text
    with pytest.raises(ValueError, match='nonempty'):
        pm.history_teacher_query(sample, ' ')


def test_target_creation_receives_only_reference_initial_history_and_query(target_environment):
    env = target_environment
    item, path = pm.get_history_target(**env.kwargs)
    assert path.is_file() and item['target_ids'] == [11, 12, 100]
    assert len(env.calls['generation']) == len(env.calls['scoring']) == 1
    generated, scored = env.calls['generation'][0], env.calls['scoring'][0]
    assert generated['query'] == scored['query'] == pm.history_teacher_query(env.sample, env.kwargs['query'])
    assert generated['image'] is scored['image'] is env.kwargs['reference']
    assert generated['do_sample'] is False and generated['max_new_tokens'] == 32
    assert scored['require_image_grad'] is False
    assert scored['target_ids'].tolist() == [[11, 12, 100]]
    assert generated['reader_resize_contract'] == scored['reader_resize_contract'] == pm.R3_QWEN_READER_RESIZE_CONTRACT
    assert not item['logits'].requires_grad and item['logits'].device.type == 'cpu'
    assert item['logits_sha256'] == pm.tensor_digest(item['logits'])


def test_same_raw_cache_is_shared_across_both_modes_and_temperatures(target_environment):
    env = target_environment
    first, path = pm.get_history_target(**env.kwargs)
    original = path.read_bytes()
    for mode, temperature in [('history_hard', 1.), ('prompt_matching', 2.), ('prompt_matching', 3.)]:
        changed = deepcopy(env.binding)
        changed.update(supervision=mode, temperature=temperature,
                       loss='teacher_generated_hard_cross_entropy' if mode == 'history_hard'
                       else 'full_vocab_soft_cross_entropy_temperature_squared')
        item, next_path = pm.get_history_target(**(env.kwargs | {'binding': changed}))
        assert next_path == path and next_path.read_bytes() == original
        torch.testing.assert_close(item['logits'], first['logits'])
        assert item['target_ids'] == first['target_ids']
    assert len(env.calls['generation']) == len(env.calls['scoring']) == 1


def test_future_labels_and_heldout_material_cannot_change_target_cache_identity(target_environment):
    env = target_environment
    before = pm.target_cache_binding(env.binding, env.sample, env.kwargs['query'])
    changed = deepcopy(env.sample)
    changed['history'][2:] = [{'role': 'user', 'content': 'DIFFERENT_FUTURE'}]
    changed['target'] = {'content': 'ANOTHER_ANSWER'}
    changed['options'] = ['OTHER_LABEL', 'OTHER_DISTRACTOR']
    changed['forms']['O1'] = 'ANOTHER_HELDOUT'
    changed['query'] = {'content': 'OTHER_FUTURE_QUERY'}
    after = pm.target_cache_binding(env.binding, changed, env.kwargs['query'])
    assert before == after
    assert pm.supervision_binding(env.settings, [changed]) == env.binding


@pytest.mark.parametrize('change', ['history', 'query', 'reader_config', 'reader_path', 'budget', 'pipeline'])
def test_semantic_or_reader_changes_invalidate_cache_identity(target_environment, change):
    env = target_environment
    changed_binding, changed_row = deepcopy(env.binding), deepcopy(env.sample)
    query = env.kwargs['query']
    if change == 'history':
        changed_row['history'][0]['content'] = 'I prefer coffee.'
    elif change == 'query':
        query = 'What else would you recommend?'
    elif change == 'reader_config':
        changed_binding['reader_config_sha256']['config.json'] = 'different'
    elif change == 'reader_path':
        changed_binding['reader_path'] += '-other'
    elif change == 'budget':
        changed_binding['teacher_max_new_tokens'] += 1
    else:
        changed_binding['target_pipeline_sha256'] = 'other-implementation'
    assert pm.digest(pm.target_cache_binding(changed_binding, changed_row, query)) != pm.digest(
        pm.target_cache_binding(env.binding, env.sample, env.kwargs['query']))


@pytest.mark.parametrize('corruption', ['tensor_bytes', 'identity', 'ids', 'shape', 'dtype', 'vocab', 'nonfinite', 'eos'])
def test_corrupt_cache_fails_closed_without_regenerating(target_environment, corruption):
    env = target_environment
    item, path = pm.get_history_target(**env.kwargs)
    item = deepcopy(item)
    if corruption == 'tensor_bytes':
        item['logits'][0, 0, 0] += 1
    elif corruption == 'identity':
        item['binding']['query'] += 'changed'
    elif corruption == 'ids':
        item['target_ids'][0] = 77
    elif corruption == 'shape':
        item['logits'] = item['logits'][:, :2]
        item['logits_sha256'] = pm.tensor_digest(item['logits'])
    elif corruption == 'dtype':
        item['logits'] = item['logits'].to(torch.int16)
        item['logits_sha256'] = pm.tensor_digest(item['logits'])
    elif corruption == 'vocab':
        item['logits'] = item['logits'][:, :, :100]
        item['logits_sha256'] = pm.tensor_digest(item['logits'])
    elif corruption == 'nonfinite':
        item['logits'][0, 0, 0] = float('nan')
        item['logits_sha256'] = pm.tensor_digest(item['logits'])
    else:
        item['generation']['truncated'] = True
    torch.save(item, path)
    with pytest.raises(ValueError):
        pm.get_history_target(**env.kwargs)
    assert len(env.calls['generation']) == 1


def test_truncated_teacher_is_not_cached_or_scored(target_environment):
    env = target_environment
    env.receipt.update(generated_token_ids=[11, 12], generated_token_count=2,
                       eos_reached=False, truncated=True, finish_reason='token_limit')
    with pytest.raises(ValueError, match='EOS'):
        pm.get_history_target(**env.kwargs)
    assert not list(env.kwargs['cache_root'].glob('*.pt'))
    assert env.calls['scoring'] == []


def test_hard_ce_binding_remains_empty_for_legacy_manifest_identity(tmp_path):
    settings = args(tmp_path, 'hard_ce')
    assert pm.supervision_binding(settings, [row()]) == {}
    legacy = {'step': 288, 'binding': {'arm': 'A', 'steps': 288, 'loss': 'legacy hard CE'}}
    assert pm.validate_teacher_manifest(legacy, arm='A') == pm.digest(legacy['binding'])
    with pytest.raises(ValueError, match='objective mismatch'):
        pm.validate_teacher_manifest(legacy, arm='A', supervision='prompt_matching')


def test_new_teacher_cannot_be_silently_loaded_as_legacy_or_other_mode(tmp_path):
    binding = pm.supervision_binding(args(tmp_path, 'prompt_matching'), [row()])
    binding['arm'] = 'A'
    done = {'step': 288, 'binding': binding}
    assert pm.validate_teacher_manifest(done, arm='A', supervision='prompt_matching') == pm.digest(binding)
    for wrong_mode in ['hard_ce', 'history_hard']:
        with pytest.raises(ValueError, match='objective mismatch'):
            pm.validate_teacher_manifest(done, arm='A', supervision=wrong_mode)
    with pytest.raises(ValueError, match='budget or task format'):
        pm.validate_teacher_manifest(done, arm='B', supervision='prompt_matching')
    with pytest.raises(ValueError, match='budget or task format'):
        pm.validate_teacher_manifest(done, arm='A', supervision='prompt_matching', steps=4)
    del binding['reader_objective_sha256']
    with pytest.raises(ValueError, match='provenance'):
        pm.validate_teacher_manifest(done, arm='A', supervision='prompt_matching')


@pytest.mark.parametrize('change', [
    {'supervision': 'unknown'}, {'temperature': 0.}, {'temperature': float('nan')},
    {'temperature': float('inf')}, {'boundary_recovery_from': 'a' * 40}, {'teacher_max_new_tokens': 0},
])
def test_invalid_new_mode_options_fail(tmp_path, change):
    settings = args(tmp_path)
    for key, value in change.items():
        setattr(settings, key, value)
    with pytest.raises(ValueError):
        pm.validate_options(settings)


@pytest.mark.parametrize('change', [{'temperature': 2.}, {'teacher_cache': Path('cache')}])
def test_new_options_cannot_silently_change_legacy_objective(tmp_path, change):
    settings = args(tmp_path, 'hard_ce')
    for key, value in change.items():
        setattr(settings, key, value)
    with pytest.raises(ValueError, match='only to history'):
        pm.validate_options(settings)


def test_reader_weight_changes_invalidate_binding_even_with_same_path_and_config(target_environment):
    env = target_environment
    weight = env.settings.reader / 'model-00001-of-00001.safetensors'
    weight.write_bytes(b'new reader weights at the same path')
    after = pm.supervision_binding(env.settings, [env.sample])
    assert after['reader_config_sha256'] == env.binding['reader_config_sha256']
    assert after['reader_path'] == env.binding['reader_path']
    assert after['reader_weights_sha256'] != env.binding['reader_weights_sha256']
    assert pm.target_cache_binding(after, env.sample, env.kwargs['query']) != pm.target_cache_binding(
        env.binding, env.sample, env.kwargs['query'])
    weight.unlink()
    with pytest.raises(ValueError, match='weight files are missing'):
        pm.supervision_binding(env.settings, [env.sample])
    (env.settings.reader / 'pytorch_model-00001-of-00001.bin').write_bytes(b'alternate format')
    restored = pm.supervision_binding(env.settings, [env.sample])
    assert set(restored['reader_weights_sha256']) == {'pytorch_model-00001-of-00001.bin'}


def test_hard_distillation_rejects_ineffective_temperature(tmp_path):
    settings = args(tmp_path, 'history_hard', temperature=2.)
    with pytest.raises(ValueError, match='requires temperature=1'):
        pm.validate_options(settings)


@pytest.mark.parametrize('different', [False, True])
def test_racing_publication_never_overwrites_winner_and_checks_agreement(target_environment, monkeypatch, different):
    env = target_environment
    published = {}

    def racing_link(temp, destination):
        winner = torch.load(temp, map_location='cpu', weights_only=True)
        winner['producer'] = 'winning process'
        if different:
            winner['logits'][0, 0, 0] += .5
            winner['logits_sha256'] = pm.tensor_digest(winner['logits'])
        torch.save(winner, destination)
        published['path'] = destination
        published['bytes'] = destination.read_bytes()
        raise FileExistsError('Another process won atomic publication')

    monkeypatch.setattr(pm.os, 'link', racing_link)
    if different:
        with pytest.raises(ValueError, match='Concurrent.*different'):
            pm.get_history_target(**env.kwargs)
    else:
        item, path = pm.get_history_target(**env.kwargs)
        assert item['producer'] == 'winning process'
        assert path == published['path']
    assert published['path'].read_bytes() == published['bytes']
    assert list(env.kwargs['cache_root'].glob('*.tmp')) == []


def test_tensor_hash_binds_dtype_and_shape():
    value = torch.zeros(1, 3, 4, dtype=torch.float16)
    assert pm.tensor_digest(value) != pm.tensor_digest(value.view(torch.int16))
    assert pm.tensor_digest(value) != pm.tensor_digest(value.reshape(1, 4, 3))
