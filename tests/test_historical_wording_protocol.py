import copy
import json
from pathlib import Path

import pytest

from scripts.experiments.historical_wording_protocol import variants, wording_index
from scripts.experiments.broader_writer_protocol import prefix_validation, SEED
from vision_memory.training.latent_bank_unet import balanced_draw


def bank():
    return json.loads((Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())


def test_augmentation_uses_all_original_events_and_excludes_validation_text():
    value = bank()
    generated = variants(value)
    assert len(generated) == 16
    original = {group['question_id']: group for group in value['groups']}
    validation = {case['event_text'] for case in prefix_validation()['cases'] if case['style'] != 'original_prefix'}
    for qid, texts in generated.items():
        assert len(set(texts)) == 9
        assert texts[0] == original[qid]['event_text']
        assert not set(texts[1:]) & validation
        assert all(len(text.splitlines()) == len(texts[0].splitlines()) for text in texts)
    changed = copy.deepcopy(value)
    for group in changed['groups']:
        group['answer'] = 'FORBIDDEN_ANSWER'
        group['question_variants'] = {'forbidden': 'FORBIDDEN_QUERY'}
    assert variants(changed) == generated


def test_every_nine_visits_cover_all_expressions_without_changing_draws():
    value = bank()
    texts = variants(value)
    choices = {qid: [] for qid in texts}
    for draw_index in range(31 * 18):
        group, teacher, noise, sigma = balanced_draw(value['groups'], SEED, draw_index, sampling_strategy='logical_condition')
        qid = group['question_id']
        if qid in texts:
            choices[qid].append(wording_index(SEED, draw_index, qid))
        assert balanced_draw(value['groups'], SEED, draw_index, sampling_strategy='logical_condition') == (group, teacher, noise, sigma)
    for indices in choices.values():
        assert len(indices) == 18
        assert set(indices[:9]) == set(indices[9:]) == set(range(9))


def test_changed_or_missing_historical_event_is_rejected():
    value = bank()
    historical = next(group for group in value['groups'] if 'historical_target_index' in group)
    historical['event_text'] += '\nInvent a different event.'
    with pytest.raises(ValueError, match='sealed original'):
        variants(value)
    value['groups'].remove(historical)
    with pytest.raises(ValueError, match='sixteen'):
        variants(value)


@pytest.mark.parametrize('enabled,historical', [(True, True), (False, True), (True, False)])
def test_actual_microbatch_changes_condition_only(monkeypatch, enabled, historical):
    from types import SimpleNamespace
    import torch
    from scripts.train import train_latent_bank_unet as training
    group = {'question_id': 'sample', 'teacher_ids': ['teacher']}
    if historical:
        group['historical_target_index'] = 0
    monkeypatch.setattr(training, 'balanced_draw', lambda *args, **kwargs: (group, 'teacher', 42, 0.375))
    source, target = torch.full((1, 1, 2, 2), 7.), torch.full((1, 1, 2, 2), 3.)
    def condition(value):
        return SimpleNamespace(prompt_embeds=torch.tensor([value]), attention_mask=torch.ones(1))
    original = condition(-1.)
    encoded = [condition(float(index)) for index in range(9)]
    bindings = [{'index': index, 'event_text_sha256': str(index)} for index in range(9)]
    context = {'source': source, 'condition': original, 'training_condition_variants': encoded,
               'training_condition_variant_binding': bindings}
    parameter = torch.tensor(2., requires_grad=True)
    captured = {}
    def predict(sampler, state, actual_source, sigma, embeds, mask, **kwargs):
        captured.update(state=state, source=actual_source, sigma=sigma, embeds=embeds)
        return torch.ones_like(state) * parameter * embeds[0]
    monkeypatch.setattr(training, 'predict_velocity', predict)
    args = SimpleNamespace(flow_protocol='official', seed=SEED, sampling_strategy='logical_condition',
                           historical_wording_augmentation=enabled, gradient_accumulation_steps=4)
    runtime = {'contexts': {'sample': context}, 'vae_device': 'cpu', 'sampler': object()}
    row = training.flow_microbatch(args, runtime, [group], {'teacher': target}, 7)
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(42))
    assert torch.equal(captured['state'], (1 - .375) * target + .375 * noise)
    assert captured['source'] is source
    assert row['teacher_id'] == 'teacher' and row['noise_seed'] == 42 and row['effective_sigma'] == .375
    if enabled and historical:
        index = wording_index(SEED, 7, 'sample')
        assert captured['embeds'] is encoded[index].prompt_embeds
        assert row['training_condition_variant'] == bindings[index]
    else:
        assert captured['embeds'] is original.prompt_embeds
        assert 'training_condition_variant' not in row
    assert parameter.grad is not None and torch.isfinite(parameter.grad)
