import json
from pathlib import Path
import pytest
from scripts.probes.historical_fp32_readback import FORMS, summarize


def fixture():
    panel = json.loads((Path(__file__).resolve().parents[1] /
        'reports/official-alignment-results-20260913/historical-fp32-readback-panel.json').read_text())
    targets = {t['target_index']: t for t in panel['targets']}
    rows = []
    for member in panel['members']:
        target = targets[member['target_index']]
        for form in FORMS:
            for prompt, query in target['question_variants'].items():
                rows.append({'target_index': target['target_index'], 'seed': member['seed'], 'condition': 'matched',
                    'image_form': form, 'prompt_id': prompt, 'query': query, 'gold': target['gold'],
                    'generated_token_ids': [7, 151645], 'scorer': {'gold_token_ids': [7], 'strict_correct': True,
                                                               'answer_followed_immediately_by_eos': True}})
    for target in panel['targets']:
        for prompt, query in target['question_variants'].items():
            rows.append({'target_index': target['target_index'], 'seed': None, 'condition': 'blank',
                'image_form': 'rgb_uint8', 'prompt_id': prompt, 'query': query, 'gold': target['gold'],
                'generated_token_ids': [8, 151645], 'scorer': {'gold_token_ids': [7], 'strict_correct': False,
                                                           'answer_followed_immediately_by_eos': False}})
    return rows, panel


def test_quantized_failure_is_retained_even_if_float_decode_passes():
    rows, panel = fixture()
    result = summarize(rows, panel)
    assert result['members_passing_both_forms_all_five'] == 64
    row = next(r for r in rows if r['condition'] == 'matched' and r['image_form'] == 'rgb_uint8')
    row['generated_token_ids'] = [8, 151645]
    row['scorer']['strict_correct'] = False
    result = summarize(rows, panel)
    assert result['members_passing_both_forms_all_five'] == 63
    assert result['passing_members_per_target']['0'] == 3
    assert result['targets_passing_all_four_seeds'] == 15
    assert not result['all_matched_correct_eos'] and not result['shared_writer_success']


def test_missing_form_duplicate_and_false_score_are_rejected():
    rows, panel = fixture()
    with pytest.raises(ValueError, match='Missing'):
        summarize(rows[1:], panel)
    with pytest.raises(ValueError, match='duplicated'):
        summarize(rows + [rows[0]], panel)
    rows[0]['generated_token_ids'] = [8, 151645]
    with pytest.raises(ValueError, match='tokens'):
        summarize(rows, panel)
