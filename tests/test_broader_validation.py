import copy
import json
from pathlib import Path
import pytest

from scripts.reporting.collect_broader_endpoint import phase_summary, GOLD_IDS, SEED, COMMIT, BANK_SHA, PLAN_SHA
from scripts.reporting.collect_broader_validation import expected_rows, summarize_rows
from scripts.probes.official_broader_confirmation import selected_cases
from scripts.probes.rgb_package_parity import replay_registration
from scripts.experiments.broader_writer_protocol import training_plan
from vision_memory.training.latent_bank_unet import stable_seed

RESULTS = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'


@pytest.fixture
def actual():
    bank = json.loads((RESULTS / 'broader151-bank-manifest.json').read_bytes())
    plan = training_plan(BANK_SHA, COMMIT)
    assert plan == json.loads((RESULTS / 'broader151-preregistered-plan.json').read_bytes())
    return bank, plan


def answer(row, *, correct=True):
    ids = GOLD_IDS[row['gold']]
    return {**row, 'raw': row['gold'] if correct else 'incorrect extra text',
        'generated_token_ids': ids + [151645] if correct else ids + [123, 151645],
        'scorer': {'gold_token_ids': ids, 'strict_correct': correct, 'answer_followed_immediately_by_eos': correct},
        'image_sha256': str((row.get('question_id', row.get('case')), row['condition'], row['noise_seed']))}


def phase_rows(bank):
    return [answer({'question_id': group['question_id'], 'gold': group['answer'], 'query': query,
        'prompt_id': prompt, 'condition': condition, 'noise_seed': seed, 'phase': 'baseline'})
        for group in bank['groups'] for prompt, query in group['question_variants'].items()
        for condition, seed in [('blank', None), ('donor', None), *(('matched', stable_seed(SEED, 'heldout-evaluation-noise', i)) for i in range(2))]]


def test_all151_actual_question_contracts_and_failure_count(actual):
    bank, _ = actual
    rows = phase_rows(bank)
    matched = next(i for i, row in enumerate(rows) if row['condition'] == 'matched')
    rows[matched] = answer(rows[matched], correct=False)
    summary, _ = phase_summary(rows, bank, 'baseline')
    assert (summary['raw_rows'], summary['correct_eos'], summary['images_passing_all_five']) == (3020, 1509, 301)
    assert summary['partitions']['historical_prefixes']['n'] == 160
    for mutation in ('missing_group', 'duplicate', 'gold_tokens', 'wrong_query', 'image_changed', 'wrong_seed'):
        broken = copy.deepcopy(rows)
        if mutation == 'missing_group':
            broken = [row for row in broken if row['question_id'] != bank['groups'][-1]['question_id']]
        elif mutation == 'duplicate':
            broken[-1] = broken[0]
        elif mutation == 'gold_tokens':
            broken[matched]['scorer']['gold_token_ids'] = broken[matched]['generated_token_ids'][:-1]
            broken[matched]['scorer'].update(strict_correct=True, answer_followed_immediately_by_eos=True)
        elif mutation == 'wrong_query':
            broken[-1]['query'] = rows[0]['query']
        elif mutation == 'image_changed':
            broken[-1]['image_sha256'] = 'query-dependent-image'
        else:
            broken[-1]['noise_seed'] += 1
        with pytest.raises(ValueError):
            phase_summary(broken, bank, 'baseline')


@pytest.mark.parametrize('mode,lane,matched,images', [('single_writes', None, 360, 72), ('rgb_chains', None, 480, 96),
    ('historical_prefixes', 0, 480, 96), ('historical_prefixes', 1, 480, 96)])
def test_independent_matrix_failures_controls_and_query_integrity(actual, mode, lane, matched, images):
    bank, plan = actual
    expected, _, _ = expected_rows(plan, bank, mode, lane)
    rows = [answer(row, correct=row['condition'] == 'matched') for row in expected.values()]
    summary, _, _ = summarize_rows(rows, plan, bank, mode, lane)
    assert summary['matched_correct_eos'] == matched and summary['generated_images'] == images
    assert summary['all_generated_correct_eos']
    index = next(i for i, row in enumerate(rows) if row['condition'] == 'matched')
    rows[index] = answer(rows[index], correct=False)
    summary, _, _ = summarize_rows(rows, plan, bank, mode, lane)
    assert summary['matched_correct_eos'] == matched - 1 and not summary['all_generated_correct_eos']
    for mutation in ('missing', 'extra', 'event', 'condition', 'image', 'query'):
        broken = copy.deepcopy(rows)
        if mutation == 'missing':
            broken.pop()
        elif mutation == 'extra':
            broken.append(broken[0])
        else:
            key = {'event': 'event_text', 'condition': 'condition', 'image': 'image_sha256', 'query': 'query'}[mutation]
            broken[0][key] = 'altered'
        with pytest.raises(ValueError):
            summarize_rows(broken, plan, bank, mode, lane)


def test_two_prefix_lanes_exhaust_all_targets_and_fixed_package_replay(actual):
    bank, plan = actual
    left = selected_cases(plan, 'historical_prefixes', 0)
    right = selected_cases(plan, 'historical_prefixes', 1)
    assert len(left) == len(right) == 24
    assert {case['target_index'] for case in left}.isdisjoint({case['target_index'] for case in right})
    assert {case['target_index'] for case in left + right} == set(range(16))
    _, _, cases = expected_rows(plan, bank, 'rgb_chains', None)
    identity = {'registered_plan': plan, 'parent_commit': COMMIT, 'bank_sha256': BANK_SHA, 'plan_file_sha256': PLAN_SHA,
        'mode': 'rgb_chains', 'selected_cases': cases}
    assert replay_registration(identity, broader=True) == cases[0]
    for mutation in ('selection', 'seed', 'plan'):
        broken = copy.deepcopy(identity)
        if mutation == 'selection':
            broken['selected_cases'] = broken['selected_cases'][1:]
        elif mutation == 'seed':
            broken['selected_cases'][0]['steps'][0]['noise_seed'] += 1
        else:
            broken['registered_plan']['optimizer_steps'] += 1
        with pytest.raises(ValueError):
            replay_registration(broken, broader=True)
