import copy
import json
from pathlib import Path

from scripts.experiments.broader_writer_protocol import augment, training_plan, additional_events
from scripts.experiments.build_broader_writer_bank import merge
from scripts.experiments.refine_historical_writer_targets import selection
from scripts.experiments.transition_warm_start_plan import plan as warm_plan


def test_full_real_condition_coverage_and_fresh_validation_separation():
    root = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'
    transition = json.loads((root / 'transition-wording-bank-manifest.json').read_bytes())
    historical = json.loads((root / 'historical-writer-bank-manifest.json').read_bytes())
    refined = copy.deepcopy(historical)
    selected = {row['question_id']: row['teacher_id'] for row in selection(refined)}
    refined['teachers'] = [teacher for teacher in refined['teachers'] if teacher['teacher_id'] in selected.values()]
    for group in refined['groups']:
        group['teacher_ids'] = [selected[group['question_id']]]
    original = merge(transition, refined, {})
    bank = augment(original)
    assert bank['groups'][:61] == original['groups']
    assert bank['teachers'][:61] == original['teachers']
    by_transition = {}
    for group in bank['groups']:
        if 'parent_transition_group' in group:
            by_transition.setdefault(group['parent_transition_group'], []).append(group)
    assert len(by_transition) == 15
    for groups in by_transition.values():
        assert len(groups) == 9
        assert len({group['event_text'] for group in groups}) == 9
        for name in ('source_latent_sha256', 'source_state', 'target_state', 'operation'):
            assert len({group[name] for group in groups}) == 1
    assert additional_events('noop', 'ambient') == additional_events('noop', 'jazz') == additional_events('noop', 'clear')
    plan = training_plan('fixture-bank', 'fixture-commit')
    assert plan['optimizer_steps'] * plan['global_batch'] == 151 * plan['draws_per_group'] == 19328
    training_texts = {group['event_text'] for group in bank['groups']}
    def expressions(value):
        return {case['event_text'] for case in value['single_writes'] if 'event_text' in case} | {
            step['event_text'] for chain in value['rgb_chains'] for step in chain['steps'] if 'event_text' in step}
    assert expressions(warm_plan()['validation']) <= training_texts
    fresh = expressions(plan['transition_validation'])
    assert len(fresh) == 8 and not fresh.intersection(training_texts)
    cases = plan['prefix_validation']['cases']
    assert len(cases) == 48 and sum(len(case['noise_seeds']) for case in cases) == 192
    assert {case['target_index'] for case in cases} == set(range(16))
    for case in cases:
        group = next(group for group in historical['groups'] if group['question_id'] == case['question_id'])
        assert len(case['event_semantics']) == len(group['source_event_stream'])
        assert case['event_semantics'][-1]['value'] == case['gold'] == group['answer']
        assert case['question_variants'] == group['question_variants']
        assert (case['event_text'] in training_texts) == (case['style'] == 'original_prefix')
