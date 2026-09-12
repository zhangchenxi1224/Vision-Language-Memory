import copy
import json
from pathlib import Path
import pytest

from scripts.probes.official_transition_confirmation import development_gate, resolve_events
from scripts.probes.transition_validation_plan import plan
from vision_memory.training.latent_bank_unet import stable_seed


def bank():
    return json.loads((Path(__file__).resolve().parents[1] /
        'reports/official-alignment-results-20260913/transition-wording-bank-manifest.json').read_text())


def rows_for_bank(current):
    return [{'condition': 'matched', 'question_id': g['question_id'], 'query': query,
             'gold': g['answer'], 'prompt_id': prompt, 'noise_seed': stable_seed(20260913, 'heldout-evaluation-noise', i),
             'generated_token_ids': [7, 151645], 'scorer': {'gold_token_ids': [7], 'strict_correct': True,
                                                        'answer_followed_immediately_by_eos': True}}
            for g in current['groups'] for i in range(4) for prompt, query in g['question_variants'].items()]


def test_full_development_gate_rejects_missing_duplicates_and_false_raw_scores():
    current = bank()
    rows = rows_for_bank(current)
    assert development_gate(rows, current)['correct_eos'] == 900
    with pytest.raises(ValueError, match='complete'):
        development_gate(rows[:-1], current)
    with pytest.raises(ValueError, match='duplicate'):
        development_gate(rows + [rows[0]], current)
    broken = copy.deepcopy(rows)
    broken[0]['generated_token_ids'].insert(1, 8)
    with pytest.raises(ValueError, match='tokens'):
        development_gate(broken, current)
    broken[0]['scorer']['answer_followed_immediately_by_eos'] = False
    result = development_gate(broken, current)
    assert not result['all_correct_eos'] and result['correct_eos'] == 899
    assert len(result['failed_cells']) == 1


def test_resolved_original_events_use_gray_groups_and_preserve_all_new_literals():
    current, registered = bank(), plan()
    before = copy.deepcopy(registered)
    originals, resolved = resolve_events(registered, current)
    assert registered == before
    assert all(g['source_state'] == 'gray' and g['wording_index'] == 0 for g in originals.values())
    for initial, actual in zip(registered['single_writes'], resolved['single_writes'], strict=True):
        if 'event_text' in initial:
            assert actual['event_text'] == initial['event_text']
    for initial, actual in zip(registered['rgb_chains'], resolved['rgb_chains'], strict=True):
        for left, right in zip(initial['steps'], actual['steps'], strict=True):
            assert right['event_text']
            if 'event_text' in left:
                assert right['event_text'] == left['event_text']
    noops = {s['event_text'] for c in resolved['rgb_chains'] if c['repetition'] < 2
             for s in c['steps'] if s['operation'] == 'noop'}
    assert len(noops) == 1
