import copy
import json
from pathlib import Path
import pytest
from scripts.probes.official_transition_confirmation import resolve_events
from scripts.probes.transition_validation_plan import plan
from scripts.reporting.collect_transition_validation import GOLD_IDS, expected_rows, summarize_rows


def fixture(mode):
    bank = json.loads((Path(__file__).resolve().parents[1] /
        'reports/official-alignment-results-20260913/transition-wording-bank-manifest.json').read_text())
    original, resolved = resolve_events(plan(), bank)
    variants = original['ambient']['question_variants']
    expected, artifacts = expected_rows(resolved, variants, mode)
    rows = [{**meta, 'prompt_id': key[-1], 'image_sha256': meta['image_artifact'], 'raw': meta['gold'],
             'generated_token_ids': GOLD_IDS[meta['gold']] + [151645],
             'scorer': {'gold_token_ids': GOLD_IDS[meta['gold']], 'strict_correct': True,
                        'answer_followed_immediately_by_eos': True}} for key, meta in expected.items()]
    return rows, resolved, variants, artifacts


def test_single_controls_do_not_inflate_confirmation_and_missing_rewrite_is_rejected():
    rows, resolved, variants, artifacts = fixture('single_writes')
    result, actual = summarize_rows(rows, resolved, variants, 'single_writes')
    assert actual == artifacts and len(artifacts) == 152
    assert result['raw_rows'] == 390 and result['matched_correct_eos'] == 360
    assert result['generated_images'] == result['all_five_prompt_images'] == 72
    with pytest.raises(ValueError, match='Missing'):
        summarize_rows(rows[:-1], resolved, variants, 'single_writes')
    changed = copy.deepcopy(rows)
    changed[0]['event_text'] = 'an easier substitute event'
    with pytest.raises(ValueError, match='event'):
        summarize_rows(changed, resolved, variants, 'single_writes')


def test_one_failed_noop_breaks_the_full_chain_and_oracle_reset_is_rejected():
    rows, resolved, variants, artifacts = fixture('rgb_chains')
    index = next(i for i, r in enumerate(rows) if r['operation'] == 'noop')
    rows[index]['generated_token_ids'] = [7, 151645]
    rows[index]['raw'] = 'wrong'
    rows[index]['scorer']['strict_correct'] = False
    result, actual = summarize_rows(rows, resolved, variants, 'rgb_chains')
    assert actual == artifacts and len(artifacts) == 194
    assert result['matched_correct_eos'] == 479 and result['all_five_prompt_images'] == 95
    assert result['complete_chains_correct_eos'] == 15 and result['chain_count'] == 16
    assert not result['all_generated_correct_eos']
    rows[index]['source_artifact'] = None
    with pytest.raises(ValueError, match='source link'):
        summarize_rows(rows, resolved, variants, 'rgb_chains')
