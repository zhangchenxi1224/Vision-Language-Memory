import copy
import json
from pathlib import Path

import pytest

from scripts.experiments.historical_wording_protocol import POLICY, variants, wording_index
from scripts.experiments.broader_writer_protocol import SEED
from scripts.reporting.verify_historical_condition_draws import identity_binding, verify_seal, verify_draw


@pytest.fixture
def evidence():
    root = Path(__file__).resolve().parents[1]
    bank = json.loads((root / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    registered = {'training_augmentation': {'policy': POLICY, 'events': variants(bank)}}
    identity = {'training_augmentation': identity_binding(registered)}
    seal = {qid: [{'index': index, 'event_text_sha256': digest,
                  'prompt_embeds_sha256': f'{index + 1:064x}', 'attention_mask_sha256': 'a' * 64}
                 for index, digest in enumerate(events)]
            for qid, events in identity['training_augmentation']['event_text_sha256'].items()}
    runtime = {'condition_sha256': {qid: entries[0]['prompt_embeds_sha256'] for qid, entries in seal.items()}}
    return bank, identity, runtime, seal, registered


def test_valid_seal_and_actual_draw_binding(evidence):
    bank, identity, runtime, seal, registered = evidence
    assert verify_seal(*evidence) == identity['training_augmentation']
    for group in bank['groups']:
        if 'historical_target_index' in group:
            qid = group['question_id']
            index = wording_index(SEED, 100, qid)
            draw = {'training_condition_variant': seal[qid][index]}
            assert verify_draw(group, draw, 100, seal) == (qid, str(index))
            wrong = {'training_condition_variant': seal[qid][(index + 1) % 9]}
            with pytest.raises(ValueError, match='different expression'):
                verify_draw(group, wrong, 100, seal)
        else:
            assert verify_draw(group, {}, 100, seal) is None
            with pytest.raises(ValueError, match='Music'):
                verify_draw(group, {'training_condition_variant': {}}, 100, seal)


@pytest.mark.parametrize('mutation', ['missing_question', 'missing_style', 'wrong_event', 'bad_mask_hash', 'original_changed', 'writer_input'])
def test_incomplete_or_changed_condition_evidence_rejected(evidence, mutation):
    bank, identity, runtime, seal, registered = copy.deepcopy(evidence)
    qid = next(iter(seal))
    if mutation == 'missing_question':
        seal.pop(qid)
    elif mutation == 'missing_style':
        seal[qid].pop()
    elif mutation == 'wrong_event':
        seal[qid][1]['event_text_sha256'] = 'b' * 64
    elif mutation == 'bad_mask_hash':
        seal[qid][1]['attention_mask_sha256'] = 'nan'
    elif mutation == 'original_changed':
        runtime['condition_sha256'][qid] = 'c' * 64
    else:
        identity['training_augmentation']['writer_input'] = True
    with pytest.raises(ValueError):
        verify_seal(bank, identity, runtime, seal, registered)


def test_actual_fixed_training_commit_selects_the_observed_plan(evidence):
    from scripts.reporting.collect_broader_endpoint import registered_protocol, HISTORICAL_WORDING_COMMIT
    commit, registered, digest = registered_protocol(evidence[0], HISTORICAL_WORDING_COMMIT)
    assert commit == 'b9f90e956eea7bda15f638c8877919941ce4fec5'
    assert digest == 'a35d3986439dab371f3bfd90243ed948bba3c3a4f4fd0887d3021e38dead081e'
    assert registered['reference_commit'] == '03f8467e5a1201c2dbd9d12484bf2338d7837727'
