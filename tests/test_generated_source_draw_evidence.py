import hashlib
import json
from pathlib import Path

import pytest

from scripts.reporting.verify_generated_source_draws import identity_binding, pool_manifest, verify_seal, verify_draw
from scripts.train.generated_source_augmentation import source_variant_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sealed():
    bank = json.loads((ROOT / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    registered = json.loads((ROOT / 'reports/official-alignment-results-20260913/generated-source-training-preregistered.json').read_bytes())
    pool = pool_manifest()
    runtime, seal = {'condition_sha256': {}}, {}
    for group in bank['groups']:
        if group.get('source_kind') != 'sealed_rgb_1024':
            continue
        qid = group['question_id']
        records = sorted((r for r in pool['records'] if r['state'] == group['source_state']), key=lambda r: r['job'])
        seal[qid] = []
        for index in range(9):
            digest = hashlib.sha256(f'fixture-native-encoding:{qid}:{index}'.encode()).hexdigest()
            seal[qid].append({'index': index, 'event_text_sha256': hashlib.sha256(group['event_text'].encode()).hexdigest(),
                'source_image_file_sha256': group['source_image_file_sha256'] if index == 0 else records[index-1]['png_sha256'],
                'source_latent_sha256': group['source_latent_sha256'] if index == 0 else records[index-1]['source_latent_sha256'],
                'prompt_embeds_sha256': digest, 'attention_mask_sha256': 'a' * 64})
        runtime['condition_sha256'][qid] = seal[qid][0]['prompt_embeds_sha256']
    return bank, {'generated_source_pool': identity_binding()}, runtime, seal, registered


def test_complete_source_pool_seal_and_actual_choice_are_required(sealed):
    bank, identity, runtime, seal, registered = sealed
    proof = verify_seal(*sealed)
    assert proof['conditions'] == 108 and proof['source_condition_pairs'] == 972
    group = next(g for g in bank['groups'] if g['question_id'] in seal)
    index = source_variant_index(20260915, 279, group['question_id'])
    draw = {'training_source_variant': seal[group['question_id']][index]}
    assert verify_draw(group, draw, 279, seal) == (group['question_id'], str(index))
    draw['training_source_variant'] = seal[group['question_id']][(index + 1) % 9]
    with pytest.raises(ValueError, match='another source'):
        verify_draw(group, draw, 279, seal)
    gray = next(g for g in bank['groups'] if g['source_kind'] == 'blank_gray_1024')
    with pytest.raises(ValueError, match='gray/historical'):
        verify_draw(gray, draw, 279, seal)


@pytest.mark.parametrize('field', ['missing_condition', 'source_image_file_sha256', 'source_latent_sha256', 'event_text_sha256', 'canonical_embedding', 'pool_identity'])
def test_seal_rejects_substitution_or_incomplete_source_coverage(sealed, field):
    bank, identity, runtime, seal, registered = sealed
    qid = next(iter(seal))
    if field == 'missing_condition':
        seal.pop(qid)
    elif field == 'canonical_embedding':
        seal[qid][0]['prompt_embeds_sha256'] = '0' * 64
    elif field == 'pool_identity':
        identity['generated_source_pool']['manifest_sha256'] = '0' * 64
    else:
        seal[qid][1][field] = '0' * 64
    with pytest.raises(ValueError):
        verify_seal(bank, identity, runtime, seal, registered)


def test_generated_png_protocol_keeps_all_cases_and_rejects_wrong_parent():
    from scripts.experiments.png_readback_protocol import plan, source_spec, parent_run_name
    commit = 'c' * 40
    generated = 'ef163b26e33f62c496ed0da8744ebb7bf1163873'
    actual = plan(commit, generated)
    assert actual['parent_commit'] == generated
    assert parent_run_name(generated) == 'ef163b2-generated-source-full4832'
    for field in ('lanes', 'total_raw_rows', 'total_matched_rows', 'total_images', 'generation', 'scoring', 'chain_parity'):
        assert actual[field] == plan()[field]
    with pytest.raises(ValueError):
        source_spec(None, generated)
    with pytest.raises(ValueError):
        source_spec(commit, 'b' * 40)

