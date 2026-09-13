import copy
import json
from pathlib import Path
import tarfile

import pytest

from scripts.experiments.build_historical_writer_bank import positive_controls, donor_member
from test_latent_bank_unet import write_bank
from vision_memory.training.latent_bank_unet import load_teacher_bank

RESULTS = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'


def panel():
    return json.loads((RESULTS / 'historical-fp32-readback-panel.json').read_bytes())


def test_original_historical_question_bytes_require_explicit_contract(tmp_path):
    path = write_bank(tmp_path)
    bank = json.loads(path.read_bytes())
    for target in panel()['targets']:
        bank['groups'][0].update(question_variants=target['question_variants'], answer=target['gold'])
        bank['teachers'][0]['answer'] = target['gold']
        bank['groups'][0].pop('question_instruction_contract', None)
        path.write_text(json.dumps(bank))
        with pytest.raises(ValueError, match='instruction suffix'):
            load_teacher_bank(path)
        bank['groups'][0]['question_instruction_contract'] = 'historical-r11-five-prompts/v1'
        path.write_text(json.dumps(bank))
        loaded, _ = load_teacher_bank(path)
        assert loaded['groups'][0]['question_variants'] == target['question_variants']
    bank['groups'][0]['question_variants']['original_open'] = 'Added hidden instruction\n' + bank['groups'][0]['question_variants']['original_open']
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match='instruction suffix'):
        load_teacher_bank(path)


def test_actual_full_readback_preserves_all64_and_rewrite_failures():
    with tarfile.open(RESULTS / 'historical-fp32-readback-evidence.tgz') as archive:
        rows = [json.loads(line) for line in archive.extractfile('generations.jsonl').read().splitlines()]
    positive, summary = positive_controls(rows, panel())
    assert len(positive) == 64 and summary['members_passing_both_forms_all_five'] == 57
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r['condition'] == 'matched' and r['prompt_id'] == 'original_open')
    row['generated_token_ids'] = [0, 151645]
    row['scorer']['strict_correct'] = False
    with pytest.raises(ValueError, match='do not silently discard'):
        positive_controls(changed, panel())


def test_donor_rule_uses_fixed_metadata_and_preserves_same_topic_when_possible():
    value = panel()
    for target in value['targets']:
        member, selected = donor_member(target, value['targets'], value['members'])
        assert member['seed'] == 0 and selected['gold'] != target['gold']
        if any(other['topic'] == target['topic'] and other['gold'] != target['gold'] for other in value['targets']):
            assert selected['topic'] == target['topic']
