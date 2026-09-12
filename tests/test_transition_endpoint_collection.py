import copy
import json
from pathlib import Path
import tarfile
import pytest
from scripts.reporting.collect_transition_endpoint import phase_summary


def actual_baseline():
    root = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'
    bank = json.loads((root / 'transition-wording-bank-manifest.json').read_text())
    with tarfile.open(root / 'transition-wording-preflight-evidence.tgz') as archive:
        rows = [json.loads(line) for line in archive.extractfile('train/baseline/generations.jsonl').read().splitlines()]
    return rows, bank


def test_real_baseline_coverage_and_missing_seed_are_distinguished():
    rows, bank = actual_baseline()
    result, paired = phase_summary(rows, bank, 'baseline')
    assert result['correct_eos'] == 0 and result['generated_images'] == 180
    assert len(result['cells']) == 45 and len(paired) == 1350
    with pytest.raises(ValueError, match='Incomplete'):
        phase_summary(rows[:-1], bank, 'baseline')
    with pytest.raises(ValueError, match='repeated'):
        phase_summary(rows + [rows[0]], bank, 'baseline')


def test_forged_gold_tokenization_and_changed_query_image_are_rejected():
    rows, bank = actual_baseline()
    changed = copy.deepcopy(rows)
    changed[0]['scorer']['gold_token_ids'] = changed[0]['generated_token_ids'][:-1]
    changed[0]['scorer']['strict_correct'] = True
    changed[0]['scorer']['answer_followed_immediately_by_eos'] = True
    with pytest.raises(ValueError, match='Gold token'):
        phase_summary(changed, bank, 'baseline')
    changed = copy.deepcopy(rows)
    changed[1]['image_sha256'] = 'different-image-for-the-second-query'
    with pytest.raises(ValueError, match='different images'):
        phase_summary(changed, bank, 'baseline')
