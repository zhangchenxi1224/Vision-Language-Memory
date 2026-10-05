from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_writer_readout import expected_keys, cohort_rows, save_once
from scripts.reporting.writer_readout_report import summarize
from scripts.inspire.run_writer_readout import accrued_seconds

SPEC = json.loads((ROOT / 'configs/experiments/context_readout_audit.json').read_text())


def test_full_cohorts_no_train_dev_overlap_and_denominator():
    groups = cohort_rows()
    assert [len(groups[k]) for k in ('pilot','dev')] == [16,90]
    ids = [r['base_pair_id'] for g in groups.values() for r in g]
    assert len(set(ids)) == 106
    assert len(expected_keys(ids, SPEC['queries'])) == 17808


def test_incomplete_report_is_rejected():
    with pytest.raises(ValueError, match='denominator'):
        summarize([], {'dev':['a','b']}, SPEC)


def synthetic():
    return [dict(pair_id=p, query_id=q, endpoint=e, control=c, chain=n,
                 kl=0 if c == 'text' else (4 if e == 'b730' or c in ('blank','mismatch') else 1+2*n),
                 target_ids=[7,9], teacher_logits_sha256='same')
            for p,q,e,c,n in expected_keys(['a','b'],SPEC['queries'])]


def test_preference_is_independent_unit_noise_is_averaged():
    report = summarize(synthetic(), {'dev':['a','b']}, SPEC)
    value = report['cohorts']['dev']['families']['recall']['endpoints']['direct-20261005']
    assert value['memory_kl'] == 2
    assert value['baseline_minus_memory']['mean'] == 2
    assert value['baseline_minus_memory']['independent_n'] == 2


def test_mismatched_teacher_prefix_and_duplicate_rows_rejected():
    values = synthetic()
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(values+[values[0]], {'dev':['a','b']}, SPEC)
    values[0]['target_ids'] = [1,9]
    with pytest.raises(ValueError, match='Unpaired'):
        summarize(values, {'dev':['a','b']}, SPEC)


def test_unsettled_cost_and_mutated_identity_rejected(tmp_path):
    (tmp_path/'attempts').mkdir()
    p = tmp_path/'attempts'/'one.json'
    p.write_text(json.dumps({'started':10}))
    with pytest.raises(ValueError, match='Unsettled'):
        accrued_seconds(tmp_path)
    p.write_text(json.dumps({'started':10,'finished':70,'exit_code':124}))
    assert accrued_seconds(tmp_path) == 60
    ident = tmp_path/'identity.json'
    save_once(ident, {'weight':'one'})
    with pytest.raises(ValueError, match='changed'):
        save_once(ident, {'weight':'two'})
