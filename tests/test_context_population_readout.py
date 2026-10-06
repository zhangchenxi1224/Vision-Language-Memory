import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.inspire import run_context_population_readout as m


def test_frozen_partition_donors_and_denominators():
    spec = m.read(m.PROTOCOL); groups = m.strata(); donors, membership = m.donors_and_groups()
    ids = [r['base_pair_id'] for rs in groups.values() for r in rs]
    assert len(set(ids)) == 32 and all(len(rs) == 16 for rs in groups.values())
    for p, d in donors.items():
        assert p != d and p.split(':')[0] == d.split(':')[0] and membership[p] == membership[d]
    assert len(m.expected_keys(ids, spec)) == 3840
    assert len(m.reused_keys(spec)) == 1152
    assert len(m.current_keys(ids, spec)) == 2688
    assert not m.current_keys(ids, spec) & m.reused_keys(spec)
    assert len(m.current_keys([r['base_pair_id'] for r in groups['original16']], spec)) == 768
    assert len(m.current_keys([r['base_pair_id'] for r in groups['added16']], spec)) == 1920


def synthetic():
    spec = m.read(m.PROTOCOL); ids = [r['base_pair_id'] for rs in m.strata().values() for r in rs]
    records = []
    for key in m.expected_keys(ids, spec):
        pid, qid, e, c, n = key
        r = dict(zip(m.KEYS, key))
        r.update(kl={'writer16': 3., 'writer32': 2., 'blank': 4., 'text': 0.}[e] + (c == 'mismatch'),
            teacher_target=pid+qid, target_ids=[1, 2], teacher_logits_sha256=pid+qid)
        records.append(r)
    return spec, records


def test_stratified_preference_means_not_question_pseudoreplication():
    spec, rows = synthetic(); out = m.summarize(rows, spec)
    assert out['combined_rows'] == 3840
    for s in out['strata'].values():
        assert s['independent_n'] == 16
        for f in s['families'].values():
            assert f['writer16_minus_writer32']['mean'] == 1.
            assert f['endpoints']['writer32']['mismatch_minus_memory']['mean'] == 1.


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'foreign', 'teacher', 'nan'])
def test_corrupt_denominator_or_pairing_rejected(fault):
    spec, rows = synthetic()
    if fault == 'missing': rows.pop()
    if fault == 'duplicate': rows.append(dict(rows[0]))
    if fault == 'foreign': rows[0]['pair_id'] = 'foreign'
    if fault == 'teacher': rows[0]['teacher_logits_sha256'] = 'wrong'
    if fault == 'nan': rows[0]['kl'] = float('nan')
    with pytest.raises(ValueError): m.summarize(rows, spec)


def test_shared_iteration_budget_counts_first_phase_failures_once(tmp_path):
    for name, cost in [('context-dev-v1', 600), ('context-population32-v1', 1900), ('context-population-readout-v1', 100)]:
        p = tmp_path/name/'attempts'; p.mkdir(parents=True)
        (p/'a.json').write_text(json.dumps(dict(started=10, finished=10+cost, exit_code=124)))
        r = tmp_path/name/'receipts'; r.mkdir(); (r/'a.json').write_text(json.dumps(dict(started=10, finished=10+cost)))
    assert m.remaining_seconds(tmp_path/'context-population-readout-v1') == (8800, 600, 1900, 100)
    (tmp_path/'context-population32-v1/attempts/live.json').write_text(json.dumps(dict(started=10)))
    with pytest.raises(ValueError, match='Unsettled'): m.remaining_seconds(tmp_path/'context-population-readout-v1')


def test_only_new_images_generated_no_training_or_selection(tmp_path):
    args = SimpleNamespace(output=tmp_path, base=Path('/base'), official_source=Path('/official'))
    jobs = m.build_jobs(args)
    assert [j['gpu'] for j in jobs] == [0, 1]
    for job, ids, endpoint in zip(jobs, (m.ADDED_IDS, m.ALL_IDS), m.ENDPOINTS):
        cmd = job['command']
        assert cmd[2] == 'rollout' and cmd[cmd.index('--ids-file')+1] == str(ids)
        assert cmd[cmd.index('--checkpoint')+1] == str(m.checkpoints()[endpoint][0])
        assert cmd[cmd.index('--inter-turns')+1] == '0' and '--snapshot-steps' not in cmd


def test_exact_donor_png_prefix_and_gray_binding():
    spec = m.read(m.PROTOCOL); donors, groups = m.donors_and_groups(); pid = next(iter(groups)); q = spec['queries'][0]
    meta = dict(path='/cache/t.pt', target_ids=[1, 2], logits_sha256='target')
    asset = dict(path='/images/m.png', sha256='png', complete_sha256='done')
    assets = {f'writer32|{donors[pid]}|0': asset}; targets = {pid+'|'+q['id']: meta}
    row = dict(pair_id=pid, query_id=q['id'], endpoint='writer32', control='mismatch', chain=0,
        split='pilot', family=q['family'], kl=1., png=asset, donor_pair_id=donors[pid], teacher_target=meta['path'],
        target_ids=meta['target_ids'], teacher_logits_sha256=meta['logits_sha256'])
    m.validate_row(row, targets, assets, spec)
    for field, value in [('donor_pair_id', pid), ('png', dict(asset, sha256='bad')), ('target_ids', [9])]:
        with pytest.raises(ValueError): m.validate_row(dict(row, **{field: value}), targets, assets, spec)
    gray = dict(row, endpoint='blank', control='blank', png=None, donor_pair_id=None)
    m.validate_row(gray, targets, assets, spec)
    with pytest.raises(ValueError): m.validate_row(dict(gray, png=asset), targets, assets, spec)
