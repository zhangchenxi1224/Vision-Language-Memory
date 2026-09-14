import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.experiments.fresh_wording_validation import plan, digest, all_events, all_seeds, PARENT_COMMIT
from scripts.experiments.historical_wording_protocol import plan as training_plan
from scripts.reporting.collect_broader_endpoint import GOLD_IDS
from scripts.reporting.collect_broader_validation import expected_rows, summarize_rows


@pytest.fixture(scope='module')
def fixed():
    bank=json.loads((Path(__file__).resolve().parents[1]/'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    training=training_plan(bank,PARENT_COMMIT)
    return bank,training,plan(training,bank)


def test_all_events_are_new_and_training_semantics_are_unchanged(fixed):
    bank,training,fresh=fixed
    artifact=Path(__file__).resolve().parents[1]/'reports/official-alignment-results-20260913/fresh-wording-preregistered-validation.json'
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==digest(fresh)=='ba77e8c2ab8742bdcda703ae12071b504ea99c33173d0ebad314ddc10292fb6b'
    assert json.loads(artifact.read_bytes())==fresh
    assert digest(training)=='a35d3986439dab371f3bfd90243ed948bba3c3a4f4fd0887d3021e38dead081e'
    assert all_events(training).isdisjoint(all_events(fresh))
    assert all_seeds(training).isdisjoint(all_seeds(fresh))
    assert all_events(fresh).isdisjoint({event for values in training['training_augmentation']['events'].values() for event in values})
    assert fresh['training_augmentation']==training['training_augmentation']
    for before,after in zip(training['prefix_validation']['cases'],fresh['prefix_validation']['cases'],strict=True):
        for key in ('target_index','question_id','event_semantics','gold','question_variants'):
            assert before[key]==after[key]
        assert len(before['event_text'].splitlines())==len(after['event_text'].splitlines())
    with pytest.raises(ValueError,match='overlaps'):
        changed=copy.deepcopy(bank)
        changed['groups'][0]['event_text']=next(iter(all_events(fresh)))
        plan(training,changed)


@pytest.mark.parametrize('mode,lane,raw,matched,images',[
    ('single_writes',None,390,360,72),('rgb_chains',None,480,480,96),
    ('historical_prefixes',0,560,480,96),('historical_prefixes',1,560,480,96)])
def test_complete_fresh_matrix_keeps_every_cell_and_strict_scorer(fixed,mode,lane,raw,matched,images):
    bank,training,fresh=fixed
    expected,_,_=expected_rows(fresh,bank,mode,lane)
    rows=[]
    for meta in expected.values():
        rows.append({**meta,'image_sha256':hashlib.sha256(meta['image_artifact'].encode()).hexdigest(),
            'raw':meta['gold'],'generated_token_ids':GOLD_IDS[meta['gold']]+[151645],
            'scorer':{'gold_token_ids':GOLD_IDS[meta['gold']],'strict_correct':True,'answer_followed_immediately_by_eos':True}})
    summary,_,_=summarize_rows(rows,fresh,bank,mode,lane)
    assert (summary['raw_rows'],summary['matched_correct_eos'],summary['generated_images'])==(raw,matched,images)
    if mode=='rgb_chains':
        assert summary['complete_chains_correct_eos']==16
    with pytest.raises(ValueError):
        summarize_rows(rows,training,bank,mode,lane)
    changed=copy.deepcopy(rows)
    row=next(row for row in changed if row['condition']=='matched')
    row['generated_token_ids']=GOLD_IDS[row['gold']]+[13,151645]
    row['scorer']['answer_followed_immediately_by_eos']=False
    failed,_,_=summarize_rows(changed,fresh,bank,mode,lane)
    assert failed['matched_correct_eos']==matched-1


def test_actual_cli_registration_accepts_only_the_fresh_sequence(fixed):
    from scripts.probes.rgb_package_parity import replay_registration
    bank,training,fresh=fixed
    identity={'bank_sha256':'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592',
        'plan_file_sha256':digest(training),'validation_set':'fresh_wording_v1','validation_plan_sha256':digest(fresh),
        'selected_cases':fresh['transition_validation']['rgb_chains'],'registered_plan':fresh,
        'mode':'rgb_chains','parent_commit':PARENT_COMMIT}
    assert replay_registration(identity,broader=True,logical_sampling_commit=PARENT_COMMIT)==fresh['transition_validation']['rgb_chains'][0]
    changed=copy.deepcopy(identity)
    changed['selected_cases'][0]['steps'][0]['event_text']='Substitute a previously tested expression.'
    with pytest.raises(ValueError,match='sequence changed'):
        replay_registration(changed,broader=True,logical_sampling_commit=PARENT_COMMIT)


def test_continuation_reuses_every_observed_case_and_preserves_lineage(fixed):
    from scripts.experiments.clear_retention_protocol import plan as continuation_plan
    from scripts.reporting.collect_broader_endpoint import CLEAR_RETENTION_COMMIT, registered_protocol
    bank, _, original = fixed
    training = continuation_plan(bank, CLEAR_RETENTION_COMMIT)
    assert registered_protocol(bank, CLEAR_RETENTION_COMMIT)[1] == training
    regression = plan(training, bank)
    assert regression['transition_validation'] == original['transition_validation']
    assert regression['prefix_validation'] == original['prefix_validation']
    assert regression['reused_validation_plan_sha256'] == digest(original)
    assert 'not a new holdout' in regression['validation_exposure']
    assert regression['parent_checkpoint_sha256'] == training['parent_checkpoint_sha256']
    assert regression['optimizer']['lr'] == 1e-5
    for mode, lane in (('single_writes', None), ('rgb_chains', None), ('historical_prefixes', 0), ('historical_prefixes', 1)):
        assert expected_rows(regression, bank, mode, lane) == expected_rows(original, bank, mode, lane)
    changed = copy.deepcopy(training)
    changed['optimizer']['lr'] = 5e-5
    with pytest.raises(ValueError, match='fixed training registration'):
        plan(changed, bank)
