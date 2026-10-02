import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reporting.summarize_prefeval_k1 import summarize
from scripts.experiments.prefeval_k1_data import load_records

def sample_rows():
    path=ROOT/'reports/prefeval-k1-l0-l2-20260924/evidence/first-frozen-teacher-B-T1.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def write_rows(tmp_path,rows):
    (tmp_path/'readback-0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

def test_one_success_is_not_full_dataset_accuracy(tmp_path):
    write_rows(tmp_path,sample_rows())
    value=summarize([('B',tmp_path)],expected_ids=[r['base_pair_id'] for r in load_records()],prefixes=[0],chains=1)
    row=next(r for r in value['metrics'] if r['control']=='memory' and r['family']=='T1' and r['task']=='mcq')
    assert row['correct']==1 and row['observed']==1 and row['expected']==64
    assert row['accuracy'] is None and row['observed_scored_accuracy']==1
    free=next(r for r in value['metrics'] if r['control']=='memory' and r['family']=='T1' and r['task']=='free')
    assert free['scored']==0 and free['accuracy'] is None

def test_joint_ood_rejects_different_images(tmp_path):
    base=next(r for r in sample_rows() if r['control']=='memory' and r['task']=='mcq')
    first=copy.deepcopy(base)
    second=copy.deepcopy(base)
    first['family']='O1'
    second['family']='O2'
    second['png_sha256']='different-image'
    write_rows(tmp_path,[first,second])
    with pytest.raises(ValueError,match='same frozen PNG'):
        summarize([('B',tmp_path)],expected_ids=[base['pair_id']],prefixes=[0],chains=1)


def test_noise_chains_share_one_preference_denominator(tmp_path):
    first=copy.deepcopy(next(r for r in sample_rows() if r['control']=='memory' and r['task']=='mcq'))
    second=copy.deepcopy(first)
    second.update(chain=1,png_sha256='second_generated_image',correct=False,predicted_letter=None,parse_failure=True)
    second['generated']['raw']='unparseable answer'
    write_rows(tmp_path,[first,second])
    value=summarize([('B',tmp_path)],expected_ids=[first['pair_id']],prefixes=[0],chains=2)
    row=next(r for r in value['noise_chain_joint'] if r['family']=='T1' and r['task']=='mcq')
    assert row['expected']==row['observed']==1 and row['accuracy']==0
    write_rows(tmp_path,[first])
    incomplete=summarize([('B',tmp_path)],expected_ids=[first['pair_id']],prefixes=[0],chains=2)
    row=next(r for r in incomplete['noise_chain_joint'] if r['family']=='T1' and r['task']=='mcq')
    assert row['expected']==1 and row['observed']==0 and row['accuracy'] is None


def test_failed_judge_is_visible_and_not_scored_as_wrong(tmp_path):
    row=next(r for r in sample_rows() if r['control']=='memory' and r['task']=='free')
    write_rows(tmp_path,[row])
    judge_dir=tmp_path/'judges'
    judge_dir.mkdir()
    (judge_dir/'failure.json').write_text(json.dumps({'input':row,'status':'judge_parse_failure',
        'judge_model':'fixed-model','model_label':'substitute_judge'}),encoding='utf-8')
    value=summarize([('B',tmp_path)],expected_ids=[row['pair_id']],prefixes=[0],chains=1,judge_dir=judge_dir)
    metric=next(r for r in value['metrics'] if r['control']=='memory' and r['family']=='T1' and r['task']=='free')
    assert metric['expected']==metric['observed']==metric['parse_failures']==1
    assert metric['scored']==0 and metric['accuracy'] is None and metric['observed_scored_accuracy'] is None
