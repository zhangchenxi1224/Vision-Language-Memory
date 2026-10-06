import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire.run_context_population import population,remaining_seconds,build_jobs,PARENT_SHA,OLD


def test_same_topics_disjoint_added_histories_and_no_dev():
    rows,new=population()
    old=json.loads((ROOT/'configs/experiments/context_coverage_ids.json').read_text())['ids']
    assert len(rows)==32 and len(new)==16
    assert not set(old)&{r['base_pair_id'] for r in new}
    assert {r['topic'] for r in rows}=={p.split(':')[0] for p in old}


def test_fixed_budget_charges_all_prior_attempts_and_no_receipt_double_count(tmp_path):
    for name,cost in [('context-dev-v1',600),('context-fit-v1',300),('context-population32-v1',200)]:
        path=tmp_path/name/'attempts';path.mkdir(parents=True)
        (path/'x.json').write_text(json.dumps(dict(started=0,finished=cost,exit_code=124)))
        rec=tmp_path/name/'receipts';rec.mkdir();(rec/'x.json').write_text(json.dumps(dict(started=0,finished=cost)))
    assert remaining_seconds(tmp_path/'context-population32-v1')==(5200,900,200)
    (tmp_path/'context-population32-v1/attempts/live.json').write_text(json.dumps(dict(started=0)))
    with pytest.raises(ValueError,match='Unsettled'):remaining_seconds(tmp_path/'context-population32-v1')


def test_only_new_teacher_rows_and_writer_starts_at_common_parent(tmp_path):
    args=SimpleNamespace(output=tmp_path,base=Path('/base'),reader=Path('/reader'),prefeval=Path('/prefeval'),official_source=Path('/official'))
    jobs=build_jobs(args,'teachers');assert [j['gpu'] for j in jobs]==[0,1]
    for j in jobs:
        cmd=j['command'];assert cmd[cmd.index('--ids-file')+1].endswith('context_population_added16_ids.json')
        assert cmd[cmd.index('--steps')+1]=='288' and cmd[cmd.index('--shards')+1]=='2'
    cmd=build_jobs(args,'writer')[0]['command'];assert cmd[2]=='train'
    assert cmd[cmd.index('--checkpoint')+1]==str(OLD/'train/checkpoint-final.pt')
    assert cmd[cmd.index('--steps')+1]=='128' and '--snapshot-steps' not in cmd
