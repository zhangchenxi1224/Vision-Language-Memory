import collections
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import load_training_records, sha
from scripts.experiments.prefeval_k1_source_bank import noise_namespace
from scripts.experiments.prefeval_k1_refresh_bank import (
    archive_uncommitted_tail, freeze_refresh_bank, load_refresh_bank,
    round_bounds, source_checkpoint, validate_resume,
)
from vision_memory.training.latent_bank_unet import stable_seed


def fixture_bank(tmp_path, index=0):
    rows = [{'base_pair_id': 'p:0'}, {'base_pair_id': 'p:1'}]
    checkpoint = tmp_path/'snapshot.pt'
    checkpoint.write_bytes(b'current model')
    variants = tmp_path/'variants.json'
    variants.write_text('{}')
    root = tmp_path/f'round-{index}'
    for shard, row in enumerate(rows):
        pid = row['base_pair_id']
        out = root/f'shard-{shard}'/pid.replace(':','_')/'seed-0'
        out.mkdir(parents=True)
        hashes, writes = {}, []
        for depth in range(10):
            path = out/f'prefix-{depth:02d}.png'
            path.write_bytes(f'{pid}:actual-output:{depth}'.encode())
            hashes[path.name] = sha(path)
            writes.append({'position':depth,'output_png_sha256':sha(path),
                'source_png_sha256':writes[-1]['output_png_sha256'] if writes else None,
                'noise_seed':stable_seed(20260924,noise_namespace(pid,0,f'refresh-{index}'),depth)})
        (out/'complete.json').write_text(json.dumps({'binding':{
            'checkpoint_sha256':sha(checkpoint),'split':'train','steps':28,'cfg':1,
            'noise_chains':1,'inter_turns':9,'noise_domain':f'refresh-{index}',
            'initial_variant':index%2,'initial_variants_sha256':sha(variants)},'png_hashes':hashes}))
        (out/'writes.jsonl').write_text('\n'.join(json.dumps(w) for w in writes)+'\n')
    return root, rows, checkpoint, variants


def test_depths_are_actual_consecutive_outputs_and_bank_uses_current_snapshot(tmp_path):
    root,rows,checkpoint,variants=fixture_bank(tmp_path,index=1)
    artifact=freeze_refresh_bank(root,rows,checkpoint,variants,1)
    assert artifact['source_optimizer_step']==5840
    paths=load_refresh_bank(root,rows,sha(checkpoint),sha(variants),1)
    assert len(paths)==2 and all(len(p)==10 for p in paths.values())
    checkpoint.write_bytes(b'wrong or stale model')
    with pytest.raises(AssertionError):
        load_refresh_bank(root,rows,sha(checkpoint),sha(variants),1)
    with pytest.raises(AssertionError):
        load_refresh_bank(root,rows[:-1],artifact['source_checkpoint_sha256'],sha(variants),1)


def test_rejects_fake_recursive_bank_with_wrong_source_binding(tmp_path):
    root,rows,checkpoint,variants=fixture_bank(tmp_path)
    trace=root/'shard-0/p_0/seed-0/writes.jsonl'
    writes=[json.loads(line) for line in trace.read_text().splitlines()]
    writes[5]['source_png_sha256']=writes[0]['output_png_sha256']
    trace.write_text('\n'.join(json.dumps(w) for w in writes)+'\n')
    with pytest.raises(AssertionError):
        freeze_refresh_bank(root,rows,checkpoint,variants,0)


def test_fixed_budget_every_round_includes_all_ten_source_depths():
    initial=collections.Counter()
    retained=collections.Counter()
    for index in range(4):
        start,end=round_bounds(index)
        cycles=range(start*2//730,end*2//730)
        depths=collections.Counter(cycle%10 for cycle in cycles)
        assert len(cycles)==16 and set(depths)==set(range(10))
        for cycle in cycles:
            initial[cycle%2]+=1
            retained[cycle%10]+=1
    assert initial=={0:32,1:32} and sum(retained.values())==64
    assert (sum(initial.values())+sum(retained.values()))*730==23360*4
    assert source_checkpoint(Path('/train'),Path('/parent.pt'),0)==Path('/parent.pt')
    assert source_checkpoint(Path('/train'),Path('/parent.pt'),2).name=='checkpoint-step-011680.pt'


def test_round_resume_requires_matching_bank_and_archives_only_unsaved_tail(tmp_path):
    validate_resume(0,0,{},'bank0')
    validate_resume(5840,1,{'refresh_round':0,'source_bank_sha256':'bank0'},'bank1')
    validate_resume(6000,1,{'refresh_round':1,'source_bank_sha256':'bank1'},'bank1')
    with pytest.raises(AssertionError):
        validate_resume(6000,1,{'refresh_round':1,'source_bank_sha256':'stale'},'bank1')
    with pytest.raises(AssertionError):
        validate_resume(5700,1,{'refresh_round':0},'bank1')
    path=tmp_path/'optimization.jsonl'
    path.write_text('\n'.join(json.dumps({'step':i}) for i in range(1,6))+'\n')
    archive_uncommitted_tail(path,3)
    assert [json.loads(x)['step'] for x in path.read_text().splitlines()]==[1,2,3]
    tail=next(tmp_path.glob('optimization-uncommitted-*.jsonl'))
    assert [json.loads(x)['step'] for x in tail.read_text().splitlines()]==[4,5]


def test_refresh_noise_does_not_reuse_evaluation_noise_or_other_rounds():
    evaluation,training=set(),set()
    for row in load_training_records('train'):
        pid=row['base_pair_id']
        evaluation.update(stable_seed(20260924,noise_namespace(pid,seed),depth)
                          for seed in range(2) for depth in range(11))
        training.update(stable_seed(20260924,noise_namespace(pid,0,f'refresh-{index}'),depth)
                        for index in range(4) for depth in range(10))
    assert len(training)==730*4*10
    assert training.isdisjoint(evaluation)
