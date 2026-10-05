from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_route_functional import (
    old_keys,new_keys,index_rows,summarize,cohort_rows,donor_map,load_target,validate_new_row,
)
from scripts.inspire.run_route_functional import remaining_seconds
SPEC=json.loads((ROOT/'configs/experiments/context_readout_audit.json').read_text())


def synthetic(keys):
    return [dict(pair_id=p,query_id=q,endpoint=e,control=c,chain=n,
        kl=0 if c=='text' else (1+2*n if e.startswith('pm-fm') and c=='memory' else 4),
        target_ids=[7,9],teacher_logits_sha256='same',teacher_target=p+'-'+q)
        for p,q,e,c,n in keys]


def test_exact_frozen_denominators_and_donors():
    rows=cohort_rows()['dev']
    ids=[r['base_pair_id'] for r in rows]
    assert len(old_keys(ids,SPEC['queries']))==15120
    assert len(new_keys(ids,SPEC))==8640
    assert len(old_keys(ids,SPEC['queries'])|new_keys(ids,SPEC))==23760
    topics={r['base_pair_id']:r['topic'] for r in rows}
    assert all(p!=d and topics[p]==topics[d] for p,d in donor_map(rows).items())


def test_route_pairing_noise_averaged_within_preference():
    ids=['a','b']
    old=synthetic(old_keys(ids,SPEC['queries']))
    new=synthetic(new_keys(ids,SPEC))
    report=summarize(old,new,ids,SPEC)
    value=report['families']['recall']['endpoints']['pm-fm-20261005']
    assert value['memory_kl']==2
    assert value['direct_minus_memory']['mean']==2
    assert value['direct_minus_memory']['independent_n']==2


def test_incomplete_foreign_duplicate_and_unpaired_rejected():
    ids=['a','b']
    old=synthetic(old_keys(ids,SPEC['queries']))
    new=synthetic(new_keys(ids,SPEC))
    for broken in (new[:-1],new+[new[0]],[] ):
        with pytest.raises(ValueError):
            summarize(old,broken,ids,SPEC)
    new[0]['target_ids']=[8,9]
    with pytest.raises(ValueError,match='Unpaired'):
        summarize(old,new,ids,SPEC)


def test_cache_missing_never_generated(tmp_path):
    p=tmp_path/'missing.pt'
    with pytest.raises(ValueError,match='Read-only'):
        load_target(dict(path=str(p),file_sha256='x'),{}, {},'query',9,0)
    assert not p.exists()


def test_binding_mutation_rejected():
    asset=dict(path='one.png',sha256='123')
    targets={'a|q':dict(path='cache.pt',target_ids=[7,9],logits_sha256='x')}
    row=dict(pair_id='a',query_id='q',endpoint='pm-fm-20261005',control='mismatch',chain=0,
        split='dev',family='recall',donor_pair_id='b',png=asset,teacher_target='cache.pt',
        target_ids=[7,9],teacher_logits_sha256='x')
    args=(targets,{'pm-fm-20261005|b|0':asset},{'a':'b'},{'q':{'family':'recall'}})
    validate_new_row(row,*args)
    for key,value in [('donor_pair_id','c'),('target_ids',[7]),('png',{'path':'wrong'})]:
        altered=deepcopy(row); altered[key]=value
        with pytest.raises(ValueError,match='binding'):
            validate_new_row(altered,*args)


def test_budget_counts_failed_attempt_once_not_success_receipt(tmp_path):
    out=tmp_path/'route-functional-v1'
    (out/'attempts').mkdir(parents=True)
    (out/'receipts').mkdir()
    attempt={'started':0,'finished':120,'exit_code':124}
    (out/'attempts/a.json').write_text(json.dumps(attempt))
    (out/'receipts/a.json').write_text(json.dumps(attempt))
    left,prior,current=remaining_seconds(out)
    assert (left,prior,current)==(3480,0,120)
    (out/'attempts/b.json').write_text(json.dumps({'started':121}))
    with pytest.raises(ValueError,match='Unsettled'):
        remaining_seconds(out)


def test_real_png_contract_and_file_digest_not_object_digest(tmp_path,monkeypatch):
    import scripts.experiments.prefeval_route_functional as module
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    source=tmp_path/'source.png'
    Image.new('RGB',(1024,1024),(128,128,128)).save(source)
    png_bytes=source.read_bytes()
    png_sha=module.sha(source)
    def write(p,value):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for seed in module.SEEDS:
        train=tmp_path/'route-comparison-v1'/('seed-'+seed)
        train.mkdir(parents=True)
        (train/'checkpoint-final.pt').write_bytes(seed.encode())
        checksum=module.sha(train/'checkpoint-final.pt')
        monkeypatch.setitem(module.CHECKPOINTS,seed,checksum)
        (train/'optimization.jsonl').write_text('\n'.join(json.dumps({'step':i}) for i in range(1,129)))
        write(train/'complete.json',dict(status='completed',steps=128,targets=512,
            checkpoint_sha256=checksum,optimization_sha256=module.sha(train/'optimization.jsonl')))
        out=tmp_path/'route-readback-v1/pm-fm'/('seed-'+seed)/'V0'
        m=dict(checkpoint_sha256=checksum,split='dev',steps=28,cfg=1,noise_chains=2,inter_turns=0,
            state='only reopened uint8 RGB PNG; fresh Gaussian each write',initial_variants_sha256=module.VARIANT_SHA,initial_variant=0)
        write(out/'images/manifest.json',m)
        write(out/'images-validated.json',dict(pngs=180,manifest_sha256=module.sha(out/'images/manifest.json')))
        for row in cohort_rows()['dev']:
            pid=row['base_pair_id']
            for chain in range(2):
                folder=out/'images'/pid.replace(':','_')/f'seed-{chain}'
                write(folder/'complete.json',dict(binding=m,png_hashes={'prefix-00.png':png_sha}))
                (folder/'prefix-00.png').write_bytes(png_bytes)
                (folder/'writes.jsonl').write_text(json.dumps(dict(position=0,source_png_sha256=None,
                    output_png_sha256=png_sha,noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),
                    event=module.event_text(row['history'][:2]))))
    assert len(module.verify_assets(tmp_path)['assets'])==360
    # A valid-looking PNG with a different history must not enter the paired report.
    writes=next((tmp_path/'route-readback-v1').glob('pm-fm/*/V0/images/*/seed-*/writes.jsonl'))
    item=json.loads(writes.read_text()); item['event']='different history'
    writes.write_text(json.dumps(item))
    with pytest.raises(ValueError,match='event or noise'):
        module.verify_assets(tmp_path)
