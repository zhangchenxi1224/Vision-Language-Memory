import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import torch

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN = ROOT/'runs/prefeval-b730-exposure512-20260927'
OLD = ROOT/'runs/prefeval-b-mcq-20260925/robust730'
CODE = RUN/'code'
OUT = RUN/'evidence/step-046720-dev-20260927'
assert not OUT.exists(), 'Inspect existing evidence before retrying'
OUT.mkdir(parents=True)
sys.path[:0] = [str(CODE), str(CODE/'src')]
from scripts.experiments.prefeval_k1_data import load_records, load_training_records, official_mcq
from vision_memory.training.latent_bank_unet import stable_seed

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
assert commit == '656fdf029c4a7c05e53bccb75483a03cf62d5f54'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
endpoint_path = RUN/'train/checkpoint-step-046720.pt'
endpoint = torch.load(endpoint_path,map_location='cpu',weights_only=False)
assert endpoint['optimizer_step'] == 46720
assert endpoint['manifest']['steps'] == 93440 and endpoint['manifest']['snapshot_steps'] == [46720,70080]
assert len(endpoint['trainable_state']) == 1075
endpoint_hash = sha(endpoint_path)
old_hash = '6c8eb92f629ccdf6932407124190d651e8231058ef6f232a0a59380370d44c8d'
del endpoint
parser = official_mcq(CODE/'third_party/prefeval_reference')['extract_choice']
families = ['T1','T2','T3','O1','O2']
ids = {r['base_pair_id'] for r in load_records('dev')}
assert len(ids) == 90
expected = {(pid,c,k,f) for pid in ids for f in families for k in ['memory','mismatch','blank','text'] for c in (range(2) if k in ['memory','mismatch'] else range(1))}
summaries, pngs = {}, {}
for variant in [0,1]:
    prior = None
    for step in [23360,46720]:
        label = f'step-{step:06d}-dev-V{variant}'
        source = (OLD/f'readback-final-dev-V{variant}' if step == 23360 else RUN/'readback'/label)
        shutil.copytree(source,OUT/label)
        rows = [json.loads(line) for p in sorted(source.glob('readback-*.jsonl')) for line in p.read_text().splitlines()]
        keyed = {(r['pair_id'],r['chain'],r['control'],r['family']):r for r in rows}
        assert len(rows) == len(keyed) == 2700 and set(keyed) == expected
        for row in rows:
            assert row['task'] == 'mcq' and row['prefix'] == 0
            assert row['correct'] == (parser(row['generated']['raw']) == row['correct_letter'])
            if row['control'] in ['memory','mismatch']:
                p = Path(row['png_path'])
                if str(p) not in pngs:
                    complete = json.loads((p.parent/'complete.json').read_text())
                    digest = sha(p)
                    assert digest == complete['png_hashes'][p.name]
                    binding = complete['binding']
                    assert binding['checkpoint_sha256'] == (old_hash if step == 23360 else endpoint_hash)
                    assert binding['initial_variant'] == variant and binding['split'] == 'dev'
                    metadata = OUT/'png-metadata'/label/p.parent.parent.name/p.parent.name
                    metadata.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(p.parent/'complete.json',metadata/'complete.json')
                    shutil.copy2(p.parent/'writes.jsonl',metadata/'writes.jsonl')
                    pngs[str(p)] = dict(path=str(p),sha256=digest,bytes=p.stat().st_size,variant=variant,step=step,label=label)
                assert row['png_sha256'] == pngs[str(p)]['sha256']
        results = {}
        for family in families:
            controls = {}
            for control in ['memory','mismatch','blank','text']:
                selected = [r for r in rows if r['family']==family and r['control']==control]
                controls[control] = dict(correct=sum(r['correct'] for r in selected),total=len(selected),parse_failure=sum(r['parse_failure'] for r in selected),truncated=sum(r['generated']['truncated'] for r in selected))
            pairs = [(keyed[p,c,'memory',family]['correct'],keyed[p,c,'mismatch',family]['correct']) for p in ids for c in range(2)]
            result = dict(controls=controls,match_minus_mismatch=sum(a-b for a,b in pairs),repaired=sum(a and not b for a,b in pairs),regressed=sum(b and not a for a,b in pairs),both_noise_correct=sum(all(keyed[p,c,'memory',family]['correct'] for c in range(2)) for p in ids),independent_preferences=90)
            if prior:
                result['versus_step23360'] = dict(fixed=sum(keyed[p,c,'memory',family]['correct'] and not prior[p,c,'memory',family]['correct'] for p in ids for c in range(2)),lost=sum(prior[p,c,'memory',family]['correct'] and not keyed[p,c,'memory',family]['correct'] for p in ids for c in range(2)))
            results[family] = result
        summary = dict(split='dev',record_count=2700,families=results,files={p.name:sha(p) for p in source.glob('readback-*.jsonl')})
        summaries[label] = summary
        if step == 46720:
            assert summary == json.loads((source/'summary.json').read_text())
        assert sum(r['label']==label for r in pngs.values()) == 180
        prior = keyed
save('summaries.json',summaries)
save('actual-png-hashes.json',list(pngs.values()))

snapshot = OUT/'resume-observed.pt'
os.link(RUN/'train/resume.pt',snapshot)
saved = torch.load(snapshot,map_location='cpu',weights_only=False)
step = saved['optimizer_step']
assert 46720 <= step < 70080 and saved['episode_cursor'] == step*4
assert len(saved['optimizer']['state']) == 1075 and {int(v['step']) for v in saved['optimizer']['state'].values()} == {step}
assert saved['manifest']['steps'] == 93440 and saved['manifest']['snapshot_steps'] == [46720,70080]
assert saved['manifest']['continued_from_sha256'] == '0acddb16b60f6f6e128d59041fe97db356a3022bc32da1fb8ed8b9813b5136f7'
assert len(saved['rng_state']['torch_cuda']) == 1
records = []
for line in (RUN/'train/optimization.jsonl').read_text().splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row['step']<=step:
        records.append(row)
assert [r['step'] for r in records] == list(range(1,step+1))
training_rows = load_training_records('train')
orders, exposures = {}, collections.Counter()
for row in records:
    for micro,d in enumerate(row['draws']):
        if row['step'] <= 46720:
            exposures[d['pair_id'],d['initial_variant']] += 1
        if row['step'] <= 23360:
            continue
        draw = (row['step']-1)*4+micro
        cycle, offset = divmod(draw,730)
        if cycle not in orders:
            orders[cycle] = torch.randperm(730,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
        assert d['pair_id'] == training_rows[orders[cycle][offset]]['base_pair_id'] and d['position']==0 and d['initial_variant']==cycle%2
        assert d['sigma'] == float(torch.rand((),generator=torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))))
assert len(exposures)==1460 and set(exposures.values())=={128}
(OUT/'optimization-committed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
save('checkpoint-evidence.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),frozen_commit=commit,endpoint_step=46720,endpoint_checkpoint_sha256=endpoint_hash,endpoint_trainable_tensors=1075,endpoint_exposure_each_variant=128,endpoint_exposure_total=256,endpoint_all_730_exposure_counts_verified=True,committed_step=step,episode_cursor=step*4,optimizer_states=1075,all_optimizer_steps_match=True,full_log_contiguous_unique=True,all_new_draws_and_sigma_match_original_schedule=True,rng_keys=list(saved['rng_state']),resume_sha256=sha(snapshot),resume_bytes=snapshot.stat().st_size,preserved_remote_snapshot=str(snapshot),final_step=93440,snapshot_steps=[46720,70080]))
for name in ['train/manifest.json','train/continuation.json','controller.json','latest-runtime.json']:
    target = OUT/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(RUN/name,target)
for variant in [0,1]:
    label = f'step-046720-dev-V{variant}'
    for stage in ['rollout','readback']:
        for category,suffix in [('logs','log'),('processes','json')]:
            name = f'{category}/{label}-{stage}.{suffix}'
            target = OUT/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(RUN/name,target)
shutil.copy2(__file__,OUT/'archive-script.py')
archive = OUT/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p not in [snapshot,archive]:
            tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
receipt = dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,png_hash_count=len(pngs),record_count=10800,committed_step=step,endpoint_step=46720)
save('archive-receipt.json',receipt)
print(json.dumps(receipt,indent=2))
