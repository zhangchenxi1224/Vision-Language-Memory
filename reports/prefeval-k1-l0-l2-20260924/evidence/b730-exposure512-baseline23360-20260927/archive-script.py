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
OUT = RUN/'evidence/baseline-23360-20260927'
assert not OUT.exists(), 'Evidence directory already exists; inspect before retrying'
OUT.mkdir(parents=True)
sys.path[:0] = [str(CODE), str(CODE/'src')]
from scripts.experiments.prefeval_k1_data import load_records, load_training_records, official_mcq
from vision_memory.training.latent_bank_unet import stable_seed

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')

assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=CODE, text=True).strip() == '656fdf029c4a7c05e53bccb75483a03cf62d5f54'
assert not subprocess.check_output(['git','status','--porcelain'], cwd=CODE, text=True).strip()
ids = {r['base_pair_id'] for r in load_records('train')}
assert len(ids) == 730
parser = official_mcq(CODE/'third_party/prefeval_reference')['extract_choice']
summaries = {}
pngs = {}
for variant, source in [(0, RUN/'readback/step-023360-train-V0'), (1, OLD/'readback-final-train-V1')]:
    dest = OUT/f'V{variant}'
    shutil.copytree(source, dest)
    rows = [json.loads(line) for p in source.glob('readback-*.jsonl') for line in p.read_text().splitlines()]
    keyed = {(r['pair_id'], r['chain'], r['control'], r['family']):r for r in rows}
    expected = {(pid, chain, control, 'T1') for pid in ids for control in ['memory','mismatch','blank','text'] for chain in (range(2) if control in ['memory','mismatch'] else range(1))}
    assert len(rows) == len(keyed) == 4380 and set(keyed) == expected
    for row in rows:
        assert row['task'] == 'mcq' and row['prefix'] == 0
        assert row['correct'] == (parser(row['generated']['raw']) == row['correct_letter'])
        if row['control'] in ['memory','mismatch']:
            p = Path(row['png_path'])
            if str(p) not in pngs:
                complete = json.loads((p.parent/'complete.json').read_text())
                digest = sha(p)
                assert digest == complete['png_hashes'][p.name]
                assert complete['binding']['checkpoint_sha256'] == '6c8eb92f629ccdf6932407124190d651e8231058ef6f232a0a59380370d44c8d'
                assert complete['binding']['initial_variant'] == variant
                assert complete['binding']['split'] == 'train'
                metadata = OUT/f'png-metadata/V{variant}'/p.parent.parent.name/p.parent.name
                metadata.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p.parent/'complete.json', metadata/'complete.json')
                shutil.copy2(p.parent/'writes.jsonl', metadata/'writes.jsonl')
                pngs[str(p)] = dict(path=str(p), sha256=digest, bytes=p.stat().st_size, variant=variant)
            assert row['png_sha256'] == pngs[str(p)]['sha256']
    controls = {}
    for control in ['memory','mismatch','blank','text']:
        selected = [r for r in rows if r['control'] == control]
        controls[control] = dict(correct=sum(r['correct'] for r in selected), total=len(selected), parse_failure=sum(r['parse_failure'] for r in selected), truncated=sum(r['generated']['truncated'] for r in selected))
    pairs = [(keyed[pid,c,'memory','T1']['correct'],keyed[pid,c,'mismatch','T1']['correct']) for pid in ids for c in range(2)]
    family = dict(controls=controls, match_minus_mismatch=sum(a-b for a,b in pairs), repaired=sum(a and not b for a,b in pairs), regressed=sum(b and not a for a,b in pairs), both_noise_correct=sum(all(keyed[pid,c,'memory','T1']['correct'] for c in range(2)) for pid in ids), independent_preferences=730)
    summaries[f'V{variant}'] = dict(split='train', record_count=4380, families={'T1':family}, files={p.name:sha(p) for p in source.glob('readback-*.jsonl')})
    if variant == 0:
        assert summaries['V0'] == json.loads((source/'summary.json').read_text())
    assert sum(p['variant'] == variant for p in pngs.values()) == 1460
save('baseline-summary.json', summaries)
save('actual-png-hashes.json', list(pngs.values()))

# Retain the same immutable file inode while training atomically replaces resume.pt.
snapshot = OUT/'resume-observed.pt'
os.link(RUN/'train/resume.pt', snapshot)
with snapshot.open('rb') as f:
    saved = torch.load(f, map_location='cpu', weights_only=False)
step = saved['optimizer_step']
assert 23360 < step < 46720 and saved['episode_cursor'] == step*4
assert len(saved['optimizer']['state']) == 1075
assert {int(v['step']) for v in saved['optimizer']['state'].values()} == {step}
assert saved['manifest']['steps'] == 93440 and saved['manifest']['snapshot_steps'] == [46720,70080]
assert saved['manifest']['continued_from_sha256'] == '0acddb16b60f6f6e128d59041fe97db356a3022bc32da1fb8ed8b9813b5136f7'
assert len(saved['rng_state']['torch_cuda']) == 1
records = []
for line in (RUN/'train/optimization.jsonl').read_text().splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row['step'] <= step:
        records.append(row)
assert [r['step'] for r in records] == list(range(1,step+1))
training_rows = load_training_records('train')
orders = {}
for row in records[23360:]:
    for micro, draw_record in enumerate(row['draws']):
        draw = (row['step']-1)*4+micro
        cycle, offset = divmod(draw,730)
        if cycle not in orders:
            orders[cycle] = torch.randperm(730,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
        assert draw_record['pair_id'] == training_rows[orders[cycle][offset]]['base_pair_id']
        assert draw_record['position'] == 0 and draw_record['initial_variant'] == cycle%2
        assert draw_record['sigma'] == float(torch.rand((),generator=torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))))
(OUT/'optimization-committed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
save('checkpoint-evidence.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), frozen_commit='656fdf029c4a7c05e53bccb75483a03cf62d5f54', committed_step=step, episode_cursor=step*4, optimizer_states=1075, all_optimizer_steps_match=True, full_log_contiguous_unique=True, all_new_draws_and_sigma_match_original_schedule=True, rng_keys=list(saved['rng_state']), resume_sha256=sha(snapshot), resume_bytes=snapshot.stat().st_size, preserved_remote_snapshot=str(snapshot), final_step=93440, snapshot_steps=[46720,70080]))
for name in ['train/manifest.json','train/continuation.json','controller.json','latest-runtime.json','processes/step-023360-train-V0-rollout.json','processes/step-023360-train-V0-readback.json','logs/step-023360-train-V0-readback.log','logs/step-023360-train-V0-rollout.log']:
    target = OUT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN/name, target)
shutil.copy2(__file__, OUT/'archive-script.py')
archive = OUT/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p not in [snapshot, archive]:
            tar.add(p, arcname=str(p.relative_to(OUT)), recursive=False)
receipt = dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size, png_hash_count=len(pngs), record_count=8760, committed_step=step)
save('archive-receipt.json', receipt)
print(json.dumps(receipt, indent=2))
