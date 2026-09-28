"""Archive and verify the fixed training endpoint after its completion sentinel exists."""
import collections
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import torch

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN = ROOT/'runs/prefeval-b730-exposure512-20260927'
CODE = RUN/'code'
OUT = RUN/'evidence/training-final'
done = json.loads((RUN/'train/complete.json').read_text())
assert done['steps'] == 93440
assert not OUT.exists(), 'Inspect existing evidence before retrying'
assert not (RUN/'failure.json').exists()
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
assert commit == '656fdf029c4a7c05e53bccb75483a03cf62d5f54'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=CODE,text=True).strip()
train_receipt = json.loads((RUN/'processes/train.json').read_text())
adoption = json.loads((RUN/'train-adoption.json').read_text())
assert train_receipt['host'] == socket.gethostname()
assert train_receipt['status'] == 'complete' and train_receipt['exit_code'] is None
assert train_receipt['completion_evidence'] == 'adopted process exited and frozen writer published train/complete.json steps93440'
train_proc = Path('/proc')/str(train_receipt['pid'])
proc_state = None
if (train_proc/'stat').exists():
    proc_state = (train_proc/'stat').read_text().rsplit(')',1)[1].split()[0]
    assert proc_state == 'Z', 'Training must have exited before endpoint archival'
sys.path[:0] = [str(CODE),str(CODE/'src')]
from scripts.experiments.prefeval_k1_data import load_training_records
from vision_memory.training.latent_bank_unet import stable_seed

def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

OUT.mkdir(parents=True)
os.link(RUN/'train/resume.pt',OUT/'resume-final.pt')
os.link(RUN/'train/checkpoint-final.pt',OUT/'checkpoint-final.pt')
saved = torch.load(OUT/'resume-final.pt',map_location='cpu',weights_only=False)
inference = torch.load(OUT/'checkpoint-final.pt',map_location='cpu',weights_only=False)
assert saved['optimizer_step'] == inference['optimizer_step'] == 93440
assert saved['episode_cursor'] == 373760
assert len(saved['optimizer']['state']) == 1075
assert {int(value['step']) for value in saved['optimizer']['state'].values()} == {93440}
manifest = json.loads((RUN/'train/manifest.json').read_text())
assert saved['manifest'] == inference['manifest'] == manifest
assert manifest['steps'] == 93440 and manifest['snapshot_steps'] == [46720,70080]
assert manifest['effective_batch'] == 4 and manifest['training_variant_indices'] == [0,1]
assert manifest['continued_from_sha256'] == '0acddb16b60f6f6e128d59041fe97db356a3022bc32da1fb8ed8b9813b5136f7'
assert set(saved['rng_state']) == {'python','numpy','torch_cpu','torch_cuda'}
assert len(saved['rng_state']['torch_cuda']) == 1
assert len(saved['trainable_state']) == len(inference['trainable_state']) == 1075
assert set(saved['trainable_state']) == set(inference['trainable_state'])
assert all(torch.equal(value,inference['trainable_state'][name]) for name,value in saved['trainable_state'].items())
endpoint_hash = sha(OUT/'checkpoint-final.pt')
assert endpoint_hash == done['checkpoint_sha256']
resume_hash = sha(OUT/'resume-final.pt')
raw = (RUN/'train/optimization.jsonl').read_bytes()
records = [json.loads(line) for line in raw.splitlines()]
assert [r['step'] for r in records] == list(range(1,93441))
training_rows = load_training_records('train')
assert len(training_rows) == 730
orders, counts = {}, collections.Counter()
losses = []
for row in records:
    assert len(row['draws']) == 4
    assert math.isfinite(row['grad_norm'])
    losses.append(sum(d['mse'] for d in row['draws'])/4)
    for micro,d in enumerate(row['draws']):
        assert math.isfinite(d['mse']) and d['mse'] >= 0
        counts[d['pair_id'],d['initial_variant']] += 1
        if row['step'] <= 23360:
            continue
        draw = (row['step']-1)*4+micro
        cycle, offset = divmod(draw,730)
        if cycle not in orders:
            orders[cycle] = torch.randperm(730,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
        assert d['pair_id'] == training_rows[orders[cycle][offset]]['base_pair_id']
        assert d['initial_variant'] == cycle%2 and d['position'] == 0
        assert d['sigma'] == float(torch.rand((),generator=torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))))
assert len(counts) == 1460 and set(counts.values()) == {256}
with gzip.open(OUT/'optimization-committed.jsonl.gz','wb') as file:
    file.write(raw)
source = dict(captured_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source=str(RUN/'train/optimization.jsonl'),source_resume=str(RUN/'train/resume.pt'),checkpoint_sha256=resume_hash,committed_step=93440,observed_log_step=93440,uncommitted_tail_omitted=0,rows=93440,first_step=1,contiguous_unique=True,effective_batch=4,metric='Unweighted mean of the four microbatch official flow-matching velocity MSEs',original_endpoint=23360,fixed_endpoints=[46720,70080,93440],frozen_commit=commit,optimizer_states=1075,checkpoint_state_steps_verified=True,raw_jsonl_sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=sha(OUT/'optimization-committed.jsonl.gz'),gzip_bytes=(OUT/'optimization-committed.jsonl.gz').stat().st_size)
proof = dict(time_utc=source['captured_at_utc'],frozen_commit=commit,final_step=93440,episode_cursor=373760,optimizer_states=1075,all_optimizer_states_same_step=True,rng_keys=list(saved['rng_state']),full_log_contiguous_unique=True,all_new_draws_and_sigma_match_original_schedule=True,all_730_exposures_each_variant=256,exposures_per_preference=512,final_inference_equals_full_resume=True,checkpoint_final_sha256=endpoint_hash,resume_final_sha256=resume_hash,resume_bytes=(OUT/'resume-final.pt').stat().st_size,mean_mse_trailing1000={str(step):sum(losses[step-1000:step])/1000 for step in [1000,23360,46720,70080,93440]},remote_resume=str(OUT/'resume-final.pt'))
proof.update(training_os_exit_code=None,training_process_exited=True,training_pid=train_receipt['pid'],training_process_state=proc_state,training_host=socket.gethostname(),training_completion_evidence=train_receipt['completion_evidence'])
for name,value in [('source.json',source),('training-final-proof.json',proof)]:
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
for name in ['train/manifest.json','train/continuation.json','train/complete.json','controller.json','reader-controller.json','runtime-primary.json','runtime-reader.json','train-adoption.json','logs/train.log','processes/train.json']:
    dest = OUT/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(RUN/name,dest)
shutil.copy2(__file__,OUT/'archive-script.py')
archive = OUT/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix != '.pt' and path != archive:
            tar.add(path,arcname=str(path.relative_to(OUT)),recursive=False)
receipt = dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
(OUT/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(proof=proof,archive=receipt),indent=2))
