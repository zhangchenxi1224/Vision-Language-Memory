"""Validate the downloaded fixed-final training archive and expose plot inputs."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
receipt = json.loads((root/'archive-receipt.json').read_text())
archive = (root/'raw.tar.gz').read_bytes()
assert len(archive) == receipt['bytes']
assert hashlib.sha256(archive).hexdigest() == receipt['sha256']
with tarfile.open(root/'raw.tar.gz') as tar:
    content = {member.name:tar.extractfile(member).read() for member in tar if member.isfile()}
source = json.loads(content['source.json'])
proof = json.loads(content['training-final-proof.json'])
manifest = json.loads(content['train/manifest.json'])
done = json.loads(content['train/complete.json'])
assert source['committed_step'] == source['observed_log_step'] == proof['final_step'] == done['steps'] == 93440
assert source['uncommitted_tail_omitted'] == 0 and proof['episode_cursor'] == 373760
assert proof['optimizer_states'] == 1075 and proof['all_optimizer_states_same_step']
assert proof['final_inference_equals_full_resume'] and proof['all_new_draws_and_sigma_match_original_schedule']
assert proof['training_os_exit_code'] is None and proof['training_process_exited']
train_receipt = json.loads(content['processes/train.json'])
assert train_receipt['exit_code'] is None and train_receipt['status'] == 'complete'
assert proof['training_pid'] == train_receipt['pid'] and proof['training_host'] == train_receipt['host']
assert set(proof['rng_keys']) == {'python','numpy','torch_cpu','torch_cuda'}
assert proof['checkpoint_final_sha256'] == done['checkpoint_sha256']
assert source['checkpoint_sha256'] == proof['resume_final_sha256']
assert manifest['steps'] == 93440 and manifest['snapshot_steps'] == [46720,70080]
assert manifest['effective_batch'] == 4 and manifest['training_variant_indices'] == [0,1]
assert len(manifest['targets']) == 730
compressed = content['optimization-committed.jsonl.gz']
assert hashlib.sha256(compressed).hexdigest() == source['gzip_sha256']
raw = gzip.decompress(compressed)
assert hashlib.sha256(raw).hexdigest() == source['raw_jsonl_sha256']
records = [json.loads(line) for line in raw.splitlines()]
assert [r['step'] for r in records] == list(range(1,93441))
counts = collections.Counter()
losses = []
for row in records:
    assert len(row['draws']) == 4
    assert math.isfinite(row['grad_norm'])
    for draw in row['draws']:
        assert math.isfinite(draw['mse']) and draw['mse'] >= 0
        assert draw['position'] == 0 and draw['initial_variant'] in [0,1]
        counts[draw['pair_id'],draw['initial_variant']] += 1
    losses.append(sum(d['mse'] for d in row['draws'])/4)
assert len(counts) == 1460 and set(counts.values()) == {256}
for end in [1000,23360,46720,70080,93440]:
    assert math.isclose(sum(losses[end-1000:end])/1000,proof['mean_mse_trailing1000'][str(end)],rel_tol=1e-12)
for name in ['source.json','training-final-proof.json','optimization-committed.jsonl.gz','archive-script.py','train/manifest.json','train/complete.json','train/continuation.json','processes/train.json','train-adoption.json','runtime-primary.json','runtime-reader.json']:
    target = root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content[name])
result = dict(archive_sha256=receipt['sha256'],fixed_final_step=93440,contiguous_unique_updates=93440,total_draws=373760,preferences=730,exposures_each_variant=256,total_exposures_per_preference=512,loss_windows_recomputed=True,final_sentinel_matches_checkpoint_proof=True)
(root/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
