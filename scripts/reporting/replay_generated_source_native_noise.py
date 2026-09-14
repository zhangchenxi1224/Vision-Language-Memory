import hashlib
import json
from pathlib import Path
import sys
import time
import torch

root = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/dreamlite-official-alignment')
pool = root / '90b41a2-generated-source-pool'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(pool / 'plan.json') == '7603bd62e10241e0542e6b5b30168a5c2ed6d22159e7113d7754c8860b95c2f3'
assert sha(pool / 'manifest.json') == '9431b9a11eac1c4b956d8178447223f92c59058ef8a851d75b575abb41b60858'
sys.path.insert(0, '/inspire/ssd/project/exploration-topic/czxs26210936/repos/dreamlite-generated-source-pool-20260914/src')
from vision_memory.repro import canonical_tensor_sha256
plan = json.loads((pool / 'plan.json').read_bytes())
records = {r['job']: r for r in json.loads((pool / 'manifest.json').read_bytes())['records']}
rows = []
for job in plan['jobs']:
    record = records[job['id']]
    tensor_path = pool / record['tensor']
    assert sha(tensor_path) == record['tensor_sha256']
    actual = torch.load(tensor_path, map_location='cpu', weights_only=True)['noise']
    expected = torch.randn((1,4,128,128), generator=torch.Generator().manual_seed(job['seed']), dtype=torch.float32)
    rows.append({'job': job['id'], 'seed': job['seed'], 'tensor_sha256': record['tensor_sha256'],
        'replayed_noise_sha256': canonical_tensor_sha256(expected), 'recorded_noise_sha256': canonical_tensor_sha256(actual),
        'bitwise_equal': torch.equal(actual, expected)})
assert len(rows) == 24 and all(row['bitwise_equal'] for row in rows)
result = {'schema': 'generated-source-native-noise-replay/v1', 'commit': '90b41a2f1e7e4e0a8d41e709cedb69c4b32e3231',
    'plan_sha256': sha(pool / 'plan.json'), 'manifest_sha256': sha(pool / 'manifest.json'),
    'torch_version': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
    'platform': sys.platform, 'time_unix': time.time(), 'replay_script_sha256': sha(Path(__file__)),
    'rows': rows, 'all_bitwise_equal': True}
path = root / '90b41a2-generated-source-pool-native-noise-replay.json'
with path.open('xb') as stream:
    stream.write((json.dumps(result, indent=2)+'\n').encode())
print(json.dumps({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
    'torch_version': result['torch_version'], 'cpu_capability': result['cpu_capability'], 'all24_bitwise_equal': True}))
