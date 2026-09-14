import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root / 'src')]
from scripts.experiments.generated_source_pool_protocol import plan, digest
from scripts.probes.generate_training_source_pool import collect
from scripts.reporting.assemble_portable_archive import assemble

reports = root / 'reports/official-alignment-results-20260913'
observed = json.loads((reports / '90b41a2-generated-source-pool-observed-artifacts.json').read_bytes())
assert observed['archive']['sha256'] == 'b8b36757b36781020f1e309f0d5a402a94e47001204d15e37801ccebe6c49403'
manifest_sha = '6c80c1003fa65c18ec0d5ead5dd9d4e87d137dca246f61c41d3f9bd29ef466a0'
parts = reports / '90b41a2-generated-source-pool-evidence.tgz.parts'
assert digest(parts / 'manifest.json') == manifest_sha
assert json.loads((parts / 'manifest.json').read_bytes())['sha256'] == observed['archive']['sha256']
archive = root / '.cache/90b41a2-source-pool-reconstructed.tgz'
assembly = assemble(parts / 'manifest.json', manifest_sha, archive)
assert assembly['bytes'] == observed['archive']['bytes']
local = root / '.cache/90b41a2-source-pool-local'
already_extracted = local.exists()
local.mkdir(exist_ok=True)
with tarfile.open(archive) as stream:
    for member in stream.getmembers():
        path = Path(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts:
            raise ValueError('Unexpected archive member')
        if already_extracted:
            assert (local / path).read_bytes() == stream.extractfile(member).read()
    if not already_extracted:
        stream.extractall(local, filter='data')
assert digest(local / 'manifest.json') == observed['manifest']['sha256']
assert digest(local / 'plan.json') == observed['plan']['sha256']
noise_path = reports / '90b41a2-generated-source-pool-native-noise-replay.json'
noise_sha = '7d919108dec63151117c28d6e1d57dbb223506ade07a47b2eca8d94d5ae34bf3'
assert digest(noise_path) == noise_sha
noise_reference = json.loads(noise_path.read_bytes())
actual = collect(SimpleNamespace(output=local, plan=local / 'plan.json',
    expected_commit='90b41a2f1e7e4e0a8d41e709cedb69c4b32e3231'), plan(), noise_reference=noise_reference)
assert actual == json.loads((local / 'manifest.json').read_bytes())
unique = {state: len({r['png_sha256'] for r in actual['records'] if r['state'] == state})
          for state in ('ambient', 'jazz', 'clear')}
import torch
local_differences = []
for job in plan()['jobs']:
    actual_noise = torch.load(local / f"lane-{job['lane']}" / (job['id'] + '.pt'), weights_only=True)['noise']
    native_windows = torch.randn(actual_noise.shape, generator=torch.Generator().manual_seed(job['seed']), dtype=torch.float32)
    delta = (actual_noise - native_windows).abs()
    local_differences.append({'job': job['id'], 'different_elements': int(torch.count_nonzero(delta)),
                             'elements': delta.numel(), 'max_absolute_difference': float(delta.max())})
result = {'archive_sha256': assembly['sha256'], 'bytes': assembly['bytes'],
    'chunks': assembly['chunks_verified'], 'parts_manifest_sha256': manifest_sha,
    'manifest_sha256': observed['manifest']['sha256'], 'plan_sha256': observed['plan']['sha256'],
    'complete_recount': actual, 'distinct_png_sha256_per_state': unique,
    'native_noise_replay_sha256': noise_sha, 'all24_native_noise_replay_bitwise_equal': True,
    'local_rng_comparison': {'torch': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
        'rows': local_differences, 'used_as_approximate_acceptance': False},
    'scope': 'All24 source PNGs, all24 actual tensor payloads/29-state trajectories and all120 raw reads recounted locally. Every archived noise tensor exactly matches the independently sealed native Linux PyTorch replay hash for its original seed. Windows CPU noise is separately compared and is not bitwise identical; no tolerance substitutes for the native replay. Official PNG-to-VAE encoding checked on GPU generation and will be checked again in training; no local GPU VAE replay claimed.'}
(reports / '90b41a2-generated-source-pool-manifest.json').write_bytes((local / 'manifest.json').read_bytes())
(reports / '90b41a2-generated-source-pool-local-verification.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode())
print(json.dumps({k: v for k,v in result.items() if k != 'complete_recount'}))
print(json.dumps({'images': actual['generated_images'], 'reads': actual['raw_reads'], 'correct': actual['correct_reads'],
                  'qualified': actual['qualified_for_training']}))
