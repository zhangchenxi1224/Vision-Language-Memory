"""Generate all preregistered training-only source PNGs on four independent GPUs."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.generated_source_pool_protocol import plan, digest, PACKAGE_SHA, CHECKPOINT_SHA


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def clean(commit):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the exact clean committed source')


def worker(a, registered):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.dreamlite.writer_package import load_writer_package, package_inference_condition
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    from vision_memory.reader.open_answer import generate_short_answer
    from scripts.inspire.model_snapshot_manifest import verify_snapshot_manifest
    from scripts.train.r11_new_frozen_dreamlite_oracle import _load_reader
    configure_strict_cuda_determinism(20260913)
    if digest(a.package / 'manifest.json') != PACKAGE_SHA:
        raise ValueError('Changed fixed4f package')
    pipe, manifest = load_writer_package(a.package, base_model=a.base_model,
        official_source=a.official_source, device=f'cuda:{a.lane}')
    if (manifest['parent_checkpoint_sha256'] != CHECKPOINT_SHA
            or manifest['parent_commit'] != registered['parent_commit']
            or manifest['guidance_scale'] != 1.0 or package_inference_condition(manifest) != 'native'):
        raise ValueError('Changed parent or inference protocol')
    expected = manifest['reader_snapshot']
    actual = verify_snapshot_manifest(manifest_path=a.reader_model / '.snapshot_manifest.json', model_dir=a.reader_model,
        expected_repo_id=expected['repo_id'], expected_revision=expected['revision'])
    if any(actual[key] != expected[key] for key in ('manifest_sha256', 'snapshot_payload_sha256')):
        raise ValueError('Changed frozen Reader snapshot')
    device = torch.device(f'cuda:{a.lane}')
    processor, reader = _load_reader(SimpleNamespace(reader=a.reader_model), device, torch.bfloat16)
    directory = a.output / f'lane-{a.lane}'
    directory.mkdir(exist_ok=False)
    records = []
    with torch.no_grad():
        for job in (j for j in registered['jobs'] if j['lane'] == a.lane):
            if time.time() >= a.deadline_unix:
                raise TimeoutError('Source generation deadline')
            # New image-only memory for every job; neither expected label nor query enters Writer.
            memory = OfficialRGBMemory(pipe, guidance_scale=1.0)
            output = memory.write(job['event'], seed=job['seed'])
            png = directory / (job['id'] + '.png')
            output.image.save(png)
            from PIL import Image
            with Image.open(png) as image:
                image = image.copy()
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
            if not torch.equal(pixels, output.pixels):
                raise ValueError('Actual persisted PNG differs from generated quantized pixels')
            source = pipe.prepare_image_latents(pipe.image_processor.preprocess(image), dtype=torch.float32, device=device).cpu()
            payload = directory / (job['id'] + '.pt')
            torch.save({'source_latent': source, 'generation_initial_source': output.source_latent,
                        'noise': output.noise, 'generated_latent': output.latent, 'trajectory': output.trajectory}, payload)
            reads = []
            for prompt_id, query in job['queries'].items():
                result = generate_short_answer(model=reader, processor=processor, image=pixels.to(device),
                    query=query, device=device, max_new_tokens=32, do_sample=False)
                row = {'job': job['id'], 'prompt_id': prompt_id, 'query': query, **result,
                       'correct_eos': result['generated_token_ids'] == job['expected_token_ids']}
                reads.append(row)
                with (directory / 'reads.jsonl').open('ab') as stream:
                    stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
            records.append({'job': job['id'], 'state': job['state'], 'seed': job['seed'],
                'png': png.name, 'png_sha256': digest(png), 'tensor': payload.name, 'tensor_sha256': digest(payload),
                'source_latent_sha256': canonical_tensor_sha256(source),
                'correct_reads': sum(row['correct_eos'] for row in reads), 'reads': len(reads)})
            print(json.dumps(records[-1]), flush=True)
    if digest(a.package / 'manifest.json') != PACKAGE_SHA:
        raise ValueError('Package changed during generation')
    clean(a.expected_commit)
    write(directory / 'complete.json', {'commit': a.expected_commit, 'plan_sha256': digest(a.plan),
        'package_sha256': PACKAGE_SHA, 'reads_sha256': digest(directory / 'reads.jsonl'), 'records': records})


def collect(a, registered, *, noise_reference=None):
    """Recount everything; optionally bind noise to an independently sealed native replay.

    Training/generation use the default exact local RNG check. A reporting caller
    on a different CPU/PyTorch build must verify the reference file's independent
    SHA before supplying it; this path compares exact hashes, never tolerances.
    """
    import torch
    from PIL import Image
    from vision_memory.repro import canonical_tensor_sha256
    reference_rows = None
    if noise_reference is not None:
        if (noise_reference.get('schema') != 'generated-source-native-noise-replay/v1'
                or noise_reference.get('commit') != a.expected_commit
                or noise_reference.get('plan_sha256') != digest(a.plan)
                or noise_reference.get('manifest_sha256') != digest(a.output / 'manifest.json')
                or noise_reference.get('all_bitwise_equal') is not True):
            raise ValueError('Wrong native noise replay binding')
        reference_rows = {r['job']: r for r in noise_reference['rows']}
        if len(noise_reference['rows']) != 24 or set(reference_rows) != {j['id'] for j in registered['jobs']}:
            raise ValueError('Incomplete independent native noise replay')
    records = []
    for lane in range(4):
        directory = a.output / f'lane-{lane}'
        complete = json.loads((directory / 'complete.json').read_bytes())
        jobs = [j for j in registered['jobs'] if j['lane'] == lane]
        if (complete['commit'] != a.expected_commit or complete['plan_sha256'] != digest(a.plan)
                or complete['package_sha256'] != PACKAGE_SHA or len(complete['records']) != len(jobs)
                or complete['reads_sha256'] != digest(directory / 'reads.jsonl')):
            raise ValueError('Missing or changed lane binding')
        rows = [json.loads(line) for line in (directory / 'reads.jsonl').read_bytes().splitlines()]
        if len(rows) != len(jobs) * 5:
            raise ValueError('Incomplete raw read coverage')
        for i, (job, record) in enumerate(zip(jobs, complete['records'], strict=True)):
            if (record['job'], record['state'], record['seed']) != (job['id'], job['state'], job['seed']):
                raise ValueError('Wrong generated case')
            if record['png'] != job['id'] + '.png' or record['tensor'] != job['id'] + '.pt':
                raise ValueError('Unexpected artifact path')
            for key in ('png', 'tensor'):
                if digest(directory / record[key]) != record[key + '_sha256']:
                    raise ValueError('Changed generated artifact')
            with Image.open(directory / record['png']) as image:
                if image.mode != 'RGB' or image.size != (1024, 1024):
                    raise ValueError('Wrong generated image shape')
            payload = torch.load(directory / record['tensor'], map_location='cpu', weights_only=True)
            tensors = [payload[k] for k in ('source_latent', 'generation_initial_source', 'noise', 'generated_latent')]
            tensors += list(payload['trajectory'])
            if (len(payload['trajectory']) != 29 or any(t.dtype != torch.float32 or t.shape != (1, 4, 128, 128)
                    or not torch.isfinite(t).all() for t in tensors)
                    or not torch.equal(payload['trajectory'][0], payload['noise'])
                    or not torch.equal(payload['trajectory'][-1], payload['generated_latent'])
                    or canonical_tensor_sha256(payload['source_latent']) != record['source_latent_sha256']):
                raise ValueError('Invalid source/generation tensor evidence')
            if reference_rows is None:
                expected_noise = torch.randn((1, 4, 128, 128), generator=torch.Generator().manual_seed(job['seed']), dtype=torch.float32)
                if not torch.equal(payload['noise'], expected_noise):
                    raise ValueError('Wrong actual generation noise')
            else:
                reference = reference_rows[job['id']]
                if (reference['seed'] != job['seed'] or reference['tensor_sha256'] != record['tensor_sha256']
                        or reference['bitwise_equal'] is not True
                        or reference['replayed_noise_sha256'] != reference['recorded_noise_sha256']
                        or canonical_tensor_sha256(payload['noise']) != reference['replayed_noise_sha256']):
                    raise ValueError('Recorded noise differs from the independently replayed native seed')
            actual_reads = rows[5*i:5*i+5]
            correct = 0
            for row, (prompt_id, query) in zip(actual_reads, job['queries'].items(), strict=True):
                if (row['job'], row['prompt_id'], row['query']) != (job['id'], prompt_id, query):
                    raise ValueError('Wrong read assignment')
                passed = row['generated_token_ids'] == job['expected_token_ids']
                if row['correct_eos'] != passed:
                    raise ValueError('Changed raw token score')
                correct += int(passed)
            if record['correct_reads'] != correct or record['reads'] != 5:
                raise ValueError('Changed qualified count')
            records.append({**record, 'png': f'lane-{lane}/' + record['png'], 'tensor': f'lane-{lane}/' + record['tensor']})
    return {'schema': registered['schema'], 'commit': a.expected_commit, 'plan_sha256': digest(a.plan),
        'parent_package_sha256': PACKAGE_SHA, 'records': records, 'generated_images': len(records),
        'raw_reads': 120, 'correct_reads': sum(r['correct_reads'] for r in records),
        'qualified_for_training': all(r['correct_reads'] == 5 for r in records), 'scope': registered['scope']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output', 'package', 'base-model', 'official-source', 'reader-model'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--lane', type=int, choices=range(4))
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    if not math.isfinite(a.deadline_unix):
        raise ValueError('Require a finite deadline')
    clean(a.expected_commit)
    registered = json.loads(a.plan.read_bytes())
    if registered != plan():
        raise ValueError('Changed fixed source generation plan')
    if a.collect_only:
        result = collect(a, registered)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
        return
    if a.lane is not None:
        return worker(a, registered)
    if time.time() + 15 * 60 > a.deadline_unix:
        raise ValueError('Reserve15 minutes for source preparation')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPUs occupied; do not evict existing work')
    if len(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines()) != 4:
        raise ValueError('Require exactly four allocated GPUs')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'plan.json').write_bytes(a.plan.read_bytes())
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    children, logs = [], []
    try:
        for lane in range(4):
            log = (a.output / f'lane-{lane}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                *sys.argv[1:], '--lane', str(lane)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, **REQUIRED_DETERMINISM_ENV}, start_new_session=True))
        write(a.output / 'status.json', {'state': 'running', 'time_unix': time.time(),
            'pids': [child.pid for child in children], 'commit': a.expected_commit, 'plan_sha256': digest(a.plan)})
        while any(child.poll() is None for child in children):
            if time.time() >= a.deadline_unix or any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('Source generation failed or reached deadline; preserve all outputs')
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError('Source generation lane failed')
        result = collect(a, registered)
        write(a.output / 'manifest.json', result)
        write(a.output / 'status.json', {'state': 'completed', 'time_unix': time.time(),
            'commit': a.expected_commit, 'manifest_sha256': digest(a.output / 'manifest.json'),
            'qualified_for_training': result['qualified_for_training'], 'correct_reads': result['correct_reads']})
    except BaseException as error:
        write(a.output / 'status.json', {'state': 'failed', 'time_unix': time.time(), 'error': repr(error)})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
