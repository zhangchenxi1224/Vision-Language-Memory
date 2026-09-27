"""Resume only frozen round2 evaluation after notebook auto-recycle."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
from collections import defaultdict

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
OLD = TASK/'round2/pilot64'
OUT = TASK/'round2/recovery-20260927-2135'
CODE = PROJECT/'repos/prefeval-multitarget-round2'
COMMIT = '224cc77d790cf3967b5a56ce2e77c364959435a2'
PYTHON = PROJECT/'envs/vlm-r3-ngc2502/bin/python'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
BASE = MODELS/'DreamLite-base-a9a0f15-20260907'
READER = MODELS/'Qwen3-VL-4B-Instruct'
OFFICIAL = PROJECT/'Vision-Language-Memory/third_party/DreamLite'
VARIANTS = PROJECT/'runs/prefeval-b-mcq-20260925/variants-train.json'
WRITER = CODE/'scripts/experiments/prefeval_k1_writer.py'
EVAL = CODE/'scripts/experiments/prefeval_k1_evaluate.py'
EXPECTED = {
    'S1': 'd9b796e99a73649026f8d9786e246a6af1a06e56182eab6ee4b22654b5f12f9c',
    'C8': 'efd7b4a22a40508e9ae5a9d7ca72305f5a1c8b3130b3e22c5e2b72aa7bb36978',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def immutable(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        save(path, value)


def key(row):
    return tuple(row[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [TASK/'pipeline.lock', OLD/'controller.lock', OUT/'controller.lock']:
        stream = path.open('a')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True).strip()
    state = dict(host=os.uname().nodename, pid=os.getpid(), code=str(CODE), commit=COMMIT,
                 recovery_script_sha256=sha(Path(__file__)), original_output=str(OLD))

    def status(stage, **extra):
        value = dict(state, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra)
        save(OUT/'controller.json', value)
        print(json.dumps(value), flush=True)

    def jobs(label, commands):
        status(label)
        for gpu, _ in commands:
            assert not subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), f'GPU {gpu} busy'
        children = []
        for gpu, command in commands:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            stream = (OUT/f'{label}-gpu{gpu}.log').open('a')
            command = list(map(str, command))
            child = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=tuple(x.fileno() for x in locks))
            children.append((child, stream))
            save(OUT/f'{label}-gpu{gpu}-process.json', dict(state, gpu=gpu, pid=child.pid, command=command))
        codes = []
        for child, stream in children:
            codes.append(child.wait())
            stream.close()
        if any(codes):
            status('failed', failed_stage=label, exit_codes=codes)
            raise RuntimeError(f'{label}: {codes}')

    ids = json.loads((OLD/'ids.json').read_text())['ids']
    assert len(ids) == 64 and len(set(ids)) == 64
    banks = {arm: json.loads((OLD/f'{arm}-bank.json').read_text()) for arm in EXPECTED}
    for arm, expected in EXPECTED.items():
        receipt = json.loads((OLD/arm/'train/complete.json').read_text())
        assert receipt['steps'] == 2048 and receipt['checkpoint_sha256'] == expected
        assert sha(OLD/arm/'train/checkpoint-final.pt') == expected
        assert set(banks[arm]['targets']) == set(ids)
    immutable(OUT/'protocol.json', dict(original_protocol=json.loads((OLD/'protocol.json').read_text()),
        checkpoint_sha256=EXPECTED, original_protocol_sha256=sha(OLD/'protocol.json'),
        bank_sha256={arm: sha(OLD/f'{arm}-bank.json') for arm in banks},
        recovery_script_sha256=state['recovery_script_sha256'],
        recovery='evaluation only; same 64 IDs, paired mt8-eval seeds, evaluator, parser, all four shards'))
    if (OUT/'complete.json').exists():
        assert json.loads((OUT/'complete.json').read_text())['results_sha256'] == sha(OUT/'results.json')
        return
    # Reuse completed S1/V0 PNGs read-only and preserve every old MCQ row byte-for-byte.
    old_images = OLD/'S1/eval-V0'
    manifest = json.loads((old_images/'manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == EXPECTED['S1']
    assert manifest['initial_variant'] == 0 and manifest['noise_domain'] == 'mt8-eval'
    reused = []
    for pid in ids:
        for chain in range(8):
            folder = old_images/pid.replace(':', '_')/f'seed-{chain}'
            receipt = json.loads((folder/'complete.json').read_text())
            assert receipt['binding'] == manifest
            for name, digest in receipt['png_hashes'].items():
                assert sha(folder/name) == digest
            reused.append({'pair_id': pid, 'chain': chain, 'png_hashes': receipt['png_hashes']})
    for shard in range(4):
        source = OLD/f'S1/read-V0/readback-{shard}.jsonl'
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len({key(row) for row in rows}) == len(rows)
        assert all(row['pair_id'] in ids[shard::4] for row in rows)
        dest = OUT/f'S1/read-V0/readback-{shard}.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes().startswith(raw)
        else:
            shutil.copyfile(source, dest)
    immutable(OUT/'reused-inputs.json', {'images': reused,
        'old_readbacks': {str(p): sha(p) for p in sorted((OLD/'S1/read-V0').glob('readback-*.jsonl'))}})
    common = ['--arm', 'B', '--base', BASE, '--official-source', OFFICIAL, '--split', 'train',
              '--ids-file', OLD/'ids.json', '--initial-variants', VARIANTS]
    for arm in banks:
        for variant in range(2):
            images = old_images if (arm, variant) == ('S1', 0) else OUT/arm/f'eval-V{variant}'
            if (arm, variant) != ('S1', 0):
                jobs(f'eval-generate-{arm}-V{variant}', [(gpu, [PYTHON, WRITER, 'rollout', *common,
                    '--checkpoint', OLD/arm/'train/checkpoint-final.pt', '--initial-variant', variant,
                    '--inter-turns', 0, '--noise-chains', 8, '--noise-domain', 'mt8-eval',
                    '--shard-index', gpu, '--shard-count', 4, '--output', images]) for gpu in range(4)])
                manifests = [json.loads((images/f'manifest-shard-{gpu}.json').read_text()) for gpu in range(4)]
                assert all(m == manifests[0] for m in manifests)
                assert all((images/pid.replace(':', '_')/f'seed-{k}/complete.json').exists()
                           for pid in ids for k in range(8))
                immutable(images/'manifest.json', manifests[0])
            jobs(f'eval-read-{arm}-V{variant}', [(gpu, [PYTHON, EVAL, '--kind', 'student',
                '--split', 'train', '--ids-file', OLD/'ids.json', '--reader', READER,
                '--initial-variants', VARIANTS, '--initial-variant', variant, '--images', images,
                '--output', OUT/arm/f'read-V{variant}', '--families', 'T1,T2,T3', '--tasks', 'mcq',
                '--prefixes', 0, '--noise-chains', 8, '--controls', 'memory,mismatch,blank,text',
                '--shard', gpu, '--shards', 4]) for gpu in range(4)])
    results = {}
    for arm, bank in banks.items():
        counts = defaultdict(lambda: [0, 0])
        per_image, per_preference = defaultdict(list), defaultdict(list)
        hashes, parse_failures = {}, 0
        for variant in range(2):
            seen = set()
            for shard in range(4):
                path = OUT/arm/f'read-V{variant}/readback-{shard}.jsonl'
                hashes[str(path)] = sha(path)
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    assert key(row) not in seen and row['pair_id'] in ids[shard::4]
                    seen.add(key(row))
                    count = counts[f"V{variant}/{row['control']}/{row['family']}"]
                    count[0] += int(row['correct']); count[1] += 1
                    parse_failures += int(row['parse_failure'])
                    if row['control'] == 'memory':
                        per_image[row['pair_id'], variant, row['chain']].append(row['correct'])
                        per_preference[row['pair_id']].append(int(row['correct']))
            assert len(seen) == len(ids)*54
        assert len(per_image) == 1024 and all(len(v) == 3 for v in per_image.values())
        for variant in range(2):
            for control in ['memory', 'mismatch', 'blank', 'text']:
                for family in ['T1', 'T2', 'T3']:
                    assert counts[f'V{variant}/{control}/{family}'][1] == 64*(8 if control in ['memory', 'mismatch'] else 1)
        results[arm] = {'correct_total': dict(counts), 'images': len(per_image),
            'all_three_forms_correct': sum(all(v) for v in per_image.values()),
            'per_preference': {k: sum(v)/len(v) for k, v in per_preference.items()},
            'qualified_target_counts': {k: len(v) for k, v in bank['targets'].items()},
            'candidate_coverage': bank['coverage'], 'parse_failures': parse_failures,
            'readback_sha256': hashes, 'checkpoint_sha256': EXPECTED[arm]}
    save(OUT/'results.json', results)
    status('round2_complete_iteration_review_required', results=str(OUT/'results.json'))
    save(OUT/'complete.json', {'status': 'round2_complete', 'results_sha256': sha(OUT/'results.json')})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save(OUT/f'failure-{os.getpid()}.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
            'type': type(error).__name__, 'message': str(error)})
        raise
