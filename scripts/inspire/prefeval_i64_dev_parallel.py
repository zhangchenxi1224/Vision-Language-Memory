"""Schedule the unchanged frozen I64 rollout on three disjoint dev partitions."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
FROZEN = PROJECT / 'repos/prefeval-b-deploy-dc8731c'
COMMIT = 'dc8731c5ec8c6b20fbb9c708730f5f613819234a'
RUN = PROJECT / 'runs/prefeval-b-mcq-20260925/identity64'
BANK = RUN / 'dev-parallel'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def records():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FROZEN, text=True).strip()
    assert actual == COMMIT
    rows = runpy.run_path(str(FROZEN / 'scripts/experiments/prefeval_k1_data.py'))['load_records']('dev')
    assert len(rows) == len({row['base_pair_id'] for row in rows}) == 90
    return rows


def partition(rows, shard):
    assert shard in range(3) and len(rows) == 90
    return rows[shard::3]


def run_shard(shard):
    with locked(BANK / f'shard-{shard}.lock'):
        assert (BANK / 'gate.json').exists(), 'Original parent must be held before parallel dev starts'
        assert not (BANK / 'merged.json').exists()
        rows = partition(records(), shard)
        output = BANK / f'shard-{shard}'
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        writer = runpy.run_path(str(FROZEN / 'scripts/experiments/prefeval_k1_writer.py'))
        writer['configure_strict_cuda_determinism'](0)
        args = argparse.Namespace(arm='B', split='dev', device='cuda:0',
            base=MODELS / 'DreamLite-base-a9a0f15-20260907',
            official_source=PROJECT / 'Vision-Language-Memory/third_party/DreamLite',
            checkpoint=RUN / 'train/checkpoint-final.pt', output=output,
            inter_turns=10, noise_chains=2, probe_initial_sources=None,
            initial_variants=None, initial_variant=0, history_file=None)
        save(BANK / f'partition-{shard}.json', {'frozen_commit': COMMIT,
            'pair_ids': [row['base_pair_id'] for row in rows], 'shard': shard, 'shards': 3,
            'seed_policy': 'unchanged frozen stable_seed(20260924, rollout:pair_id:chain, position)'})
        pipe = writer['load_pipe'](args)
        writer['rollout'](args, pipe, rows)
        save(BANK / f'complete-{shard}.json', {'pairs': 30, 'chains': 60, 'shard': shard})


def validate_shards(bank, rows, checkpoint_sha):
    binding = {'checkpoint_sha256': checkpoint_sha, 'split': 'dev', 'steps': 28, 'cfg': 1,
        'noise_chains': 2, 'inter_turns': 10,
        'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write'}
    paths = {}
    for shard in range(3):
        assert json.loads((bank / f'complete-{shard}.json').read_text()) == {
            'pairs': 30, 'chains': 60, 'shard': shard}
        root = bank / f'shard-{shard}'
        assert json.loads((root / 'manifest.json').read_text()) == binding
        expected = {row['base_pair_id'].replace(':', '_') for row in partition(rows, shard)}
        assert {p.name for p in root.iterdir() if p.is_dir()} == expected
        for name in sorted(expected):
            assert name not in paths
            paths[name] = root / name
            assert {p.name for p in paths[name].iterdir() if p.is_dir()} == {'seed-0', 'seed-1'}
            for chain in range(2):
                parent = paths[name] / f'seed-{chain}'
                done = json.loads((parent / 'complete.json').read_text())
                assert done['binding'] == binding
                assert set(done['png_hashes']) == {f'prefix-{i:02d}.png' for i in range(11)}
                assert all(sha(parent / name) == digest for name, digest in done['png_hashes'].items())
                writes = [json.loads(line) for line in (parent / 'writes.jsonl').read_text().splitlines()]
                # Recovery may retain an interrupted earlier trace; bind the final complete chain.
                assert [w['position'] for w in writes[-11:]] == list(range(11))
                for i, write in enumerate(writes[-11:]):
                    assert write['output_png_sha256'] == done['png_hashes'][f'prefix-{i:02d}.png']
                    assert write['source_png_sha256'] == (done['png_hashes'][f'prefix-{i-1:02d}.png'] if i else None)
    assert len(paths) == 90
    return binding, paths


def publish():
    with locked(BANK / 'publish.lock'):
        if (BANK / 'merged.json').exists():
            return
        binding, paths = validate_shards(BANK, records(), sha(RUN / 'train/checkpoint-final.pt'))
        target = RUN / 'dev'
        assert not target.exists(), 'Never overwrite a dev output produced by another worker'
        staging = BANK / 'publish-staging'
        staging.mkdir(exist_ok=True)
        save(staging / 'manifest.json', binding)
        for name, source in paths.items():
            link = staging / name
            if link.is_symlink():
                assert link.resolve() == source.resolve()
            else:
                link.symlink_to(source.resolve(), target_is_directory=True)
        staging.rename(target)
        save(BANK / 'merged.json', {'pairs': 90, 'chains': 180, 'pngs': 1980,
            'checkpoint_sha256': binding['checkpoint_sha256'], 'target': str(target),
            'note': 'Links organize fully generated immutable chains; no PNG bypasses a Writer update'})


def gate(parent, child):
    with locked(BANK / 'gate.lock'):
        assert str(Path(f'/proc/{parent}/cwd').resolve()) == str(FROZEN)
        assert b'run_prefeval_k1_identity64.sh' in Path(f'/proc/{parent}/cmdline').read_bytes()
        command = Path(f'/proc/{child}/cmdline').read_bytes().split(bytes([0]))
        assert b'prefeval_k1_writer.py' in command[1] and b'pilot' in command
        assert str(child) in Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        assert not (RUN / 'dev').exists()
        os.kill(parent, signal.SIGSTOP)  # Only the scheduling shell; its active pilot child keeps running.
        save(BANK / 'gate.json', {'parent_pid': parent, 'active_pilot_pid': child,
            'host': os.uname().nodename, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'state': 'hold next dev stage; active pilot inference continues'})
        while not all((BANK / f'complete-{i}.json').exists() for i in range(3)):
            time.sleep(15)
        publish()
        assert b'run_prefeval_k1_identity64.sh' in Path(f'/proc/{parent}/cmdline').read_bytes()
        os.kill(parent, signal.SIGCONT)
        save(BANK / 'gate-released.json', {'parent_pid': parent, 'time_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['shard', 'gate'])
    parser.add_argument('--shard', type=int, choices=range(3))
    parser.add_argument('--parent', type=int)
    parser.add_argument('--child', type=int)
    args = parser.parse_args()
    label = f'shard-{args.shard}' if args.mode == 'shard' else 'gate'
    try:
        if args.mode == 'shard':
            run_shard(args.shard)
        else:
            gate(args.parent, args.child)
    except BaseException:
        save(BANK / f'{label}-exit.json', {'exit': 1})
        raise
    save(BANK / f'{label}-exit.json', {'exit': 0})
