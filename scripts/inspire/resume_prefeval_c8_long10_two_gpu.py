"""Resume frozen long10 evaluation with four logical shards on two physical GPUs.

The original protocol.json is retained byte-for-byte as experiment identity.
Its original physical_gpus=4 is superseded by execution-topology.json, which
records the user-requested resource migration; no scoring or seed changes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
ORIGINAL = PROJECT/'repos/prefeval-c8-long10-20260928'
SPEC = importlib.util.spec_from_file_location('long10_frozen', ORIGINAL/'scripts/inspire/run_prefeval_c8_long10.py')
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)
base = frozen.base
save, sha = frozen.save, frozen.sha
PROTOCOL_SHA = 'b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33'


def main(output):
    output = output.resolve()
    assert output.parent == TASK/'long10'
    assert base.git_head(ORIGINAL) == '6563d16322c792d6ad0007c17cd6be9c6ca279e3'
    assert sha(ORIGINAL/'scripts/inspire/run_prefeval_c8_long10.py') == 'c5a9051c1d8426f45e017b69c960188c0eacf2c2dc507a782fc264a63783e4ce'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ORIGINAL, text=True).strip()
    assert sha(output/'protocol.json') == PROTOCOL_SHA
    assert (output/'migration-preparation.json').exists()
    assert len(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines()) == 2
    os.fstat(9)
    lock = (output/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allocation = (TASK/'allocation-locks'/f'{os.uname().nodename}.lock').open('a+')
    fcntl.flock(allocation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'host': os.uname().nodename, 'pid': os.getpid(), 'code': str(Path(__file__)),
        'code_sha256': sha(Path(__file__)), 'frozen_controller_commit': base.git_head(ORIGINAL),
        'worker_commit': base.FROZEN_COMMIT}
    topology = dict(state, physical_gpus=2, logical_shards=4, batches=[[0, 1], [2, 3]],
        mapping={'0': 0, '1': 1, '2': 0, '3': 1}, protocol_sha256=PROTOCOL_SHA,
        note='Original protocol physical_gpus=4 is historical. Actual hardware is two GPUs; all four logical shards, seeds and scores are unchanged.')
    save(output/'execution-topology.json', topology, immutable=True)

    def status(stage, **extra):
        save(output/'controller.json', dict(state, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def jobs(label, commands):
        assert [s for s, _ in commands] == list(range(4))
        for start in [0, 2]:
            for gpu in range(2):
                busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                assert not busy, f'GPU {gpu} occupied: {busy}'
            status(label, logical_batch=[start, start+1])
            children = []
            for shard, command in commands[start:start+2]:
                gpu = shard % 2
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                stream = (output/f'{label}-shard{shard}.log').open('a')
                command = list(map(str, command))
                child = subprocess.Popen(command, cwd=base.FROZEN, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(9, lock.fileno(), allocation.fileno()))
                children.append((child, stream))
                save(output/f'{label}-shard{shard}-process.json', dict(state, gpu=gpu, logical_shard=shard, pid=child.pid, command=command))
            codes = []
            for child, stream in children:
                codes.append(child.wait())
                stream.close()
            assert not any(codes), f'{label} batch{start}: {codes}'

    try:
        status('preflight')
        protocol = frozen.prepare(output, False)  # Frozen data/checkpoint hashing.
        assert sha(output/'protocol.json') == PROTOCOL_SHA
        assert protocol['logical_shards'] == 4 and protocol['noise_chains'] == 2
        assert not (output/'complete.json').exists(), 'Already complete; do not relaunch'
        smoke = TASK/'long10/smoke'
        done = json.loads((smoke/'complete.json').read_text())
        assert done['results_sha256'] == sha(smoke/'results.json')
        assert done['status'] == 'long10_integrity_smoke_complete'
        writer = base.FROZEN/'scripts/experiments/prefeval_k1_writer.py'
        evaluator = base.FROZEN/'scripts/experiments/prefeval_k1_evaluate.py'
        common = ['--arm', 'B', '--base', base.MODELS/'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT/'Vision-Language-Memory/third_party/DreamLite', '--split', 'train',
            '--ids-file', output/'ids.json', '--initial-variants', base.VARIANTS]
        audit = {}
        for arm in protocol['arms']:
            for variant in protocol['initial_variants']:
                images = output/arm/f'eval-V{variant}'
                jobs(f'generate-{arm}-V{variant}', [(s, [base.PYTHON, writer, 'rollout', *common,
                    '--checkpoint', base.CHECKPOINTS[arm][0], '--initial-variant', variant, '--inter-turns', 10,
                    '--noise-chains', protocol['noise_chains'], '--noise-domain', 'mt8-eval',
                    '--shard-index', s, '--shard-count', protocol['logical_shards'], '--output', images]) for s in range(4)])
                audit[f'{arm}/V{variant}'] = frozen.verify_png_chains(images, protocol, arm, variant)
                save(output/'png-chain-audit.json', audit)
                jobs(f'read-{arm}-V{variant}', [(s, [base.PYTHON, evaluator, '--kind', 'student', '--split', 'train',
                    '--ids-file', output/'ids.json', '--reader', base.MODELS/'Qwen3-VL-4B-Instruct',
                    '--initial-variants', base.VARIANTS, '--initial-variant', variant, '--images', images,
                    '--output', output/arm/f'read-V{variant}', '--families', 'T1,T2,T3', '--tasks', 'mcq',
                    '--prefixes', '0,5,10', '--noise-chains', protocol['noise_chains'],
                    '--controls', 'memory,mismatch,blank,text', '--shard', s, '--shards', 4]) for s in range(4)])
        save(output/'results.json', frozen.summarize(output, protocol))
        status('complete_analysis_required')
        save(output/'complete.json', {'status': 'fixed_C8_B0_long10_complete', 'results_sha256': sha(output/'results.json'),
            'protocol_sha256': sha(output/'protocol.json'), 'execution_topology_sha256': sha(output/'execution-topology.json')})
    except Exception as error:
        save(output/'failure.json', dict(state, time_utc=datetime.now(timezone.utc).isoformat(), error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
