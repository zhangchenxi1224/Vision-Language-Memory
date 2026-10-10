"""Two-host, eight-GPU DAG for the registered official-pretrained/random experiment.

Each host owns half the finite tasks. No shared flock and no expired-lease stealing.
Immutable checkpoint ready markers let evaluation overlap the four training jobs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

STEPS = (5840, 11680, 23360, 46720, 93440)
ARMS = ('P-L', 'P-H', 'R-L', 'R-H')
PLACEMENT = {'P-L': (0, 0), 'R-H': (0, 1), 'P-H': (1, 0), 'R-L': (1, 1)}
SHARDS = {'train': 24, 'dev': 4, 'diagnostics': 4, 'opposites': 2, 'official': 8}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'token': fields[19], 'state': fields[0],
                'command': (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


@dataclasses.dataclass
class Task:
    name: str
    host: int
    command: list[str]
    receipts: list[Path]
    dependencies: list[Path]
    priority: tuple
    gpu: int | None = None
    training: bool = False
    numeric_precheck: bool = False


def command(args, script, *items):
    return [str(args.python), str(args.repo / 'scripts' / 'experiments' / script),
            *[str(x) for x in items]]


def done(args, label):
    return args.run_root / 'tasks' / (label + '.complete.json')


def precheck_output(args, arm, attempt):
    return args.run_root / 'precheck' / f'attempt-{attempt}' / arm


def checkpoint(args, arm, step):
    return args.run_root / 'formal' / arm / f'checkpoint-step-{step:06d}.pt'


def writer_command(args, arm, mode, output, lr, attempt=0):
    cmd = command(args, 'prefeval_aug4_writer.py', mode,
                  '--data', args.data, '--teacher-root', args.run_root / 'teachers',
                  '--condition-root', args.run_root / 'conditions', '--base', args.base,
                  '--official-source', args.official_source, '--output', output,
                  '--init', 'pretrained' if arm.startswith('P') else 'random',
                  '--learning-rate', lr, '--train-seed', 20260924, '--init-seed', 20261009)
    if mode == 'train':
        cmd += ['--expected-init-audit', str(precheck_output(args, arm, attempt) / 'manifest.json')]
    if (output / 'resume.pt').exists() or (output / 'manifest.json').exists():
        cmd.append('--resume')
    return cmd


def build_tasks(args):
    root = args.run_root
    teacher_deps = [done(args, f'teacher-{i}') for i in range(8)]
    prepared = root / 'teacher-accepted.json'
    tasks = []
    for i in range(8):
        tasks.append(Task(f'teacher-{i}', i % 2,
            command(args, 'prefeval_aug4_teacher.py', '--data', args.data, '--base', args.base,
                    '--reader', args.reader, '--prefeval', args.prefeval, '--output', root / 'teachers',
                    '--shard', i, '--shards', 8),
            [root / 'teachers' / f'finished-{i}.json'], [], (0, i)))
        tasks.append(Task(f'conditions-{i}', i % 2,
            command(args, 'prefeval_aug4_writer.py', 'cache-conditions', '--data', args.data,
                    '--teacher-root', root / 'teachers', '--condition-root', root / 'conditions',
                    '--base', args.base, '--official-source', args.official_source,
                    '--output', root / 'condition-workers' / str(i), '--init', 'pretrained',
                    '--learning-rate', 5e-5, '--shard-index', i, '--shard-count', 8),
            [root / 'conditions' / f'shard-{i}.json'], teacher_deps, (1, i)))
    for i in range(24):
        tasks.append(Task(f'teacher-read-{i}', i % 2,
            command(args, 'prefeval_aug4_evaluate.py', 'read', '--kind', 'teacher', '--split', 'train',
                    '--data', args.data, '--reader', args.reader, '--prefeval', args.prefeval,
                    '--images', root / 'teachers', '--output', root / 'teacher-evaluation',
                    '--shard', i, '--shards', 24),
            [root / 'teacher-evaluation' / f'read-finished-{i}.json'], teacher_deps, (2, i)))
    ready = [prepared, *[done(args, f'conditions-{i}') for i in range(8)]]
    for arm in ARMS:
        h, g = PLACEMENT[arm]
        output = precheck_output(args, arm, 0)
        tasks.append(Task(f'precheck-0-{arm}', h,
            writer_command(args, arm, 'precheck', output, 5e-5 if arm.endswith('L') else 5e-4),
            [output / 'complete.json'], ready, (3, arm), gpu=g, numeric_precheck=arm.endswith('H')))
    decision = root / 'precheck-decision.json'
    if decision.exists():
        choice = read(decision)
        if choice['fallback']:
            for arm in ('P-H', 'R-H'):
                h, g = PLACEMENT[arm]
                output = precheck_output(args, arm, 1)
                tasks.append(Task(f'precheck-1-{arm}', h,
                    writer_command(args, arm, 'precheck', output, 1.5e-4),
                    [output / 'complete.json'], [decision], (4, arm), gpu=g))
        for arm in ARMS:
            h, g = PLACEMENT[arm]
            attempt = int(choice['fallback'] and arm.endswith('H'))
            output = root / 'formal' / arm
            tasks.append(Task(f'formal-{arm}', h,
                writer_command(args, arm, 'train', output,
                               5e-5 if arm.endswith('L') else choice['high_learning_rate'], attempt),
                [output / 'complete.json'], [root / 'training-released.json'],
                (5, arm), gpu=g, training=True))
    for step in STEPS:
        for arm in ARMS:
            ck = checkpoint(args, arm, step)
            for split, count in SHARDS.items():
                if split == 'official' and step != STEPS[-1]:
                    continue
                label = f'{arm}-{step:06d}-{split}'
                out = root / 'evaluation' / label
                gen_dependencies = [ck.with_suffix('.ready.json')]
                if split == 'official':
                    gen_dependencies += [done(args, f'formal-{a}') for a in ARMS]
                for i in range(count):
                    gen_name = f'generate-{label}-{i}'
                    common = ['--kind', 'student', '--split', split, '--data', args.data,
                              '--images', out / 'images', '--output', out,
                              '--reader', args.reader, '--prefeval', args.prefeval,
                              '--shard', i, '--shards', count]
                    canonical_dependencies = []
                    if split in ('diagnostics', 'opposites'):
                        common += ['--canonical-images', root / 'evaluation' / f'{arm}-{step:06d}-train' / 'images']
                        canonical_dependencies = [done(args, f'generate-{arm}-{step:06d}-train-{j}')
                                                  for j in range(SHARDS['train'])]
                    tasks.append(Task(gen_name, i % 2,
                        command(args, 'prefeval_aug4_evaluate.py', 'generate', *common,
                                '--base', args.base, '--official-source', args.official_source,
                                '--checkpoint', ck),
                        [out / f'generate-finished-{i}.json'], gen_dependencies + canonical_dependencies,
                        (6, step, list(SHARDS).index(split), arm, 0, i)))
                    tasks.append(Task(f'read-{label}-{i}', i % 2,
                        command(args, 'prefeval_aug4_evaluate.py', 'read', *common,
                                '--checkpoint', ck),
                        [out / f'read-finished-{i}.json'],
                        [done(args, f'generate-{label}-{j}') for j in range(count)] + canonical_dependencies,
                        (6, step, list(SHARDS).index(split), arm, 1, i)))
    return tasks


def numeric_failure(path):
    if not path.exists():
        return False
    data = read(path)
    reason = str(data.get('reason', '')).lower()
    return reason.startswith('nonfinite_')


def prepare_precheck_decision(args):
    root = args.run_root
    if (root / 'precheck-decision.json').exists():
        return
    complete = [(done(args, f'precheck-0-{a}').exists() or
                 (root / 'tasks' / f'precheck-0-{a}.numeric-failure.json').exists()) for a in ARMS]
    if not all(complete):
        return
    fallback = any((root / 'tasks' / f'precheck-0-{a}.numeric-failure.json').exists() for a in ('P-H', 'R-H'))
    save(root / 'precheck-decision.json', {'fallback': fallback,
         'high_learning_rate': 1.5e-4 if fallback else 5e-4,
         'reason': 'explicit_nonfinite_high_lr_precheck' if fallback else 'all_four_prechecks_passed',
         'time': time.time()})


def finalize_teachers(args):
    root = args.run_root
    if (root / 'teacher-accepted.json').exists():
        return
    if not all(done(args, f'teacher-read-{i}').exists() for i in range(24)):
        return
    metrics = {k: {'correct': 0, 'total': 0, 'parse_failures': 0} for k in ('memory', 'mismatch', 'blank', 'text')}
    assigned = []
    data_sha = sha(args.data)
    for i in range(24):
        result = read(root / 'teacher-evaluation' / f'read-finished-{i}.json')
        binding = result['binding']
        if (result['status'] != 'completed' or binding['data_sha256'] != data_sha
                or binding['kind'] != 'teacher' or binding['split'] != 'train'
                or binding['shard'] != i or binding['shards'] != 24
                or result['items'] != result['expected_items']):
            raise RuntimeError('Incomplete or wrong teacher evaluation shard')
        assigned.extend(binding['assignment'])
        for control in metrics:
            for key in metrics[control]:
                metrics[control][key] += result['counts'][control][key]
    expected_ids = [r['base_pair_id'] for r in read(args.data)['train']]
    if sorted(assigned) != sorted(expected_ids) or len(assigned) != 730:
        raise RuntimeError('Teacher evaluation does not cover each train preference exactly once')
    if any(v['total'] != 730 * 3 * 4 for v in metrics.values()):
        raise RuntimeError('Teacher question/position/control coverage incomplete')
    summary = {'controls': metrics, 'preferences': 730, 'data_sha256': data_sha,
               'selection': 'all_train_preferences_retained', 'time': time.time()}
    save(root / 'teacher-summary.json', summary)
    scores = {k: v['correct'] / v['total'] for k, v in metrics.items() if v['total']}
    if not (scores['memory'] > scores['mismatch'] and scores['memory'] > scores['blank']):
        raise RuntimeError('The complete frozen teacher bank has not shown memory benefit; no filtering or automatic retuning permitted')
    save(root / 'teacher-accepted.json', {'data_sha256': sha(args.data),
         'summary_sha256': sha(root / 'teacher-summary.json'), 'scores': scores,
         'policy': 'all730_retained_matched_beats_mismatch_and_blank_on_training_questions', 'time': time.time()})


def release_training(args):
    root = args.run_root
    decision = root / 'precheck-decision.json'
    if not decision.exists() or (root / 'training-released.json').exists():
        return
    choice = read(decision)
    manifests = {}
    for arm in ARMS:
        attempt = int(choice['fallback'] and arm.endswith('H'))
        if not done(args, f'precheck-{attempt}-{arm}').exists():
            return
        manifests[arm] = read(precheck_output(args, arm, attempt) / 'manifest.json')
    for key in ('shared_training_identity', 'architecture_sha256', 'parameter_schema_sha256',
                'unet_parameter_count', 'frozen_module_hashes', 'official_unet_sha256',
                'official_unet_config_sha256', 'runtime'):
        if any(manifests[a][key] != manifests['P-L'][key] for a in ARMS):
            raise RuntimeError(f'Paired training conditions differ: {key}')
    for a, b in [('P-L', 'P-H'), ('R-L', 'R-H')]:
        if manifests[a]['initial_state_sha256'] != manifests[b]['initial_state_sha256']:
            raise RuntimeError(f'Paired initial states differ: {a}, {b}')
    if manifests['P-L']['initial_state_sha256'] == manifests['R-L']['initial_state_sha256']:
        raise RuntimeError('Random and pretrained initial states unexpectedly match')
    save(root / 'training-released.json', {'time': time.time(), 'data_sha256': sha(args.data),
         'decision': choice, 'initial_states': {a: m['initial_state_sha256'] for a, m in manifests.items()},
         'placement': PLACEMENT, 'steps': STEPS, 'batch': 4})


def update_summaries(args):
    """Publish one completed checkpoint/split per pass while GPU children continue."""
    from scripts.reporting.summarize_prefeval_aug4 import summarize
    records = []
    pending = None
    for step in STEPS:
        for arm in ARMS:
            for split, shards in SHARDS.items():
                if split == 'official' and step != STEPS[-1]:
                    continue
                label = f'{arm}-{step:06d}-{split}'
                directory = args.run_root / 'evaluation' / label
                path = directory / 'summary.json'
                if path.exists():
                    records.append({'arm': arm, 'step': step, 'split': split,
                                    'summary': str(path), 'sha256': sha(path)})
                elif pending is None and all(done(args, f'read-{label}-{i}').exists() for i in range(shards)):
                    pending = directory
    if pending is not None:
        save(pending / 'summary.json', summarize(args.data, pending))
    index = args.run_root / 'results-index.json'
    if not index.exists() or read(index)['summaries'] != records:
        save(index, {'summaries': records, 'primary_endpoint': 93440,
                     'selection': 'fixed_endpoint_no_test_based_selection', 'time': time.time()})
    return len(records) == len(ARMS) * (len(STEPS) * 4 + 1)


def register(args):
    root = args.run_root
    root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.repo, text=True).strip():
        raise RuntimeError('Frozen execution checkout has tracked modifications')
    if not args.model_seal.is_file():
        raise RuntimeError('An independently verified model asset seal is required')
    seal = read(args.model_seal)
    if seal.get('schema') != 'model-assets/v1' or not seal.get('files'):
        raise RuntimeError('Invalid model asset seal')
    identity = {'schema': 'prefeval-aug4-balanced-eight-h200/v1', 'commit': commit,
        'data_sha256': sha(args.data), 'repo': str(args.repo), 'data': str(args.data),
        'base': str(args.base), 'reader': str(args.reader), 'official_source': str(args.official_source),
        'model_seal_sha256': sha(args.model_seal),
        'placement': {a: list(v) for a, v in PLACEMENT.items()}, 'teacher_shards': 8,
        'evaluation_shards': SHARDS, 'steps': list(STEPS), 'hosts': args.host_names,
        'shared_lock_policy': 'fixed_host_task_ownership_no_ttl_stealing',
        'initialization': 'official_pretrained_or_standard_random_no_task_checkpoint'}
    target = root / 'identity.json'
    if args.host_index == 0:
        if target.exists() and read(target) != identity:
            raise RuntimeError('Experiment registration changed')
        if not target.exists():
            save(target, identity)
    else:
        deadline = time.time() + 120
        while not target.exists() and time.time() < deadline:
            time.sleep(1)
        if not target.exists() or read(target) != identity:
            raise RuntimeError('Host registrations do not match')
    if socket.gethostname() != args.host_names[args.host_index]:
        raise RuntimeError('Wrong host identity')
    lock = root / 'hosts' / str(args.host_index) / 'controller.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        previous = read(lock / 'owner.json')
        current = proc(previous['pid']) if previous['host'] == socket.gethostname() else None
        if not args.resume or (current and current['token'] == previous['token'] and current['state'] not in ('Z', 'X')):
            raise RuntimeError('Controller owner exists; reconcile before recovery')
        # --resume is an operator action, never a timeout-based ownership steal.
    for receipt in (root / 'processes').glob('*.json'):
        child = read(receipt)
        if child.get('host_index') != args.host_index:
            continue
        if child.get('host') != socket.gethostname():
            if not (root / 'reconciled-hosts' / str(args.host_index) / child['host']).exists():
                raise RuntimeError('Prior host task ownership needs explicit reconciliation before restarting')
            continue
        current = proc(child['pid'])
        if current and current['token'] == child['token'] and current['state'] not in ('Z', 'X'):
            raise RuntimeError(f'Prior child is still alive; refusing duplicate work: {receipt.name}')
    owner = dict(proc(os.getpid()), host=socket.gethostname(), time=time.time())
    save(lock / 'owner.json', owner)
    save(lock.parent / 'registered.json', {'host': socket.gethostname(), 'commit': commit,
         'data_sha256': identity['data_sha256'], 'owner': owner})
    return identity


def run(args):
    identity = register(args)
    root = args.run_root
    running = {}
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            if stopped or (root / 'STOP').exists():
                raise RuntimeError('Stop requested; checkpoints and evaluation rows retained')
            if (root / 'failure.json').exists():
                raise RuntimeError('Another worker recorded a failure')
            for gpu, (task, child, log) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                log.close()
                receipt = {'task': task.name, 'host': args.host_index, 'gpu': gpu,
                           'returncode': code, 'time': time.time(), 'data_sha256': identity['data_sha256']}
                if code == 0 and all(p.exists() for p in task.receipts):
                    receipt['receipts'] = {str(p): sha(p) for p in task.receipts}
                    save(done(args, task.name), receipt)
                elif task.numeric_precheck and numeric_failure(Path(task.command[task.command.index('--output') + 1]) / 'numeric-failure.json'):
                    save(root / 'tasks' / (task.name + '.numeric-failure.json'), receipt)
                else:
                    raise RuntimeError(f'Task failed or receipt missing: {task.name}, exit={code}')
                del running[gpu]
            if args.host_index == 0:
                finalize_teachers(args)
                prepare_precheck_decision(args)
                release_training(args)
            summaries_complete = False
            if args.host_index == 0 and (root / 'training-released.json').exists():
                summaries_complete = update_summaries(args)
            tasks = build_tasks(args)
            both = all((root / 'hosts' / str(h) / 'registered.json').exists() for h in (0, 1))
            if both:
                for task in sorted(tasks, key=lambda t: t.priority):
                    if task.host != args.host_index or done(args, task.name).exists():
                        continue
                    if (root / 'tasks' / (task.name + '.numeric-failure.json')).exists():
                        continue
                    if any(t.name == task.name for t, _, _ in running.values()):
                        continue
                    if not all(p.exists() for p in task.dependencies):
                        continue
                    available = [i for i in range(4) if i not in running]
                    if task.gpu is not None:
                        available = [g for g in available if g == task.gpu]
                    elif task.priority[0] >= 6:
                        # Reserve training GPUs until their corresponding formal job is done.
                        available = [g for g in available if g >= 2 or all(
                            done(args, 'formal-' + a).exists() for a, (h, k) in PLACEMENT.items()
                            if h == args.host_index and k == g)]
                    if not available:
                        continue
                    gpu = available[0]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               PYTHONPATH=str(args.repo / 'src') + ':' + str(args.repo),
                               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8',
                               AUG4_ASSET_SEAL=str(args.model_seal),
                               AUG4_ASSET_SEAL_SHA256=identity['model_seal_sha256'])
                    log_path = root / 'logs' / (task.name + '.log')
                    log_path.parent.mkdir(exist_ok=True)
                    log = log_path.open('a')
                    child = subprocess.Popen(task.command, cwd=args.repo, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    save(root / 'processes' / (task.name + '.json'),
                         dict(proc(child.pid), host=socket.gethostname(), host_index=args.host_index,
                              gpu=gpu, command=task.command, time=time.time()))
                    running[gpu] = (task, child, log)
            save(root / 'hosts' / str(args.host_index) / 'status.json', {
                'time': time.time(), 'host': socket.gethostname(), 'pid': os.getpid(),
                'active': {str(g): t.name for g, (t, _, _) in running.items()},
                'completed': sum(done(args, t.name).exists() for t in tasks if t.host == args.host_index),
                'total': sum(t.host == args.host_index for t in tasks)})
            if (root / 'training-released.json').exists() and all(
                    done(args, t.name).exists() or (root / 'tasks' / (t.name + '.numeric-failure.json')).exists()
                    for t in tasks) and not running and (args.host_index != 0 or summaries_complete):
                save(root / 'hosts' / str(args.host_index) / 'complete.json', {'time': time.time()})
                if args.host_index == 0:
                    save(root / 'complete.json', {'time': time.time(), 'identity': identity, 'tasks': len(tasks)})
                return
            time.sleep(3)
    except BaseException as exc:
        failure = {'host': socket.gethostname(), 'host_index': args.host_index,
                   'error': repr(exc), 'traceback': traceback.format_exc(), 'time': time.time()}
        save(root / 'hosts' / str(args.host_index) / 'failure.json', failure)
        if not (root / 'failure.json').exists():
            save(root / 'failure.json', failure)
        for _, child, _ in running.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        deadline = time.time() + 30
        while any(p.poll() is None for _, p, _ in running.values()) and time.time() < deadline:
            time.sleep(.5)
        for _, child, log in running.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            log.close()
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['repo', 'run-root', 'data', 'python', 'base', 'reader', 'official-source', 'prefeval', 'model-seal']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--host-index', type=int, choices=[0, 1], required=True)
    p.add_argument('--host-names', nargs=2, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.plan_only:
        print(json.dumps([dataclasses.asdict(t) for t in build_tasks(a)], default=str, indent=2))
    else:
        run(a)
