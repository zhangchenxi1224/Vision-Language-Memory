"""External evaluation scheduling only; never changes the registered experiment.

Use a separate checkout for this helper and --repo for the frozen execution tree.
The coordinator temporarily stops only the original controller PID. Its four
writer processes keep running. An independent guard restores the controller if
coordination fails. All evaluation uses the frozen controller's commands and
validator; original official evaluation and final audit remain with that controller.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

SCHEMA = 'prefeval-unet-eval-assist/v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
        return {'pid': int(pid), 'start_token': fields[19], 'state': fields[0], 'command': command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(receipt, *, command=True):
    current = process(receipt['pid'])
    return bool(current and current['start_token'] == receipt['start_token'] and
                (not command or current['command'] == receipt['command']))


def live(receipt):
    current = process(receipt['pid'])
    return bool(current and current['start_token'] == receipt['start_token'] and current['state'] not in ('Z', 'X'))


def signal_exact(receipt, signum):
    require(receipt['host'] == socket.gethostname(), 'Cannot signal a process on another host')
    require(matches(receipt), 'Process identity changed; refusing to signal reused PID')
    os.kill(receipt['pid'], signum)  # Deliberately never killpg for the original controller.


@contextlib.contextmanager
def claim(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)



def atomic_task_claim(path, owner):
    """Shared-storage mkdir is atomic; flock is NOT cross-host on this volume.

    A claim is permanent. A failed owner requires stopping the entire assistance
    and reconciling its processes; claims are never stolen by TTL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        return False
    save(path / 'owner.json', owner)
    return True


@contextlib.contextmanager
def atomic_registration(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 30
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            require(time.monotonic() < deadline, 'Registration mutex is still owned; inspect before retry')
            time.sleep(.2)
    try:
        yield
    finally:
        path.rmdir()

def load_controller(repo):
    path = Path(repo) / 'scripts/inspire/run_prefeval_unet_init_ablation.py'
    spec = importlib.util.spec_from_file_location('frozen_unet_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_args(module, args, identity):
    # Paths and devices are reconstructed from the immutable registration, not
    # from possibly changed machine defaults.
    inputs = identity['inputs']
    values = {'repo': identity['repo'], 'run-root': identity['run_root'],
              'python': read(args.run_root / 'processes/formal-P-L.json')['command'][0], 'base': inputs['base_model']['path'],
              'reader': inputs['reader_model']['path'], 'official-source': inputs['official_source']['path'],
              'parent-checkpoint': inputs['parent']['path'],
              'train-variants': inputs['variants']['train']['path'],
              'dev-variants': inputs['variants']['dev']['path']}
    first_target = next(iter(inputs['targets'].values()))
    values['teachers'] = str(Path(first_target['latent']['path']).parent.parent)
    argv = [item for key, value in values.items() for item in ('--' + key, str(value))]
    argv += ['--gpus', ','.join(identity['gpus']), '--eval-gpus', ','.join(identity['eval_gpus'])]
    configured = module.configure(module.parser().parse_args(argv))
    require(module.file_binding(configured.python) == inputs['python'], 'Original interpreter binding changed')
    require(configured.repo.resolve() == args.repo.resolve(), 'Frozen repo differs from original identity')
    require(configured.run_root.resolve() == args.run_root.resolve(), 'Run root differs from original identity')
    return configured


def runtime():
    import torch
    import diffusers
    import transformers
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version', '--format=csv,noheader'], text=True)
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'diffusers': diffusers.__version__, 'transformers': transformers.__version__,
            'gpus': gpu.strip().splitlines(),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'diffusers', 'transformers', 'numpy')},
            'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version()}


def preflight(args):
    module = load_controller(args.repo)
    identity = read(args.run_root / 'identity.json')
    configured = original_args(module, args, identity)
    require(module.preflight(configured) == identity['inputs'], 'Frozen input preflight differs from registered identity')
    environment = runtime()
    # Driver/UUID are recorded, while model libraries and CUDA must match training.
    training_runtime = read(args.run_root / 'formal/P-L/manifest.json')['runtime']
    from packaging.version import Version
    require(bool(training_runtime.get('versions')), 'Training runtime package versions missing')
    for name, version in training_runtime['versions'].items():
        require(name in environment['versions'] and Version(environment['versions'][name]) == Version(version),
                f'Runtime package version mismatch: {name}')
    for key in ('cuda_version', 'cudnn_version'):
        require(environment[key] == training_runtime[key], f'Runtime mismatch: {key}')
    require(all('NVIDIA H200' in row for row in environment['gpus']), 'Expected matching H200 devices')
    return module, configured, identity, environment


def controller_receipt(args, module, identity):
    recorded = read(args.run_root / 'controller.json')
    require(recorded['pid'] == args.controller_pid and recorded['host'] == socket.gethostname(), 'Wrong controller PID/host')
    require(recorded['identity'] == object_sha(identity), 'Controller identity differs')
    current = process(args.controller_pid)
    require(current and current['start_token'] == args.controller_token and current['state'] not in ('Z', 'X', 'T'),
            'Controller token/state differs')
    require(str(args.repo / 'scripts/inspire/run_prefeval_unet_init_ablation.py') in current['command'], 'Not the frozen controller command')
    current['host'] = socket.gethostname()
    return current


def writer_receipts(args, module):
    receipts = []
    for arm in module.ARMS:
        value = read(args.run_root / 'processes' / f'formal-{arm}.json')
        require(value['host'] == socket.gethostname() and value['status'] == 'running', 'Writer is not running on controller host')
        require(matches(value) and live(value), f'Writer identity no longer active: {arm}')
        command = value['command']
        require(str(args.repo / 'scripts/experiments/prefeval_k1_writer.py') in command and 'train' in command,
                'Process is not the expected writer')
        require(str(args.run_root / 'formal' / arm) in command, 'Writer output differs')
        receipts.append(value)
    return receipts


def checkpoint_ready(module, configured, job, cache):
    path = module.checkpoint(configured, job['arm'], job['step'])
    if not path.is_file():
        return None
    complete = module.stage_output(configured, job['arm']) / 'complete.json'
    if job['step'] == module.STEPS and not complete.is_file():
        return None
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    key = str(path)
    if key in cache and cache[key]['stat'] == signature:
        return cache[key]
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    require(checkpoint.get('schema_version') == 1 and checkpoint.get('optimizer_step') == job['step'], 'Checkpoint step/schema mismatch')
    manifest = read(module.stage_output(configured, job['arm']) / 'manifest.json')
    require(checkpoint.get('manifest') == manifest, 'Checkpoint training manifest mismatch')
    require(bool(checkpoint.get('trainable_state')), 'Empty checkpoint state')
    digest = sha(path)
    after = path.stat()
    require(signature == (after.st_size, after.st_mtime_ns, after.st_ino), 'Checkpoint changed during readiness check')
    if job['step'] == module.STEPS:
        done = read(complete)
        require(done['steps'] == module.STEPS and done['checkpoint_sha256'] == digest and done['frozen_verified'] is True,
                'Formal checkpoint completion differs')
    result = {'path': str(path), 'sha256': digest, 'step': job['step'], 'stat': signature}
    cache[key] = result
    return result


def completed(args, job, identity_sha, checkpoint=None):
    path = args.assist_root / 'completed' / (job['label'] + '.json')
    if not path.exists():
        return False
    receipt = read(path)
    require(receipt['identity_sha256'] == identity_sha and receipt['label'] == job['label'], 'Assist completion identity differs')
    require(receipt['command'] == job['command'], 'Assist evaluation command changed')
    for filename, digest in receipt['artifacts'].items():
        require(sha(Path(job['output']) / filename) == digest, f'Completed evaluation artifact changed: {job["label"]}')
    if checkpoint is not None:
        require(receipt['checkpoint_sha256'] == checkpoint['sha256'], 'Completed checkpoint changed')
    return True


def heartbeat_alive(args):
    path = args.assist_root / 'heartbeat.json'
    if not path.exists():
        return False
    value = read(path)
    return time.time() - value['time'] < args.stale_seconds and value['state'] == 'active'


def should_abort(args):
    return (args.assist_root / 'abort.json').exists() or not heartbeat_alive(args)


def worker_pool(args, module, configured, identity_sha, runner, stop):
    jobs = list(module.evaluation_jobs(configured))
    def worker(gpu):
        cache = {}
        while not stop.is_set():
            require(not should_abort(args), 'Coordinator stopped or heartbeat expired')
            pending, worked = False, False
            for original in jobs:
                if completed(args, original, identity_sha):
                    continue
                pending = True
                # Checkpoint probing happens before permanent ownership so unavailable
                # final checkpoints do not consume the claim during early training.
                binding = checkpoint_ready(module, configured, original, cache)
                if binding is None:
                    continue
                owner = process(os.getpid())
                owner.update(host=socket.gethostname(), gpu=gpu, time=time.time(), identity_sha256=identity_sha)
                if not atomic_task_claim(args.assist_root / 'claims' / original['label'], owner):
                    continue
                require(not should_abort(args), 'Coordinator stopped before task dispatch')
                job = dict(original, gpu=gpu)
                result = module.evaluate_job(configured, runner, job)
                output = Path(job['output'])
                save(args.assist_root / 'completed' / (job['label'] + '.json'), {
                    'schema': SCHEMA, 'identity_sha256': identity_sha, 'label': job['label'],
                    'checkpoint_sha256': binding['sha256'], 'command': original['command'],
                    'host': socket.gethostname(), 'gpu': gpu, 'finished': time.time(), 'validation': result,
                    'artifacts': {name: sha(output / name) for name in ('complete.json', 'binding.json', 'summary.json')}})
                worked = True
                break
            if not pending:
                return
            if not worked:
                stop.wait(args.poll_seconds)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException as exc:
            save(args.assist_root / 'abort.json', {'error': repr(exc), 'host': socket.gethostname(), 'time': time.time()})
            stop.set()
            runner.wait_for_shutdown()
            raise



def owned_groups(args):
    """Find whole helper sessions, including children orphaned before receipt write.

    The dedicated inherited environment marker excludes every original writer.
    Descendants inherit the session and marker, so a dead wrapper leader cannot
    hide an active rollout/Reader child from cleanup.
    """
    marker = ('PREFEVAL_EVAL_ASSIST_ROOT=' + str(args.assist_root.resolve())).encode()
    groups = set()
    host_root = args.assist_root / 'hosts' / socket.gethostname()
    reserved = {read(host_root / 'worker.json')['pid']}
    if (host_root / 'guard-ready.json').exists():
        reserved.add(read(host_root / 'guard-ready.json')['pid'])
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if marker not in (path / 'environ').read_bytes().split(b'\0'):
                continue
            value = process(int(path.name))
            if value and value['state'] not in ('Z', 'X'):
                group = os.getpgid(value['pid'])
                # Owners and guards must each be session leaders. Every other
                # marked group is one of our evaluator sessions, including an
                # orphan launched immediately before its durable receipt write.
                if group not in reserved:
                    groups.add(group)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return groups


def cleanup_receipts(args):
    """Kill only marked helper evaluation sessions; never original writers."""
    groups = owned_groups(args)
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while owned_groups(args) and time.monotonic() < deadline:
        time.sleep(.2)
    for group in owned_groups(args):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    deadline = time.monotonic() + 15
    while owned_groups(args) and time.monotonic() < deadline:
        time.sleep(.2)
    require(not owned_groups(args), 'Helper GPU process groups did not exit; refusing drained acknowledgment')

def resume_original(args, lease):
    controller = lease['controller']
    require(matches(controller), 'Original controller is no longer the same process; manual recovery required')
    signal_exact(controller, signal.SIGCONT)
    deadline = time.monotonic() + 5
    while True:
        current = process(controller['pid'])
        require(current and current['start_token'] == controller['start_token'] and current['state'] not in ('Z', 'X'),
                'Original controller exited during recovery; inspect its failure log')
        if current['state'] not in ('T', 't'):
            break
        require(time.monotonic() < deadline, 'Original controller did not resume')
        time.sleep(.05)
    save(args.assist_root / 'resumed.json', {'time': time.time(), 'controller': controller, 'observed_state': current['state']})


def drain_local(args):
    cleanup_receipts(args)
    save(args.assist_root / 'hosts' / socket.gethostname() / 'drained.json',
         {'time': time.time(), 'host': socket.gethostname(), 'helper_gpu_children_terminated': True})


def all_hosts_drained(args):
    # Explicit expected hosts; a missing/stale directory listing or heartbeat
    # is never evidence that a remote CUDA process has exited.
    lease_path = args.assist_root / 'lease.json'
    if not lease_path.exists():
        return False
    primary = read(lease_path)['controller']['host']
    for host in (primary, args.extra_host):
        root = args.assist_root / 'hosts' / host
        if not (root / 'worker.json').exists() or not (root / 'drained.json').exists():
            return False
        drained = read(root / 'drained.json')
        if drained.get('helper_gpu_children_terminated') is not True or drained.get('host') != host:
            return False
    return True


def guard(args):
    host_root = args.assist_root / 'hosts' / socket.gethostname()
    owner = read(host_root / 'guard-owner.json')
    save(host_root / 'guard-ready.json', {'pid': os.getpid(), 'time': time.time()})
    while not (args.assist_root / 'released.json').exists():
        worker_receipt = read(host_root / 'worker.json')
        if worker_receipt.get('status') == 'exited':
            drain_local(args)
            return
        if not live(owner) or should_abort(args):
            save(args.assist_root / 'abort.json', {'reason': 'Assistant died or coordinator heartbeat expired',
                                                  'host': socket.gethostname(), 'time': time.time()})
            # Prevent the local owner from launching anything while the guard drains.
            if live(owner) and matches(owner):
                signal_exact(owner, signal.SIGTERM)
                deadline = time.monotonic() + 30
                while live(owner) and time.monotonic() < deadline:
                    time.sleep(.2)
                if live(owner) and matches(owner):
                    signal_exact(owner, signal.SIGKILL)
                deadline = time.monotonic() + 10
                while live(owner) and time.monotonic() < deadline:
                    time.sleep(.1)
                require(not live(owner), 'Owner did not exit; refusing drained acknowledgment')
            drain_local(args)
            if owner['role'] == 'coordinate':
                lease = read(args.assist_root / 'lease.json')
                save(args.assist_root / 'recovery-wait.json', {
                    'time': time.time(), 'reason': 'Waiting for every registered host to confirm GPU children drained; controller remains stopped'})
                while not all_hosts_drained(args):
                    time.sleep(min(args.poll_seconds, 5))
                resume_original(args, lease)
                save(args.assist_root / 'recovered.json', {'time': time.time(), 'all_hosts_drained': True})
            return
        time.sleep(min(args.poll_seconds, 5))


def start_guard(args, host_root, owner):
    require(not (host_root / 'guard-ready.json').exists(), 'Stale guard receipt requires inspection before reuse')
    save(host_root / 'guard-owner.json', owner)
    command = [sys.executable, str(Path(__file__).resolve()), 'guard', '--repo', str(args.repo),
               '--run-root', str(args.run_root), '--assist-root', str(args.assist_root),
               '--extra-host', args.extra_host, '--stale-seconds', str(args.stale_seconds)]
    with (host_root / 'guard.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 30
    while not (host_root / 'guard-ready.json').exists():
        require(child.poll() is None and time.monotonic() < deadline, 'Guard did not become ready')
        time.sleep(.1)
    return child

def common_registration(args, identity, environment):
    root = args.assist_root
    require(root.resolve() != args.run_root.resolve() and args.run_root.resolve() not in root.resolve().parents,
            'Assist receipts must be outside registered run root')
    registration = {'schema': SCHEMA, 'identity_sha256': object_sha(identity),
                    'identity_file_sha256': sha(args.run_root / 'identity.json'),
                    'helper_sha256': sha(Path(__file__)), 'repo': str(args.repo), 'run_root': str(args.run_root),
                    'scheduling_change': 'Early train/dev evaluation may overlap training on separate H200 GPUs; '
                    'official phase remains exclusively with original controller after all 16 endpoints.',
                    'expected_extra_host': args.extra_host, 'per_host_gpus': list(args.gpus)}
    with atomic_registration(root / 'registration.mutex'):
        path = root / 'registration.json'
        if path.exists():
            require(read(path) == registration, 'Assist registration changed')
        else:
            save(path, registration)
    host_root = root / 'hosts' / socket.gethostname()
    save(host_root / 'preflight.json', {'time': time.time(), 'registration': registration, 'runtime': environment,
                                       'inputs_match_original_identity': True})
    return host_root


def active_monitor(args, runner, stop, guard_process):
    while not stop.wait(min(args.poll_seconds, 5)):
        if (args.assist_root / 'released.json').exists():
            return
        if guard_process.poll() is not None:
            save(args.assist_root / 'abort.json', {'reason': 'Independent guard exited unexpectedly', 'host': socket.gethostname(), 'time': time.time()})
        if should_abort(args):
            runner.terminate()
            stop.set()
            return


def run_role(args):
    require(os.name == 'posix', 'Live assistance requires Linux /proc, flock, and signals')
    require(os.getpgrp() == os.getpid(), 'Launch each assistance role in its own session (start_new_session=True)')
    module, configured, identity, environment = preflight(args)
    identity_sha = object_sha(identity)
    host_root = common_registration(args, identity, environment)
    with claim(host_root / 'host.lock') as acquired:
        require(acquired, 'Another assistant is already active on this host')
        require(not (args.assist_root / 'abort.json').exists(), 'Previous assistance aborted; inspect receipts before recovery')
        require(not (args.assist_root / 'released.json').exists(), 'Assistance already released to original controller')
        runner_args = copy.copy(configured)
        runner_args.run_root = host_root
        os.environ['PREFEVAL_EVAL_ASSIST_ROOT'] = str(args.assist_root.resolve())
        runner = module.Runner(runner_args)
        stop = threading.Event()
        me = process(os.getpid())
        me['host'] = socket.gethostname()
        me['role'] = args.role
        me['status'] = 'running'
        save(host_root / 'worker.json', me)
        handlers = {}
        def terminate(signum, _frame):
            stop.set()
            runner.terminate()
            raise RuntimeError(f'Assistant received signal {signum}')
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, terminate)
        lease, heartbeat_stop, guard_process = None, threading.Event(), None
        try:
            if args.role == 'coordinate':
                controller = controller_receipt(args, module, identity)
                writers = writer_receipts(args, module)
                lease = {'schema': SCHEMA, 'controller': controller, 'coordinator': me,
                         'writers': writers, 'identity_sha256': identity_sha}
                require(not (args.assist_root / 'lease.json').exists(), 'Existing coordination lease must be inspected before reuse')
                save(args.assist_root / 'lease.json', lease)
                def beat():
                    while not heartbeat_stop.is_set():
                        save(args.assist_root / 'heartbeat.json', {'state': 'active', 'time': time.time(), 'coordinator': me})
                        heartbeat_stop.wait(5)
                beat_thread = threading.Thread(target=beat, daemon=True)
                beat_thread.start()
                while not heartbeat_alive(args):
                    time.sleep(.05)
                guard_process = start_guard(args, host_root, me)
                signal_exact(controller, signal.SIGSTOP)
                deadline = time.monotonic() + 5
                while process(controller['pid'])['state'] not in ('T', 't'):
                    require(time.monotonic() < deadline, 'Controller did not stop')
                    time.sleep(.05)
                for receipt in writers:
                    require(matches(receipt) and process(receipt['pid'])['state'] not in ('T', 't'), 'A writer was unexpectedly stopped')
                save(args.assist_root / 'paused.json', {'time': time.time(), 'controller': controller,
                                                      'only_controller_pid_stopped': True})
                while not all((module.stage_output(configured, arm) / 'complete.json').exists() for arm in module.ARMS):
                    require(guard_process.poll() is None, 'Independent guard exited unexpectedly')
                    require(not should_abort(args), 'Assistance aborted while waiting for training')
                    for arm, receipt in zip(module.ARMS, writers):
                        if not live(receipt):
                            require((module.stage_output(configured, arm) / 'complete.json').exists(), f'Writer exited without completion: {arm}')
                    stop.wait(args.poll_seconds)
                while any(live(receipt) for receipt in writers):
                    require(not should_abort(args), 'Assistance aborted while waiting for writers to exit')
                    stop.wait(args.poll_seconds)
                save(host_root / 'training-released-gpus.json', {'time': time.time(), 'writers_finished': True})
            else:
                require(socket.gethostname() == args.extra_host, 'Worker is on the wrong extra host')
                deadline = time.monotonic() + 600
                while not (args.assist_root / 'paused.json').exists():
                    require(time.monotonic() < deadline and not (args.assist_root / 'abort.json').exists(), 'No active coordinator pause lease')
                    stop.wait(1)
                guard_process = start_guard(args, host_root, me)
            monitor = threading.Thread(target=active_monitor, args=(args, runner, stop, guard_process), daemon=True)
            monitor.start()
            worker_pool(args, module, configured, identity_sha, runner, stop)
            require(not stop.is_set(), 'Worker pool stopped before successful completion')
            runner.wait_for_shutdown()
            drain_local(args)
            if args.role == 'coordinate':
                extra = args.assist_root / 'hosts' / args.extra_host / 'worker.json'
                while not extra.exists() or read(extra).get('status') != 'exited':
                    require(not should_abort(args), 'Assistance aborted while waiting for extra workers')
                    stop.wait(args.poll_seconds)
                require(all_hosts_drained(args), 'Registered helper host has not confirmed GPU children drained')
                require(all(completed(args, job, identity_sha) for job in module.evaluation_jobs(configured)), 'Not all 16 train/dev endpoints validated')
                resume_original(args, lease)
                save(args.assist_root / 'released.json', {'time': time.time(), 'all_16_validated': True,
                                                        'original_controller_handles_official_and_final_audit': True})
            me.update(status='exited', finished=time.time())
            save(host_root / 'worker.json', me)
        except BaseException as exc:
            save(args.assist_root / 'abort.json', {'host': socket.gethostname(), 'error': repr(exc), 'time': time.time()})
            runner.wait_for_shutdown()
            me.update(status='failed', error=repr(exc), finished=time.time())
            save(host_root / 'worker.json', me)
            drain_local(args)
            if lease is not None and all_hosts_drained(args):
                resume_original(args, lease)
            raise
        finally:
            stop.set()
            runner.wait_for_shutdown()
            heartbeat_stop.set()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=('coordinate', 'worker', 'guard'))
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--assist-root', type=Path, required=True)
    p.add_argument('--extra-host', required=True)
    p.add_argument('--controller-pid', type=int)
    p.add_argument('--controller-token')
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--stale-seconds', type=float, default=180)
    p.add_argument('--plan-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.gpus = tuple(args.gpus.split(','))
    require(len(args.gpus) == 4 and len(set(args.gpus)) == 4 and all(x.isdecimal() for x in args.gpus), 'Require four distinct local GPU IDs')
    require(args.poll_seconds > 0 and args.stale_seconds >= 30, 'Invalid poll/heartbeat deadline')
    if args.plan_only:
        print(json.dumps({'schema': SCHEMA, 'role': args.role, 'frozen_repo': str(args.repo),
                          'run_root': str(args.run_root), 'assist_root': str(args.assist_root),
                          'per_host_gpus': args.gpus, 'extra_host': args.extra_host,
                          'formal_training_changed': False, 'early_evaluation_overlap': True,
                          'official_evaluation_owner': 'original controller'}, indent=2))
    elif args.role == 'guard':
        guard(args)
    else:
        require(args.role != 'coordinate' or (args.controller_pid and args.controller_token), 'Coordinator requires exact controller PID and start token')
        run_role(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
