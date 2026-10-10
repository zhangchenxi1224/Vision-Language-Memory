"""Preregistered B730 parent/random U-Net comparison; never optimizes teachers.

Four independent GPUs run the parent/random by low/high-LR factorial design.
After all training finishes, a dedicated worker on each GPU shares an evaluation queue.
Plan mode has no side effects.
Only explicit numeric precheck failures permit one joint high-LR reduction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
OFFICIAL_COMMIT = 'a6e20c8cc94027f37dd7c5a81b0b3b472aa18409'
ARMS = ('P-L', 'P-H', 'R-L', 'R-H')
LOW_ARMS = ('P-L', 'R-L')
HIGH_ARMS = ('P-H', 'R-H')
PARENT_ARMS = ('P-L', 'P-H')
RANDOM_ARMS = ('R-L', 'R-H')
STEPS, EARLY, PRECHECK, BATCH = 23360, 2048, 128, 4
LOW_LR, HIGH_LR, FALLBACK_LR = 5e-5, 5e-4, 1.5e-4
TRAIN_SEED, INIT_SEED, CONDITIONING_SEED = 20260924, 20261009, 0
PROBE_INTERVAL, PROBE_COUNT, PROBE_SEED = 128, 32, 20261009


class ProtocolError(RuntimeError):
    """Fail closed on inconsistent inputs, stale outputs, or unexpected failures."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def immutable_json(path, value):
    if Path(path).exists():
        require(read_json(path) == value, f'Identity changed: {path}')
    else:
        save(path, value)


def parse_gpus(value):
    result = tuple(x.strip() for x in value.split(','))
    if len(result) != 4 or len(set(result)) != 4 or any(not x.isdecimal() for x in result):
        raise argparse.ArgumentTypeError('--gpus requires four distinct numeric GPU IDs, e.g. 0,1,2,3')
    return result


def parse_eval_gpus(value):
    result = tuple(x.strip() for x in value.split(','))
    if not 1 <= len(result) <= 4 or len(set(result)) != len(result) or any(not x.isdecimal() for x in result):
        raise argparse.ArgumentTypeError('--eval-gpus requires one to four distinct numeric GPU IDs, e.g. 0,1,2,3')
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'repo': REPO,
        'run-root': ROOT / 'runs/prefeval-unet-init-ablation-20261009',
        'python': ROOT / 'envs/vlm-r3-ngc2502/bin/python',
        'base': MODELS / 'DreamLite-base-a9a0f15-20260907',
        'official-source': ROOT / 'Vision-Language-Memory/third_party/DreamLite',
        'reader': MODELS / 'Qwen3-VL-4B-Instruct',
        'parent-checkpoint': ROOT / 'runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt',
        'teachers': ROOT / 'runs/prefeval-k1-scale730-20260924/teachers/B',
        'train-variants': ROOT / 'runs/prefeval-b-mcq-20260925/variants-train.json',
        'dev-variants': ROOT / 'runs/prefeval-b-mcq-20260925/variants-dev.json',
    }
    for name, default in defaults.items():
        p.add_argument('--' + name, type=Path, default=default)
    p.add_argument('--writer', type=Path)
    p.add_argument('--evaluator', type=Path)
    p.add_argument('--benchmark-ack', type=Path)
    p.add_argument('--gpus', type=parse_gpus, default=('0', '1', '2', '3'))
    evaluation = p.add_mutually_exclusive_group()
    evaluation.add_argument('--eval-gpus', type=parse_eval_gpus,
                            help='Evaluation workers after training; defaults to all four --gpus (0,1,2,3).')
    evaluation.add_argument('--eval-gpu', dest='legacy_eval_gpu',
                            help='Compatibility alias for a single evaluation worker.')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume only the identical registration in this run root.')
    return p


def configure(args):
    align = args.repo / 'reports/prefeval-official-alignment-20260923'
    # The record adapter resolves these relative to its repository. Do not offer
    # misleading alternate paths that the actual Writer would never consume.
    args.split_file = align / 'WRITER_IMPLEMENTATION_SPLIT.json'
    args.train_data = align / 'data/sft-train-10interturn.jsonl.gz'
    args.writer = args.writer or args.repo / 'scripts/experiments/prefeval_k1_writer.py'
    args.evaluator = args.evaluator or args.repo / 'scripts/experiments/prefeval_unet_init_evaluate.py'
    args.benchmark_ack = args.benchmark_ack or args.repo / 'scripts/experiments/prefeval_k1_benchmark_ack.py'
    if args.legacy_eval_gpu is not None:
        require(str(args.legacy_eval_gpu).isdecimal(), '--eval-gpu must be a numeric GPU ID')
        args.eval_gpus = (str(args.legacy_eval_gpu),)
    elif args.eval_gpus is None:
        args.eval_gpus = tuple(args.gpus)
    require(1 <= len(args.eval_gpus) <= 4 and len(set(args.eval_gpus)) == len(args.eval_gpus) and
            all(str(gpu).isdecimal() for gpu in args.eval_gpus), 'Evaluation requires distinct numeric GPU IDs')
    require(set(args.eval_gpus) <= set(args.gpus), 'Evaluation must reuse the four training GPUs after training')
    args.eval_gpu = args.eval_gpus[0]  # One coordinator generates the shared official history.
    return args


def learning_rates(high):
    return {arm: LOW_LR if arm in LOW_ARMS else high for arm in ARMS}


def protocol():
    return {
        'schema': 'prefeval-unet-init-ablation-2x2/v2',
        'groups': {'P-L': {'unet_init': 'parent', 'learning_rate': LOW_LR},
                   'P-H': {'unet_init': 'parent', 'learning_rate': HIGH_LR},
                   'R-L': {'unet_init': 'random', 'learning_rate': LOW_LR},
                   'R-H': {'unet_init': 'random', 'learning_rate': HIGH_LR}},
        'training': {'steps_per_group': STEPS, 'total_formal_updates': len(ARMS) * STEPS,
                     'effective_batch': BATCH, 'preferences': 730,
                     'exposures_per_preference': 128, 'assistant_variants': [0, 1],
                     'exposures_per_assistant_variant': 64, 'snapshots': [EARLY],
                     'train_seed': TRAIN_SEED, 'init_seed': INIT_SEED, 'conditioning_seed': CONDITIONING_SEED,
                     'fm_probe_interval': PROBE_INTERVAL, 'fm_probe_count': PROBE_COUNT, 'fm_probe_seed': PROBE_SEED,
                     'teachers': 'unchanged complete B teachers at step 288; no optimization or filtering',
                     'user_text': 'existing B730 originals; NOT the new 4x3 augmentation'},
        'precheck': {'steps_per_run': PRECHECK, 'initial_groups': list(ARMS),
                     'joint_high_lr_fallback': FALLBACK_LR, 'max_fallback_attempts': 1,
                     'trigger': 'numeric-failure.json with explicit numeric-failure schema in a high-LR precheck',
                     'other_failures': 'fail immediately; no learning-rate fallback',
                     'formal_start': 'fresh new outputs; never continue from precheck',
                     'cost': 'logged separately from 23360 formal updates'},
        'evaluation': {'steps': [EARLY, STEPS], 'splits': ['train', 'dev'], 'train_dev_runs': 16, 'total_runs': 20,
                       'initial_variant': 1, 'noise_chains': 2,
                       'schedule': 'after all four training jobs finish; one worker per evaluation GPU shares a train-first queue; no overlap with training',
                       'official180': {'runs': 4, 'step': STEPS, 'after': 'all 16 train/dev evaluations complete',
                                       'history': 'one frozen tested-Reader acknowledgment per explicit preference',
                                       'ack_system_prompt': 'You are a helpful assistant.', 'ack_max_new_tokens': 300,
                                       'ack_do_sample': False, 'ack_seed': 0,
                                       'ack_input': 'preference only; no future query, options, or SFT acknowledgment'},
                       'selection': 'no dev-based selection, teacher filtering, or budget extension'},
    }


def stage_output(args, arm, attempt=None):
    return args.run_root / ('formal' if attempt is None else f'precheck/attempt-{attempt}') / arm


def train_command(args, arm, lr, output, *, precheck=False, fresh=True):
    command = [args.python, args.writer, 'train', '--arm', 'B', '--split', 'train', '--stage', 'write',
               '--base', args.base, '--official-source', args.official_source,
               '--checkpoint', args.parent_checkpoint, '--teachers', args.teachers,
               '--initial-variants', args.train_variants, '--output', output, '--device', 'cuda:0',
               '--steps', str(PRECHECK if precheck else STEPS), '--unet-init', 'random' if arm in RANDOM_ARMS else 'parent',
               '--learning-rate', str(lr), '--train-seed', str(TRAIN_SEED), '--init-seed', str(INIT_SEED),
               '--conditioning-seed', str(CONDITIONING_SEED), '--audit-updates',
               '--fm-probe-interval', str(PROBE_INTERVAL), '--fm-probe-count', str(PROBE_COUNT),
               '--fm-probe-seed', str(PROBE_SEED)]
    if not precheck:
        command += ['--snapshot-steps', str(EARLY)]
        attempt = 1 if arm in HIGH_ARMS and lr == FALLBACK_LR else 0
        command += ['--expected-init-audit', str(stage_output(args, arm, attempt) / 'manifest.json')]
    if fresh:
        command += ['--fresh-start']
    return [str(x) for x in command]


def checkpoint(args, arm, step):
    return stage_output(args, arm) / ('checkpoint-final.pt' if step == STEPS else f'checkpoint-step-{step:06d}.pt')


def evaluation_jobs(args, *, official=False):
    for split in (('official',) if official else ('train', 'dev')):
        for arm in ARMS:
            for step in ((STEPS,) if official else (EARLY, STEPS)):
                label = f'{arm}-step-{step:06d}-{split}'
                output = args.run_root / 'evaluation' / label
                command = [args.python, args.evaluator, '--checkpoint', checkpoint(args, arm, step),
                           '--expected-step', str(step), '--split', split, '--output', output,
                           '--base', args.base, '--official-source', args.official_source, '--reader', args.reader,
                           '--device', 'cuda:0', '--noise-chains', '2', '--initial-variant', '1']
                command += (['--history-file', args.run_root / 'official-history.json'] if official else
                            ['--initial-variants', args.train_variants if split == 'train' else args.dev_variants])
                yield {'label': label, 'arm': arm, 'step': step, 'split': split, 'eligible_gpus': list(args.eval_gpus),
                       'output': str(output), 'command': [str(x) for x in command]}


def plan(args):
    result = {'protocol': protocol(), 'prechecks': [], 'fallback_prechecks': [], 'formal_alternatives': {},
              'resources': {'training_gpus': list(args.gpus), 'evaluation_gpus': list(args.eval_gpus),
                            'evaluation_workers': len(args.eval_gpus), 'official_history_gpu': args.eval_gpu,
                            'training_evaluation_overlap': False, 'max_simultaneous_gpus': len(args.gpus),
                            'evaluation_dispatch': 'one dedicated worker per GPU; shared queue, train before dev; GPU assigned when dequeued'},
              'evaluations': list(evaluation_jobs(args)), 'official_history': ack_command(args),
              'official_evaluations_after_train_dev': list(evaluation_jobs(args, official=True))}
    for arm, gpu in zip(ARMS, args.gpus):
        lr = LOW_LR if arm in LOW_ARMS else HIGH_LR
        result['prechecks'].append({'arm': arm, 'gpu': gpu, 'command': train_command(
            args, arm, lr, stage_output(args, arm, 0), precheck=True)})
        if arm in HIGH_ARMS:
            result['fallback_prechecks'].append({'arm': arm, 'gpu': gpu, 'command': train_command(
                args, arm, FALLBACK_LR, stage_output(args, arm, 1), precheck=True)})
    for high in (HIGH_LR, FALLBACK_LR):
        result['formal_alternatives'][str(high)] = [
            {'arm': arm, 'gpu': gpu, 'command': train_command(args, arm, LOW_LR if arm in LOW_ARMS else high,
                                                            stage_output(args, arm))}
            for arm, gpu in zip(ARMS, args.gpus)]
    return result


def file_binding(path):
    path = Path(path)
    require(path.is_file(), f'Missing input: {path}')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def tree_binding(path):
    path = Path(path)
    require(path.is_dir(), f'Missing model directory: {path}')
    files = {p.relative_to(path).as_posix(): file_binding(p) for p in sorted(path.rglob('*'))
             if p.is_file() and '.git' not in p.parts and '.cache' not in p.parts}
    require(files, f'Empty model directory: {path}')
    return {'path': str(path.resolve()), 'files': files, 'sha256': object_sha(files)}


CHECKPOINT_PROBE = """
import hashlib,json,sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if p.get('schema_version')!=1 or not p.get('trainable_state'): raise ValueError('Invalid parent/checkpoint schema')
state=p['trainable_state']
if not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()):
    raise ValueError('Checkpoint includes invalid/nonfinite parameters')
shape={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()}
print(json.dumps({'optimizer_step':p.get('optimizer_step'),'manifest':p.get('manifest'),
 'parameter_count':sum(v.numel() for v in state.values()),'tensor_count':len(state),
 'architecture_sha256':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
 'runtime':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}}))
"""


def inspect_checkpoint(args, path):
    output = subprocess.check_output([str(args.python), '-c', CHECKPOINT_PROBE, str(path)], text=True)
    return json.loads(output)


def preflight(args):
    """Read and hash every fixed teacher and all immutable model/input files; no CUDA work."""
    split = read_json(args.split_file)
    ids, dev_ids = split['train_ids'], split['internal_dev_ids']
    require(len(ids) == len(set(ids)) == 730, 'Training split must contain exactly 730 distinct IDs')
    require(len(dev_ids) == len(set(dev_ids)) == 90 and not set(ids) & set(dev_ids), 'Invalid dev90 split')
    with gzip.open(args.train_data, 'rt', encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    by_id = {r['base_pair_id']: r for r in records}
    require(len(by_id) == len(records), 'Duplicate source rows')
    variant_bindings = {}
    for name, path, expected in [('train', args.train_variants, ids), ('dev', args.dev_variants, dev_ids)]:
        artifact = read_json(path)
        require(artifact.get('review_complete') is True, f'Unreviewed {name} acknowledgment variants')
        require(set(artifact['items']) == set(expected), f'{name} variant IDs differ from the frozen split')
        for pid in expected:
            item, history = artifact['items'][pid], by_id[pid]['history']
            require(item['preference'] == history[0]['content'], f'User preference changed: {pid}')
            values = item['variants']
            require(len(values) == 3 and all(isinstance(v, str) and v.strip() for v in values), f'Bad V0/V1/V2: {pid}')
            require(values[0] == history[1]['content'] and values[2] not in values[:2], f'Invalid acknowledgment variants: {pid}')
        variant_bindings[name] = file_binding(path)
    targets = {}
    for pid in ids:
        directory = args.teachers / pid.replace(':', '_')
        done = read_json(directory / 'complete.json')
        require(done.get('step') == 288 and done.get('binding', {}).get('arm') == 'B', f'Incomplete/non-B teacher: {pid}')
        latent = file_binding(directory / 'latent.pt')
        require(latent['sha256'] == done['latent_sha256'], f'Teacher latent hash mismatch: {pid}')
        targets[pid] = {'latent': latent, 'completion': file_binding(directory / 'complete.json')}
    official = subprocess.check_output(['git', '-C', str(args.official_source), 'rev-parse', 'HEAD'], text=True).strip()
    require(official == OFFICIAL_COMMIT, 'Official DreamLite commit differs from the fixed protocol')
    dirty = subprocess.check_output(['git', '-C', str(args.official_source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    require(not dirty.strip(), 'Tracked official DreamLite source is modified')
    code = {str(p.relative_to(args.repo)): file_binding(p) for folder in ('src', 'scripts/experiments', 'scripts/eval')
            for p in sorted((args.repo / folder).rglob('*.py'))}
    code['controller'] = file_binding(Path(__file__))
    code['writer'] = file_binding(args.writer)
    code['evaluator'] = file_binding(args.evaluator)
    code['benchmark_ack'] = file_binding(args.benchmark_ack)
    reports = args.repo / 'reports/prefeval-k1-l0-l2-20260924'
    forms = {name: file_binding(reports / name) for name in
             ('train-question-forms.json', 'dev-question-forms.json', 'official-question-forms.json')}
    align_data = args.repo / 'reports/prefeval-official-alignment-20260923/data'
    return {'protocol_sha256': object_sha(protocol()), 'train_ids': ids, 'dev_ids': dev_ids,
            'source_rows': file_binding(args.train_data), 'split': file_binding(args.split_file),
            'benchmark': file_binding(align_data / 'benchmark-disclosures.jsonl.gz'),
            'variants': variant_bindings, 'question_forms': forms, 'targets': targets,
            'parent': file_binding(args.parent_checkpoint), 'parent_metadata': inspect_checkpoint(args, args.parent_checkpoint),
            'base_model': tree_binding(args.base), 'reader_model': tree_binding(args.reader),
            'official_source': {'path': str(args.official_source.resolve()), 'commit': official},
            'python': file_binding(args.python), 'code': code}


@contextlib.contextmanager
def controller_lock(run_root):
    require(os.name == 'posix', 'Training orchestration requires POSIX flock and process groups; plan/preflight are portable')
    import fcntl
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / 'controller.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(f'Another controller holds {run_root}/controller.lock') from exc
        yield


def process_start_token(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, IndexError):
        return None


def refuse_live_previous_children(run_root):
    for path in (run_root / 'processes').glob('*.json'):
        receipt = read_json(path)
        if receipt.get('status') != 'running':
            continue
        if receipt.get('host') != socket.gethostname():
            raise ProtocolError(f'Unresolved process on another host: {path}')
        token = receipt.get('start_token')
        require(not token or process_start_token(receipt['pid']) != token,
                f'Previous child is still alive; refusing duplicate work: {path}')


class Runner:
    def __init__(self, args):
        self.args, self.stop = args, threading.Event()
        self.children, self.mutex = {}, threading.RLock()

    def terminate(self):
        self.stop.set()
        with self.mutex:
            children = list(self.children.values())
        for proc in children:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)

    def wait_for_shutdown(self):
        self.terminate()
        with self.mutex:
            children = list(self.children.values())
        for proc in children:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)

    def run(self, label, gpu, command):
        require(not self.stop.is_set(), 'Controller is stopping')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        log_path = self.args.run_root / 'logs' / f'{label}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nCONTROLLER_START {time.time()}\n'); log.flush()
            with self.mutex:
                require(not self.stop.is_set(), 'Controller is stopping')
                proc = subprocess.Popen(command, cwd=self.args.repo, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                self.children[label] = proc
            state = {'label': label, 'gpu': str(gpu), 'pid': proc.pid, 'start_token': process_start_token(proc.pid),
                     'host': socket.gethostname(), 'command': command, 'started': time.time(), 'status': 'running'}
            receipt = self.args.run_root / 'processes' / f'{label}.json'
            save(receipt, state)
            code = proc.wait()
            with self.mutex:
                self.children.pop(label, None)
            state.update(exit_code=code, finished=time.time(), status='complete' if code == 0 else 'failed')
            save(receipt, state)
        return code


def register(args, binding):
    identity = {'schema': 'prefeval-unet-init-controller-2x2/v3', 'run_root': str(args.run_root.resolve()),
                'gpus': list(args.gpus), 'eval_gpus': list(args.eval_gpus),
                'official_history_gpu': args.eval_gpu, 'repo': str(args.repo.resolve()),
                'protocol': protocol(), 'inputs': binding}
    path = args.run_root / 'identity.json'
    if path.exists():
        require(args.resume, 'Run root already registered; use --resume only for the identical experiment')
        require(read_json(path) == identity, 'Resume identity changed; use a new run root')
    else:
        require(not args.resume, '--resume requires an existing identity.json in this exact run root')
        unexpected = [p.name for p in args.run_root.iterdir() if p.name != 'controller.lock']
        require(not unexpected, f'Fresh run root contains unregistered files: {unexpected}')
        save(path, identity)
    immutable_json(args.run_root / 'preregister.json', protocol())
    save(args.run_root / 'preflight.json', binding)
    return object_sha(identity)


def successful_stage(args, output, steps):
    done, manifest = read_json(output / 'complete.json'), read_json(output / 'manifest.json')
    require(done.get('steps') == steps and manifest.get('steps') == steps, f'Unexpected completed budget: {output}')
    require(done.get('checkpoint_sha256') == sha(output / 'checkpoint-final.pt'), f'Checkpoint hash mismatch: {output}')
    require(done.get('frozen_verified') is True, f'Frozen-module verification missing: {output}')
    require(done.get('shared_training_identity') == manifest.get('shared_training_identity'), f'Completion identity mismatch: {output}')
    require(done.get('initial_state_sha256') == manifest.get('initial_state_sha256'), f'Initial state mismatch: {output}')
    metadata = inspect_checkpoint(args, output / 'checkpoint-final.pt')
    require(metadata['optimizer_step'] == steps and metadata['manifest'] == manifest, f'Checkpoint metadata differs: {output}')
    if steps == STEPS:
        early = inspect_checkpoint(args, output / f'checkpoint-step-{EARLY:06d}.pt')
        require(early['optimizer_step'] == EARLY and early['manifest'] == manifest, f'Early checkpoint differs: {output}')
    audit = audit_optimization(output / 'optimization.jsonl', steps, set(manifest['targets']))
    return {'status': 'complete', 'output': str(output), 'steps': steps, 'manifest': manifest,
            'checkpoint_sha256': done['checkpoint_sha256'], **audit}


def audit_optimization(path, steps, ids):
    """Verify actual paired draws and exposures, excluding loss/gradient/timing values."""
    require(len(ids) == 730, 'Optimization audit requires all 730 teacher IDs')
    sequence, prefix = hashlib.sha256(), hashlib.sha256()
    exposure, variants = Counter(), Counter()
    completed = 0
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            completed += 1
            require(row['step'] == completed and len(row['draws']) == BATCH, 'Optimization steps/draw counts are not contiguous')
            for micro, draw in enumerate(row['draws']):
                pid, variant = draw['pair_id'], draw['initial_variant']
                require(pid in ids and variant in (0, 1) and draw['position'] == 0, 'Unexpected training draw/variant/position')
                require(draw['draw'] == (completed - 1) * BATCH + micro, 'Global training draw index differs')
                require(isinstance(draw['noise_seed'], int), 'Missing actual noise seed')
                require(isinstance(draw['noise_sha256'], str) and len(draw['noise_sha256']) == 64 and
                        all(c in '0123456789abcdef' for c in draw['noise_sha256']), 'Missing actual noise tensor hash')
                require(math.isfinite(draw['sigma']) and 0 <= draw['sigma'] < 1, 'Invalid sigma in actual draw log')
                entry = {k: draw[k] for k in ('pair_id', 'draw', 'initial_variant', 'sigma', 'noise_seed', 'noise_sha256')}
                packed = canonical(entry) + b'\n'
                sequence.update(packed)
                if completed <= PRECHECK:
                    prefix.update(packed)
                exposure[pid] += 1
                variants[pid, variant] += 1
    require(completed == steps, f'Optimization log has {completed} updates, expected {steps}')
    if steps == STEPS:
        require(set(exposure) == ids and set(exposure.values()) == {128}, 'Formal preference exposure differs from 128')
        require(set(variants) == {(pid, v) for pid in ids for v in (0, 1)} and set(variants.values()) == {64},
                'Formal V0/V1 exposure differs from 64 each')
    return {'optimization_sha256': sha(path), 'actual_draw_signature': sequence.hexdigest(),
            'first_128_draw_signature': prefix.hexdigest(), 'logged_updates': completed,
            'logged_draws': completed * BATCH, 'preference_count_observed': len(exposure)}


def validate_stage_options(args, result, arm, lr, *, formal=False):
    manifest = result['manifest']
    expected = {'learning_rate': lr, 'unet_init': 'random' if arm in RANDOM_ARMS else 'parent',
                'train_seed': TRAIN_SEED, 'init_seed': INIT_SEED, 'conditioning_seed': CONDITIONING_SEED}
    for key, value in expected.items():
        require(manifest.get(key) == value, f'Unexpected {key} for {arm}: {manifest.get(key)}')
    if hasattr(args, '_inputs'):
        require(manifest['parent_sha256'] == args._inputs['parent']['sha256'], f'Unexpected parent: {arm}')
        targets = {pid: value['latent']['sha256'] for pid, value in args._inputs['targets'].items()}
        require(manifest['targets'] == targets, f'Teacher bank changed or filtered: {arm}')
    if formal:
        attempt = 1 if arm in HIGH_ARMS and lr == FALLBACK_LR else 0
        audit = stage_output(args, arm, attempt) / 'manifest.json'
        require(manifest.get('expected_init_audit_sha256') == sha(audit), f'Missing pre-update initialization gate: {arm}')
    return result


def numeric_failure(output):
    path = output / 'numeric-failure.json'
    if not path.exists():
        return None
    value = read_json(path)
    require(value.get('schema') == 'prefeval-k1-numeric-failure/v1' and value.get('reason'),
            f'Unrecognized numeric failure sentinel: {path}')
    return value


def train_stage(args, runner, identity, arm, lr, attempt=None):
    output = stage_output(args, arm, attempt)
    precheck = attempt is not None
    label = f'precheck-{attempt}-{arm}' if precheck else f'formal-{arm}'
    command = train_command(args, arm, lr, output, precheck=precheck, fresh=False)
    receipt = args.run_root / 'bindings' / f'{label}.json'
    receipt_value = {'identity': identity, 'command_without_fresh_flag': command, 'output': str(output.resolve())}
    if receipt.exists():
        require(read_json(receipt) == receipt_value and args.resume, f'Existing stage requires identical --resume: {label}')
    else:
        require(not output.exists(), f'Unregistered output already exists: {output}')
        save(receipt, receipt_value)
    failed = numeric_failure(output)
    if failed:
        require(precheck and arm in HIGH_ARMS, f'Numeric failure outside high-LR precheck: {output}')
        return {'status': 'numeric_failure', 'output': str(output), 'failure': failed}
    if (output / 'complete.json').exists():
        return validate_stage_options(args, successful_stage(args, output, PRECHECK if precheck else STEPS),
                                      arm, lr, formal=not precheck)
    nonempty = output.exists() and any(output.iterdir())
    require(not nonempty or (args.resume and (output / 'resume.pt').is_file()),
            f'Partial stage has no safe same-directory resume checkpoint: {output}')
    command = train_command(args, arm, lr, output, precheck=precheck, fresh=not nonempty)
    code = runner.run(label, args.gpus[ARMS.index(arm)], command)
    failed = numeric_failure(output)
    if failed:
        require(precheck and arm in HIGH_ARMS, f'Numeric failure outside high-LR precheck: {output}')
        return {'status': 'numeric_failure', 'output': str(output), 'failure': failed, 'exit_code': code}
    require(code == 0, f'{label} failed ({code}); no numeric sentinel, so no LR fallback')
    return validate_stage_options(args, successful_stage(args, output, PRECHECK if precheck else STEPS),
                                  arm, lr, formal=not precheck)


def compare_training(results, *, expected_shared=None):
    require(set(results) == set(ARMS), 'All four factorial groups must be present')
    manifests = {arm: result['manifest'] for arm, result in results.items()}
    reference = manifests['P-L']
    shared = reference['shared_training_identity']
    required = {'conditions_sha256', 'targets_sha256', 'training_rows_sha256', 'schedule_sha256', 'train_seed', 'effective_batch'}
    require(set(shared) >= required and shared['train_seed'] == TRAIN_SEED and shared['effective_batch'] == BATCH,
            'Missing/inconsistent shared training identity')
    require(expected_shared is None or expected_shared == shared, 'Precheck/formal shared identity changed')
    for arm, manifest in manifests.items():
        require(manifest['shared_training_identity'] == shared, f'Conditioning/target/schedule mismatch: {arm}')
        for key in ('architecture_sha256', 'frozen_module_hashes', 'runtime', 'conditioning_seed', 'parent_sha256'):
            require(key in manifest and manifest[key] == reference[key], f'Cross-group mismatch in {key}: {arm}')
        require(manifest['conditioning_seed'] == CONDITIONING_SEED, 'Conditioning seed differs')
        require(manifest['unet_init'] == ('random' if arm in RANDOM_ARMS else 'parent'), f'Wrong initialization: {arm}')
        require(results[arm]['actual_draw_signature'] == results['P-L']['actual_draw_signature'],
                f'Actual paired draw/noise logs differ: {arm}')
    require(manifests['P-L']['initial_state_sha256'] == manifests['P-H']['initial_state_sha256'], 'Parent initial states differ')
    require(manifests['R-L']['initial_state_sha256'] == manifests['R-H']['initial_state_sha256'], 'Random initial states differ')
    require(manifests['R-H']['initial_state_sha256'] != manifests['P-L']['initial_state_sha256'], 'Random initial state equals parent')
    return shared


def parallel_training(args, runner, identity, arms, rates, attempt=None):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
        futures = {pool.submit(train_stage, args, runner, identity, arm, rates[arm], attempt): arm for arm in arms}
        try:
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            runner.wait_for_shutdown()
            raise
    return results


def precheck_cost(attempts):
    runs = []
    for number, outcomes in enumerate(attempts):
        for arm, result in outcomes.items():
            path = Path(result['output']) / 'optimization.jsonl'
            updates = 0
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        updates = max(updates, json.loads(line)['step'])
            runs.append({'attempt': number, 'arm': arm, 'status': result['status'], 'output': result['output'],
                         'logged_completed_optimizer_updates': updates,
                         'numeric_failure': result.get('failure'),
                         'failure_step_is_not_a_committed_update_count': bool(result.get('failure')),
                         'separate_from_formal_budget': True})
    return {'runs': runs, 'total_logged_completed_updates': sum(x['logged_completed_optimizer_updates'] for x in runs),
            'formal_updates_per_arm': STEPS,
            'counting_limit': 'A failing optimizer update may occur before its log is committed. Preserve numeric failure step/reason separately; logged counts are not exact attempted-update cost.'}


def existing_precheck_cost(args):
    attempts = []
    for attempt in (0, 1):
        results = {}
        for arm in ARMS:
            output = stage_output(args, arm, attempt)
            if not output.exists():
                continue
            failure = read_json(output / 'numeric-failure.json') if (output / 'numeric-failure.json').exists() else None
            results[arm] = {'output': str(output), 'status': 'numeric_failure' if failure else
                            ('complete' if (output / 'complete.json').exists() else 'incomplete'), 'failure': failure}
        attempts.append(results)
    return precheck_cost(attempts)


def ack_command(args):
    return [str(x) for x in (args.python, args.benchmark_ack, '--reader', args.reader,
                            '--output', args.run_root / 'official-history.json', '--device', 'cuda:0')]


def freeze_official_history(args, runner, identity):
    history = args.run_root / 'official-history.json'
    frozen = args.run_root / 'official-history-binding.json'
    if frozen.exists():
        require(read_json(frozen) == {'identity': identity, 'file': file_binding(history)}, 'Frozen official history changed')
    else:
        require(runner.run('official-history', args.eval_gpu, ack_command(args)) == 0, 'Official acknowledgment generation failed')
    artifact = read_json(history)
    benchmark = args.repo / 'reports/prefeval-official-alignment-20260923/data/benchmark-disclosures.jsonl.gz'
    with gzip.open(benchmark, 'rt', encoding='utf-8') as handle:
        rows = [r for line in handle if (r := json.loads(line))['form'] == 'explicit' and r['split'] == 'eval_topic']
    require(len(rows) == len({r['base_pair_id'] for r in rows}) == 180, 'Expected official180 disclosures')
    acknowledgments = artifact['acknowledgments']
    require(set(acknowledgments) == {r['base_pair_id'] for r in rows}, 'Official history is incomplete or contains extra IDs')
    expected_binding = {'reader': str(args.reader), 'benchmark_sha256': sha(benchmark),
                        'system_prompt': 'You are a helpful assistant.', 'max_new_tokens': 300, 'do_sample': False,
                        'input': 'official explicit preference only; no future query, options, or SFT acknowledgment'}
    require(artifact['binding'] == expected_binding, 'Official history generation protocol differs')
    for row in rows:
        value = acknowledgments[row['base_pair_id']]
        require(value['preference'] == row['input']['disclosure'][0]['content'], 'Official preference changed')
        require(isinstance(value['generated']['raw'], str) and value['generated']['raw'].strip(), 'Empty official acknowledgment')
    immutable_json(frozen, {'identity': identity, 'file': file_binding(history)})
    return file_binding(history)


def evaluate_job(args, runner, job):
    require(runner.run('eval-' + job['label'], job['gpu'], job['command']) == 0, f'Evaluation failed: {job["label"]}')
    output = Path(job['output'])
    done, binding = read_json(output / 'complete.json'), read_json(output / 'binding.json')
    expected_n = {'train': 730, 'dev': 90, 'official': 180}[job['split']]
    digest = sha(checkpoint(args, job['arm'], job['step']))
    require(done.get('status') == 'complete' and done.get('optimizer_step') == job['step'] and
            done.get('split') == job['split'] and done.get('checkpoint_sha256') == digest,
            f'Evaluation completion identity differs: {job["label"]}')
    require(done['binding_sha256'] == sha(output / 'binding.json') and
            done['summary_sha256'] == sha(output / 'summary.json'), f'Evaluation hashes changed: {job["label"]}')
    require(done['preferences'] == expected_n and done['primary_records'] == expected_n * 18 and
            done['position_records'] == expected_n * 25 and done['actual_pngs'] == expected_n * 2,
            f'Evaluation coverage differs: {job["label"]}')
    require(binding['checkpoint_sha256'] == digest and binding['optimizer_step'] == job['step'] and
            binding['split'] == job['split'], f'Evaluation binding differs: {job["label"]}')
    if job['split'] == 'official':
        require(binding['history_sha256'] == sha(args.run_root / 'official-history.json'), 'Official history changed during evaluation')
    return {'label': job['label'], 'complete': file_binding(output / 'complete.json')}


def evaluation_pool(args, runner, identity, jobs, *, phase):
    """A GPU belongs to one worker for the whole phase, including receipt validation.

    A resumed job still enters the evaluator, which revalidates its immutable binding
    and complete artifacts before returning. Only dispatch order changes between runs.
    """
    jobs = sorted(jobs, key=lambda job: {'train': 0, 'dev': 1, 'official': 2}[job['split']])
    require(len({job['label'] for job in jobs}) == len(jobs), 'Duplicate evaluation jobs')
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    cancelled, state_lock = threading.Event(), threading.Lock()
    results, active, first_failure = {}, {}, []

    def progress():
        save(args.run_root / 'status.json', {'phase': phase, 'identity': identity,
             'eval_gpus': list(args.eval_gpus), 'total': len(jobs), 'completed': len(results),
             'pending': pending.qsize(), 'active': dict(active)})

    def worker(gpu):
        try:
            while True:
                with state_lock:
                    if cancelled.is_set() or runner.stop.is_set():
                        return
                    try:
                        job = pending.get_nowait()
                    except queue.Empty:
                        return
                    active[gpu] = job['label']
                    progress()
                result = evaluate_job(args, runner, {**job, 'gpu': gpu})
                with state_lock:
                    results[job['label']] = result
                    del active[gpu]
                    progress()
        except BaseException as exc:
            with state_lock:
                if not first_failure:
                    first_failure.append(exc)
                cancelled.set()
            # Stop subprocess admission before another worker can launch another job.
            runner.terminate()
            raise

    progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.eval_gpus)) as pool:
        try:
            futures = [pool.submit(worker, gpu) for gpu in args.eval_gpus]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            cancelled.set()
            # Kill child process groups before waiting for the other worker threads.
            runner.wait_for_shutdown()
            if first_failure:
                raise first_failure[0]
            raise
    require(len(results) == len(jobs), 'Evaluation pool stopped before all jobs completed')
    return [results[job['label']] for job in jobs]


def execute(args, runner, identity):
    lock_path = args.run_root / 'lr-lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        require(args.resume and lock['identity'] == identity, 'Learning-rate lock identity differs')
        require(lock['high_learning_rate'] in (HIGH_LR, FALLBACK_LR), 'Invalid locked high LR')
        rates = learning_rates(lock['high_learning_rate'])
        require(lock['rates'] == rates, 'Learning-rate lock rates differ')
        shared = lock['shared_training_identity']
        successful = {}
        for arm in ARMS:
            attempt = 1 if arm in HIGH_ARMS and rates[arm] == FALLBACK_LR else 0
            output = stage_output(args, arm, attempt)
            require(lock['precheck_outputs'][arm] == str(output), 'Locked precheck path changed')
            successful[arm] = validate_stage_options(args, successful_stage(args, output, PRECHECK), arm, rates[arm])
        compare_training(successful, expected_shared=shared)
        require(lock['initial_states'] == {a: x['manifest']['initial_state_sha256'] for a, x in successful.items()},
                'Locked precheck initialization hashes changed')
        require(lock['precheck_draw_signature'] == successful['P-L']['actual_draw_signature'],
                'Locked precheck draw signature changed')
        if lock['high_learning_rate'] == FALLBACK_LR:
            require(any(numeric_failure(stage_output(args, arm, 0)) for arm in HIGH_ARMS),
                    'Fallback lock has no original high-LR numeric failure')
    else:
        save(args.run_root / 'status.json', {'phase': 'precheck', 'attempt': 0, 'identity': identity})
        rates = learning_rates(HIGH_LR)
        first = parallel_training(args, runner, identity, ARMS, rates, attempt=0)
        attempts, successful = [first], dict(first)
        save(args.run_root / 'precheck-cost.json', precheck_cost(attempts))
        if any(first[arm]['status'] == 'numeric_failure' for arm in HIGH_ARMS):
            rates.update({'P-H': FALLBACK_LR, 'R-H': FALLBACK_LR})
            save(args.run_root / 'status.json', {'phase': 'precheck', 'attempt': 1, 'identity': identity})
            second = parallel_training(args, runner, identity, HIGH_ARMS, rates, attempt=1)
            attempts.append(second)
            save(args.run_root / 'precheck-cost.json', precheck_cost(attempts))
            require(all(x['status'] == 'complete' for x in second.values()), 'High-LR numeric precheck failed after the single joint reduction')
            successful.update(second)
        require(all(x['status'] == 'complete' for x in successful.values()), 'Prechecks did not all pass')
        shared = compare_training(successful)
        lock = {'identity': identity, 'high_learning_rate': rates['P-H'], 'rates': rates,
                'shared_training_identity': shared, 'precheck_outputs': {a: x['output'] for a, x in successful.items()},
                'initial_states': {a: x['manifest']['initial_state_sha256'] for a, x in successful.items()},
                'precheck_draw_signature': successful['P-L']['actual_draw_signature'],
                'policy': 'locked before all four fresh formal runs; no later LR fallback'}
        save(lock_path, lock)
    save(args.run_root / 'status.json', {'phase': 'formal_training', 'identity': identity, 'rates': rates})
    formal = parallel_training(args, runner, identity, ARMS, rates)
    compare_training(formal, expected_shared=shared)
    for arm, result in formal.items():
        require(result['manifest']['learning_rate'] == rates[arm], f'Wrong actual learning rate: {arm}')
        require(result['manifest']['initial_state_sha256'] == lock['initial_states'][arm], f'Formal run was not fresh from the audited initialization: {arm}')
        require(result['first_128_draw_signature'] == lock['precheck_draw_signature'], f'Formal draw/noise prefix differs from precheck: {arm}')
    save(args.run_root / 'training-consistency.json', {'identity': identity, 'shared_training_identity': shared,
                                                      'groups': formal, 'rates': rates})
    evaluations = evaluation_pool(args, runner, identity, evaluation_jobs(args), phase='evaluation')
    require(len(evaluations) == 16, 'Incomplete registered evaluation coverage')
    save(args.run_root / 'status.json', {'phase': 'official_history', 'identity': identity})
    history = freeze_official_history(args, runner, identity)
    evaluations.extend(evaluation_pool(args, runner, identity, evaluation_jobs(args, official=True),
                                       phase='official_final_only'))
    require(len(evaluations) == 20, 'Incomplete fixed-endpoint evaluation coverage')
    result = {'identity': identity, 'rates': rates, 'formal_updates_per_arm': STEPS,
              'total_formal_updates': len(ARMS) * STEPS,
              'exposures_per_preference': 128, 'evaluation_count': 20, 'evaluations': evaluations,
              'precheck_cost': read_json(args.run_root / 'precheck-cost.json'), 'official180_run': True,
              'official_history': history, 'selection_policy': 'no official-result-based training changes'}
    save(args.run_root / 'results.json', result)
    save(args.run_root / 'complete.json', {'status': 'registered_training_and_fixed_evaluations_complete',
                                          'identity': identity, 'results_sha256': sha(args.run_root / 'results.json')})
    save(args.run_root / 'status.json', {'phase': 'complete', 'identity': identity})
    return result


def main(argv=None):
    args = configure(parser().parse_args(argv))
    if args.plan_only:
        print(json.dumps(plan(args), indent=2))
        return 0
    if args.preflight_only:
        print(json.dumps(preflight(args), indent=2))
        return 0
    with controller_lock(args.run_root):
        runner = Runner(args)
        registered = False
        old_handlers = {}
        def stop(signum, _frame):
            runner.terminate()
            raise ProtocolError(f'Controller received signal {signum}; child process groups terminated')
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop)
        try:
            binding = preflight(args)
            identity = register(args, binding)
            args._inputs = binding
            registered = True
            refuse_live_previous_children(args.run_root)
            save(args.run_root / 'controller.json', {'pid': os.getpid(), 'host': socket.gethostname(),
                                                     'started': time.time(), 'identity': identity})
            execute(args, runner, identity)
        except BaseException as exc:
            runner.terminate()
            if registered:
                save(args.run_root / 'failure.json', {'error': repr(exc), 'time': time.time()})
                save(args.run_root / 'status.json', {'phase': 'failed', 'error': repr(exc)})
                save(args.run_root / 'precheck-cost.json', existing_precheck_cost(args))
            raise
        finally:
            runner.wait_for_shutdown()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
