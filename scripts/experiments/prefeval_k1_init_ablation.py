"""Opt-in initialization controls and numerical diagnostics for the K1 Writer.

No Reader loss, teacher optimization, or alternate flow objective lives here.
"""
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from vision_memory.repro import canonical_object_sha256, named_tensors_manifest


class NumericFailure(RuntimeError):
    """Only explicit nonfinite arithmetic is eligible for the LR fallback."""
    def __init__(self, reason, *, step=0, details=None):
        super().__init__(reason)
        self.reason, self.step, self.details = reason, int(step), details or {}


def write_numeric_failure(output, failure):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = {'schema': 'prefeval-k1-numeric-failure/v1', 'reason': failure.reason,
              'step': failure.step, 'details': failure.details}
    temporary = output / 'numeric-failure.json.tmp'
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output / 'numeric-failure.json')
    print('K1_NUMERIC_FAILURE ' + json.dumps(record, allow_nan=False), file=sys.stderr, flush=True)


def enabled(args):
    return (args.unet_init != 'parent' or args.learning_rate != 5e-5
            or args.train_seed != 20260924 or args.conditioning_seed is not None
            or args.fresh_start or args.audit_updates or args.fm_probe_interval > 0
            or args.expected_init_audit is not None)


def validate_args(args):
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be positive and finite')
    if args.fm_probe_interval < 0 or args.fm_probe_count < 1:
        raise ValueError('Invalid fixed FM probe interval/count')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if args.unet_init == 'random' and args.mode != 'train':
        raise ValueError('Random initialization is allowed only for train; rollout loads its checkpoint')
    if (args.fresh_start or args.audit_updates or args.fm_probe_interval) and args.mode != 'train':
        raise ValueError('Training diagnostics cannot be requested for rollout')
    if args.fm_probe_interval and args.stage != 'write':
        raise ValueError('The fixed train FM probe currently supports only the write stage')
    if args.fresh_start:
        ensure_fresh_output(args.output)


def ensure_fresh_output(output):
    path = Path(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('fresh-start requires an absent or empty output directory; never resumes prior updates')


@contextmanager
def isolated_rng(seed):
    """Seed and restore Python, NumPy, CPU Torch and every visible CUDA RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def seeded_factory(factory, seed):
    if seed is None:
        return factory()
    with isolated_rng(seed):
        return factory()


def tensor_schema(module):
    return {'parameters': {n: [list(p.shape), str(p.dtype)] for n, p in module.named_parameters()},
            'buffers': {n: [list(p.shape), str(p.dtype)] for n, p in module.named_buffers()}}


def model_hash(module):
    return named_tensors_manifest([
        *((f'parameter:{n}', p) for n, p in module.named_parameters()),
        *((f'buffer:{n}', p) for n, p in module.named_buffers()),
    ])['bundle_sha256']


def frozen_hash(module):
    """Parameters and persistent buffers; exclude transient inference caches only."""
    return named_tensors_manifest(module.state_dict())['bundle_sha256']


def initialize_unet(pipe, args, load_parent):
    """Use the model class's constructor defaults, never reset pretrained tensors in-place."""
    if args.unet_init == 'parent':
        pipe.unet.requires_grad_(True)
        load_parent(args.checkpoint, trainable_module=pipe.unet)
    else:
        original = pipe.unet
        expected = tensor_schema(original)
        expected_buffers = named_tensors_manifest(original.named_buffers())['bundle_sha256']
        config = copy.deepcopy(dict(original.config))
        cls = type(original)
        # Release the pretrained GPU allocation before transferring the replacement.
        original.to('cpu')
        with isolated_rng(args.init_seed):
            replacement = cls.from_config(config)
        replacement = replacement.to(dtype=torch.float32)
        if tensor_schema(replacement) != expected:
            raise RuntimeError('from_config changed the U-Net parameter/buffer schema')
        if named_tensors_manifest(replacement.named_buffers())['bundle_sha256'] != expected_buffers:
            raise RuntimeError('Random constructor changed buffers; parameter-only checkpoints cannot reproduce these buffers')
        pipe.unet = replacement
        del original
        gc.collect()
        pipe.unet.to(args.device)
    # eval mode is intentional: the legacy full-U-Net Writer uses no dropout.
    pipe.unet.eval().requires_grad_(True)


def initialization_manifest(pipe, args):
    finite_tensors(pipe.unet.named_parameters(), reason='nonfinite_initial_parameter', step=0)
    # Underscore-prefixed Diffusers fields identify checkpoint provenance, not architecture.
    config = {key: value for key, value in dict(pipe.unet.config).items() if not key.startswith('_')}
    # Diffusers configs contain JSON-compatible configuration values.
    config = json.loads(json.dumps(config, sort_keys=True, default=str))
    schema = tensor_schema(pipe.unet)
    class_name = type(pipe.unet).__module__ + '.' + type(pipe.unet).__qualname__
    frozen = {name: frozen_hash(getattr(pipe, name)) for name in ('vae', 'text_encoder')}
    assert_frozen(pipe)
    return {'initial_state_sha256': model_hash(pipe.unet),
            'architecture_sha256': canonical_object_sha256({'class': class_name, 'config': config, 'schema': schema}),
            'parameter_schema_sha256': canonical_object_sha256(schema),
            'unet_class': class_name, 'config_sha256': canonical_object_sha256(config),
            'frozen_module_hashes': frozen,
            'frozen_hash_scope': 'state_dict_parameters_and_persistent_buffers',
            'unet_parameter_count': sum(p.numel() for p in pipe.unet.parameters()),
            'reader_in_training': False}


def assert_frozen(pipe):
    for name in ('vae', 'text_encoder'):
        module = getattr(pipe, name)
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError(f'Frozen {name} acquired training state or gradients')


def verify_frozen(pipe, expected):
    assert_frozen(pipe)
    current = {name: frozen_hash(getattr(pipe, name)) for name in expected}
    if current != expected:
        raise RuntimeError('Frozen VAE/text encoder tensors changed during training')
    return current


def runtime_manifest(device):
    versions = {}
    for package in ('torch', 'diffusers', 'transformers', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda_device = torch.device(device)
    return {'versions': versions, 'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device_type': cuda_device.type,
            'device_name': torch.cuda.get_device_name(cuda_device) if cuda_device.type == 'cuda' else 'cpu',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'unet_mode': 'eval_with_gradients'}


def conditions_manifest(cache):
    # Write-stage conditions live in a dict; all three tensors are content-hashed.
    return {json.dumps(key, separators=(',', ':')): named_tensors_manifest(value)['bundle_sha256']
            for key, value in sorted(cache.items(), key=lambda item: json.dumps(item[0]))}


def shared_identity(rows, target_hashes, condition_hashes, args, variants_sha256):
    row_hash = canonical_object_sha256([(r['base_pair_id'], r['history']) for r in rows])
    schedule = {'version': 'k1-balanced-draw/v1', 'seed': args.train_seed,
                'ids_in_order': [r['base_pair_id'] for r in rows], 'stage': args.stage,
                'effective_batch': 4, 'variants_sha256': variants_sha256,
                'variants': 'cycle%2' if variants_sha256 else '0',
                'order': 'randperm/stable_seed(seed,order,cycle)',
                'sigma': 'CPU rand/stable_seed(seed,sigma,draw)',
                'noise': 'device randn/stable_seed(seed,noise,draw)'}
    return {'conditions_sha256': canonical_object_sha256(condition_hashes),
            'targets_sha256': canonical_object_sha256(target_hashes),
            'training_rows_sha256': row_hash, 'schedule_sha256': canonical_object_sha256(schedule),
            'train_seed': args.train_seed, 'effective_batch': 4}


def finite_tensors(named, *, reason, step):
    """One host synchronization per device, not per model parameter."""
    groups = {}
    for name, tensor in named:
        if isinstance(tensor, torch.Tensor):
            groups.setdefault(tensor.device, []).append((name, torch.isfinite(tensor).all()))
    for items in groups.values():
        checks = torch.stack([value for _, value in items]).detach().cpu().tolist()
        bad = [name for (name, _), good in zip(items, checks) if not good]
        if bad:
            raise NumericFailure(reason, step=step, details={'nonfinite_tensors': bad[:20]})


class UpdateAudit:
    """True before/after updates at eight fixed coordinates in every parameter tensor.

    This is explicitly a sampled-coordinate measurement, not a full-model update norm.
    All parameters and Adam states are nevertheless checked for finiteness each step.
    """
    def __init__(self, module):
        self.module = module
        self.samples = []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and parameter.numel():
                indices = torch.linspace(0, parameter.numel()-1, min(8, parameter.numel()),
                                         device=parameter.device).round().long().unique()
                self.samples.append((name, parameter, indices))

    def capture(self):
        return torch.cat([p.detach().reshape(-1)[indices] for _, p, indices in self.samples]).clone()

    def after_step(self, before, optimizer, step):
        finite_tensors(self.module.named_parameters(), reason='nonfinite_parameter_after_step', step=step)
        state_tensors = []
        for name, parameter in self.module.named_parameters():
            for key, tensor in optimizer.state.get(parameter, {}).items():
                if isinstance(tensor, torch.Tensor):
                    state_tensors.append((name + ':' + key, tensor))
        finite_tensors(state_tensors, reason='nonfinite_optimizer_state', step=step)
        after = self.capture()
        delta = (after.double() - before.double())
        update = float(torch.linalg.vector_norm(delta))
        previous = float(torch.linalg.vector_norm(before.double()))
        return {'scope': 'eight_fixed_coordinates_per_trainable_tensor_not_full_model_norm',
                'sampled_parameter_count': len(self.samples), 'sampled_coordinate_count': before.numel(),
                'sampled_update_l2': update, 'sampled_parameter_l2_before': previous,
                'sampled_relative_update': update / max(previous, 1e-30),
                'sampled_changed_coordinates': int(torch.count_nonzero(delta))}


def clip_gradients(parameters, step):
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    except RuntimeError as error:
        if 'non-finite' in str(error).lower() or 'nonfinite' in str(error).lower():
            raise NumericFailure('nonfinite_gradient_norm', step=step) from error
        raise


def probe_ids(rows, count, seed):
    return sorted([r['base_pair_id'] for r in rows],
                  key=lambda pid: hashlib.sha256(f'{seed}:train-fm-probe:{pid}'.encode()).digest())[:count]


def archive_probe_tail(path, first):
    """Probe steps are sparse and include step zero, unlike the optimization log."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding='utf-8').splitlines()
    # Before the first durable optimizer checkpoint, regenerate the entire diagnostic.
    valid = [line for line in lines if first > 0 and json.loads(line)['step'] <= first]
    tail = [line for line in lines if first == 0 or json.loads(line)['step'] > first]
    steps = [json.loads(line)['step'] for line in valid]
    if steps != sorted(set(steps)):
        raise RuntimeError('Probe log contains duplicate or unordered committed steps')
    if tail:
        archive = path.with_name(f'{path.stem}-uncommitted-{time.time_ns()}.jsonl')
        archive.write_text('\n'.join(tail) + '\n', encoding='utf-8')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(('\n'.join(valid) + '\n') if valid else '', encoding='utf-8')
        temporary.replace(path)


def expected_init_audit(manifest, path):
    """Gate a fresh formal run against its completed preflight's initialization inputs."""
    path = Path(path)
    contents = path.read_bytes()
    reference = json.loads(contents)
    fields = ('initial_state_sha256', 'architecture_sha256', 'frozen_module_hashes', 'runtime',
              'shared_training_identity', 'unet_init', 'learning_rate', 'train_seed',
              'init_seed', 'conditioning_seed')
    missing = [key for key in fields if key not in reference or key not in manifest]
    mismatches = [key for key in fields if key not in missing and manifest[key] != reference[key]]
    if missing or mismatches:
        raise RuntimeError(f'Expected initialization audit mismatch: missing={missing}, changed={mismatches}')
    return hashlib.sha256(contents).hexdigest()
