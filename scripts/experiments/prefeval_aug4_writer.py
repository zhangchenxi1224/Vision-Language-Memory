"""Registered W0-W3, neutral-ack, official-pretrained/random full-FM Writer.

This independent entry point never loads a historical task-trained checkpoint.
Heavy DreamLite imports are delayed so the schedule and recovery can be CPU tested.
"""
# ruff: noqa: E402 -- establish repository imports and deterministic CUDA env first.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from scripts.experiments import prefeval_k1_init_ablation as control
from scripts.experiments.prefeval_k1_data import sha
from scripts.experiments.prefeval_aug4_assets import verify_asset_seal
from scripts.experiments.prefeval_k1_refresh_bank import archive_uncommitted_tail
from vision_memory.repro import canonical_object_sha256, named_tensors_manifest, configure_strict_cuda_determinism
from vision_memory.training.checkpoint import save_training_checkpoint, load_training_checkpoint
from vision_memory.training.latent_bank_unet import (
    OFFICIAL_REFERENCE_COMMIT, official_flow_bridge, predict_velocity, stable_seed,
)

OFFICIAL_UNET_SHA256 = 'f983bd1710344cb44f7d12d05c0ca961984b48391dcbb095ad710609cbde3246'
SNAPSHOTS = (5840, 11680, 23360, 46720, 93440)
VARIANTS = ('W0', 'W1', 'W2', 'W3')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def load_data(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data['train']
    if len(rows) != 730 or len({r['base_pair_id'] for r in rows}) != 730:
        raise ValueError('The registered training split must contain 730 unique preferences')
    for row in rows:
        pid = row['base_pair_id']
        if any(c in pid for c in ('/', '\\', '..')):
            raise ValueError('Unsafe preference id')
        if set(row['writer_user_forms']) != set(VARIANTS):
            raise ValueError(f'{pid}: exactly W0-W3 are required')
        if set(row['teacher_question_forms']) != {'T1', 'T2', 'T3'}:
            raise ValueError(f'{pid}: exactly T1-T3 are required')
        if not all(isinstance(v, str) and v.strip() for v in row['writer_user_forms'].values()):
            raise ValueError('Empty writer wording')
    return rows


def writer_text(row, variant):
    # Question/options/answers deliberately never enter this path.
    return 'user: ' + row['writer_user_forms'][variant] + '\nassistant: Understood.'


class BalancedSchedule:
    """Every n consecutive draws visit each preference once; every four cycles visit every W once."""
    def __init__(self, rows, seed):
        self.rows, self.seed, self.cycle, self.order = rows, seed, None, None

    def draw(self, index):
        cycle, offset = divmod(index, len(self.rows))
        if cycle != self.cycle:
            self.order = torch.randperm(len(self.rows), generator=torch.Generator().manual_seed(
                stable_seed(self.seed, 'order', cycle))).tolist()
            self.cycle = cycle
        row = self.rows[self.order[offset]]
        return row, VARIANTS[cycle % 4]

    def exposure(self, draws):
        cycles, remainder = divmod(draws, len(self.rows))
        base, extra = divmod(cycles, 4)
        counts = {r['base_pair_id']: {v: base + int(i < extra) for i, v in enumerate(VARIANTS)}
                  for r in self.rows}
        for index in range(cycles * len(self.rows), cycles * len(self.rows) + remainder):
            row, variant = self.draw(index)
            counts[row['base_pair_id']][variant] += 1
        return counts


def official_files(args):
    file = args.base / 'unet/diffusion_pytorch_model.safetensors'
    actual = sha(file)
    if actual != OFFICIAL_UNET_SHA256:
        raise RuntimeError(f'Official Base U-Net SHA256 mismatch: {actual}')
    return {'official_unet_sha256': actual, 'official_unet_config_sha256': sha(args.base / 'unet/config.json'),
            'official_commit': OFFICIAL_REFERENCE_COMMIT}


def load_pipe(args):
    identity = official_files(args)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.official_source, text=True).strip()
    if commit != OFFICIAL_REFERENCE_COMMIT:
        raise RuntimeError('Unexpected official source commit')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=args.official_source, text=True).strip()
    if dirty:
        raise RuntimeError('Official source contains tracked changes')
    sys.path.insert(0, str(args.official_source))
    from dreamlite import DreamLitePipelineLoRA
    pipe = DreamLitePipelineLoRA.from_pretrained(args.base, local_files_only=True, torch_dtype=torch.float32)
    pipe.to(args.device)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    initialize_unet(pipe, args)
    pipe.set_progress_bar_config(disable=True)
    return pipe, identity


def initialize_unet(pipe, args):
    if args.init == 'random' and args.mode != 'cache-conditions':
        # Existing audited helper constructs the same class from config under an isolated seed.
        options = SimpleNamespace(unet_init='random', init_seed=args.init_seed, device=args.device)
        control.initialize_unet(pipe, options, lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('Historical checkpoint loading is forbidden')))
    elif args.init != 'pretrained' and args.mode != 'cache-conditions':
        raise ValueError('Unsupported initialization')
    pipe.unet.eval().requires_grad_(True)


def condition_binding(args, pipe, official):
    return {'schema': 'prefeval-aug4-conditions/v1', 'data_sha256': sha(args.data), **official,
            'asset_seal_sha256': getattr(args, 'asset_seal_sha256', None),
            'conditioning_seed': args.conditioning_seed, 'dtype': 'float32', 'ack': 'Understood.',
            'frozen_module_hashes': {name: control.frozen_hash(getattr(pipe, name))
                                     for name in ('vae', 'text_encoder')},
            'conditioning_code_sha256': sha(ROOT / 'src/vision_memory/dreamlite/conditioning.py'),
            'writer_code_sha256': sha(Path(__file__))}


def condition_path(root, pid, variant):
    return Path(root) / pid.replace(':', '_') / f'{variant}.pt'


@torch.no_grad()
def encode_condition(pipe, text, device):
    from PIL import Image
    from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition
    image = Image.new('RGB', (1024, 1024), (128, 128, 128))
    source = pipe.prepare_image_latents(pipe.image_processor.preprocess(image), dtype=torch.float32, device=device)
    encoded = encode_native_base_edit_condition(pipe, image, text, device=device, dtype=torch.float32)
    return {'source': source.cpu(), 'embeds': encoded.prompt_embeds.cpu(), 'mask': encoded.attention_mask.cpu()}


def cache_conditions(args, pipe, rows, official):
    root = args.condition_root or args.output / 'conditions'
    root.mkdir(parents=True, exist_ok=True)
    binding = condition_binding(args, pipe, official)
    binding_sha = canonical_object_sha256(binding)
    jobs = [(row, variant) for row in rows for variant in VARIANTS]
    hashes = {}
    for index in range(args.shard_index, len(jobs), args.shard_count):
        row, variant = jobs[index]
        pid = row['base_pair_id']
        path = condition_path(root, pid, variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=True)
            if record['binding_sha256'] != binding_sha or record['key'] != [pid, variant]:
                raise RuntimeError('Stale condition cache')
        else:
            seed = stable_seed(args.conditioning_seed, f'condition:{pid}:{variant}', 0)
            condition = control.seeded_factory(lambda: encode_condition(pipe, writer_text(row, variant), args.device), seed)
            record = {'key': [pid, variant], 'binding_sha256': binding_sha,
                      'text_sha256': hashlib.sha256(writer_text(row, variant).encode()).hexdigest(),
                      'condition': condition,
                      'tensor_sha256': named_tensors_manifest(condition)['bundle_sha256']}
            temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
            torch.save(record, temporary)
            os.replace(temporary, path)
        hashes[f'{pid}/{variant}'] = sha(path)
        print(json.dumps({'cached_index': index, 'total': len(jobs), 'shard': args.shard_index}), flush=True)
    save_json(root / f'shard-{args.shard_index}.json', {'binding': binding, 'binding_sha256': binding_sha,
              'shard_index': args.shard_index, 'shard_count': args.shard_count, 'files': hashes})


def load_conditions(args, pipe, rows, official):
    root = args.condition_root or args.output / 'conditions'
    if args.condition_root is None:
        cache_conditions(args, pipe, rows, official)
    binding = condition_binding(args, pipe, official)
    binding_sha = canonical_object_sha256(binding)
    cache, hashes = {}, {}
    for row in rows:
        for variant in VARIANTS:
            pid = row['base_pair_id']
            record = torch.load(condition_path(root, pid, variant), map_location='cpu', weights_only=True)
            if record['binding_sha256'] != binding_sha or record['key'] != [pid, variant]:
                raise RuntimeError('Condition is not bound to the registered dataset/model')
            if record['text_sha256'] != hashlib.sha256(writer_text(row, variant).encode()).hexdigest():
                raise RuntimeError('Writer cache text mismatch')
            tensor_hash = named_tensors_manifest(record['condition'])['bundle_sha256']
            if tensor_hash != record['tensor_sha256']:
                raise RuntimeError('Condition tensor hash mismatch')
            cache[pid, variant] = record['condition']
            hashes[f'{pid}/{variant}'] = tensor_hash
    return cache, hashes


def load_targets(args, rows):
    targets, hashes = {}, {}
    data_sha = sha(args.data)
    for row in rows:
        pid = row['base_pair_id']
        root = args.teacher_root / pid.replace(':', '_')
        done = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
        if done['binding'].get('asset_seal_sha256') != getattr(args, 'asset_seal_sha256', None):
            raise RuntimeError('Teacher and Writer model asset seals differ')
        if (done['step'] != 288 or done['binding']['data_sha256'] != data_sha
                or done['binding'].get('schema') != 'prefeval-aug4-teacher/v1'
                or done['binding'].get('technical_smoke') or not done.get('numeric_finite')):
            raise RuntimeError('Teacher is not the registered 288-step augmented target')
        target_hash = sha(root / 'latent.pt')
        if target_hash != done['latent_sha256'] or sha(root / 'memory.png') != done['png_sha256']:
            raise RuntimeError('Teacher latent or PNG hash mismatch')
        target = torch.load(root / 'latent.pt', map_location='cpu', weights_only=True)
        if not isinstance(target, torch.Tensor) or tuple(target.shape) != (1, 4, 128, 128):
            raise RuntimeError('Unexpected teacher latent shape')
        if target.dtype != torch.float32 or not torch.isfinite(target).all():
            raise RuntimeError('Teacher latent must be finite float32')
        targets[pid], hashes[pid] = target, target_hash
    return targets, hashes


def expected_audit(manifest, path):
    reference = json.loads(Path(path).read_text(encoding='utf-8'))
    done_path = Path(path).parent / 'complete.json'
    done = json.loads(done_path.read_text(encoding='utf-8'))
    if reference['steps'] != 128 or done['steps'] != 128 or not done['frozen_verified']:
        raise RuntimeError('Formal run requires a completed independent 128-step precheck')
    fields = ('asset_seal_sha256', 'initial_state_sha256', 'architecture_sha256', 'frozen_module_hashes',
              'shared_training_identity', 'init_mode', 'learning_rate', 'train_seed',
              'init_seed', 'conditioning_seed', 'official_unet_sha256', 'official_unet_config_sha256')
    changed = [field for field in fields if manifest.get(field) != reference.get(field)]
    if changed:
        raise RuntimeError(f'Precheck/formal initialization mismatch: {changed}')
    return sha(path)


def save_snapshot(output, module, step, manifest, schedule):
    """Only the ready marker publishes a snapshot; published files are never overwritten."""
    output = Path(output)
    stem = output / f'checkpoint-step-{step:06d}'
    inference = stem.with_suffix('.pt')
    full = stem.with_suffix('.full.pt')
    ready = stem.with_suffix('.ready.json')
    if ready.exists():
        record = json.loads(ready.read_text(encoding='utf-8'))
        if record['checkpoint_sha256'] != sha(inference) or record['full_checkpoint_sha256'] != sha(full):
            raise RuntimeError('Published immutable snapshot was changed')
        if record['manifest_sha256'] != canonical_object_sha256(manifest):
            raise RuntimeError('Published snapshot manifest mismatch')
        return
    # Resume is saved first at this exact step, and atomically replaced on subsequent saves.
    if full.exists():
        full.unlink()  # unpublished crash tail only
    try:
        os.link(output / 'resume.pt', full)
    except OSError:
        shutil.copyfile(output / 'resume.pt', full)
    payload = {'schema_version': 1, 'trainable_state': {name: p.detach().cpu()
               for name, p in module.named_parameters()}, 'optimizer_step': step, 'manifest': manifest}
    temporary = inference.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, inference)
    counts = schedule.exposure(step * 4)
    save_json(stem.with_suffix('.exposures.json'), {'step': step, 'draws': step * 4, 'counts': counts})
    save_json(ready, {'step': step, 'checkpoint_sha256': sha(inference), 'full_checkpoint_sha256': sha(full),
              'manifest_sha256': canonical_object_sha256(manifest), 'exposures_sha256':
              sha(stem.with_suffix('.exposures.json'))})


def train_loop(args, pipe, targets, cache, rows, manifest, predictor):
    """Kept independent of heavyweight loaders for exact CPU recovery tests."""
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.learning_rate,
                                 betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
    checkpoint = args.output / 'resume.pt'
    first = 0
    saved = None
    if args.resume and checkpoint.exists():
        saved = load_training_checkpoint(checkpoint, trainable_module=pipe.unet,
                                         optimizer=optimizer, expected_manifest=manifest)
        first = saved['optimizer_step']
        if saved['episode_cursor'] != first * 4:
            raise RuntimeError('Resume draw cursor mismatch')
    elif checkpoint.exists():
        raise RuntimeError('Existing optimizer state requires explicit --resume')
    schedule = BalancedSchedule(rows, args.train_seed)
    exposures = schedule.exposure(first * 4)
    if saved is not None and saved['trainer_state']['exposure_counts'] != exposures:
        raise RuntimeError('Recovered exposure counts do not match the balanced draw cursor')
    archive_uncommitted_tail(args.output / 'optimization.jsonl', first)
    control.archive_probe_tail(args.output / 'fm-probe.jsonl', first)
    if first in args.snapshot_steps:
        save_snapshot(args.output, pipe.unet, first, manifest, schedule)
    audit = control.UpdateAudit(pipe.unet)
    probe_ids = control.probe_ids(rows, args.fm_probe_count, args.train_seed)
    started = time.monotonic()

    @torch.no_grad()
    def probe(step):
        measurements = []
        with control.isolated_rng(args.train_seed):
            for index, pid in enumerate(probe_ids):
                variant = VARIANTS[index % 4]
                c, target = cache[pid, variant], targets[pid].to(args.device)
                sigma = float(torch.rand((), generator=torch.Generator().manual_seed(
                    stable_seed(args.train_seed, 'fixed-probe-sigma', index))))
                noise = torch.randn(target.shape, dtype=target.dtype, device=args.device,
                    generator=torch.Generator(device=args.device).manual_seed(
                        stable_seed(args.train_seed, 'fixed-probe-noise', index)))
                state, velocity = official_flow_bridge(noise, target, sigma)
                predicted = predict_velocity(predictor, state, c['source'].to(args.device), sigma,
                    c['embeds'].to(args.device), c['mask'].to(args.device), integer_timestep=True)
                loss = (predicted.float() - velocity.float()).square().mean()
                if not torch.isfinite(loss):
                    raise control.NumericFailure('nonfinite_fixed_train_probe', step=step)
                measurements.append({'pair_id': pid, 'variant': variant, 'mse': float(loss)})
        append(args.output / 'fm-probe.jsonl', {'step': step, 'scope': 'fixed_training_inputs_no_update',
               'mse': sum(r['mse'] for r in measurements) / len(measurements), 'draws': measurements})

    if first == 0 and args.fm_probe_interval:
        probe(0)
    for step in range(first, args.steps):
        optimizer.zero_grad(set_to_none=True)
        draws = []
        for micro in range(4):
            draw = step * 4 + micro
            row, variant = schedule.draw(draw)
            pid = row['base_pair_id']
            condition, target = cache[pid, variant], targets[pid].to(args.device)
            sigma = float(torch.rand((), generator=torch.Generator().manual_seed(stable_seed(args.train_seed, 'sigma', draw))))
            noise_seed = stable_seed(args.train_seed, 'noise', draw)
            noise = torch.randn(target.shape, dtype=target.dtype, device=args.device,
                                generator=torch.Generator(device=args.device).manual_seed(noise_seed))
            state, velocity = official_flow_bridge(noise, target, sigma)
            prediction = predict_velocity(predictor, state, condition['source'].to(args.device), sigma,
                condition['embeds'].to(args.device), condition['mask'].to(args.device), integer_timestep=True)
            loss = (prediction.float() - velocity.float()).square().mean()
            if not torch.isfinite(loss):
                raise control.NumericFailure('nonfinite_fm_loss', step=step + 1,
                                             details={'pair_id': pid, 'draw': draw})
            (loss / 4).backward()
            exposures[pid][variant] += 1
            draws.append({'draw': draw, 'pair_id': pid, 'variant': variant,
                          'sigma': sigma, 'noise_seed': noise_seed, 'mse': float(loss.detach())})
        norm = control.clip_gradients(pipe.unet.parameters(), step + 1)
        measure = args.mode == 'precheck' or step < 8 or (step + 1) % args.audit_interval == 0
        before = audit.capture() if measure else None
        optimizer.step()
        record = {'step': step + 1, 'draws': draws, 'grad_norm': float(norm),
                  'seconds': time.monotonic() - started}
        if measure:
            record['update_audit'] = audit.after_step(before, optimizer, step + 1)
        control.assert_frozen(pipe)
        append(args.output / 'optimization.jsonl', record)
        if args.fm_probe_interval and ((step + 1) % args.fm_probe_interval == 0 or step + 1 == args.steps):
            probe(step + 1)
        if (step + 1) % 32 == 0:
            print(json.dumps({'step': step + 1, 'mse': sum(d['mse'] for d in draws) / 4,
                              'seconds': time.monotonic() - started}), flush=True)
        if (step + 1) % args.save_interval == 0 or step + 1 in args.snapshot_steps or step + 1 == args.steps:
            if exposures != schedule.exposure((step + 1) * 4):
                raise RuntimeError('Actual preference/wording exposure counts drifted from the registered schedule')
            save_training_checkpoint(checkpoint, trainable_module=pipe.unet, optimizer=optimizer,
                epoch=0, episode_cursor=(step + 1) * 4, optimizer_step=step + 1, manifest=manifest,
                trainer_state={'schedule': 'aug4-balanced/v1', 'draw_cursor': (step + 1) * 4,
                               'exposure_counts': exposures})
        if step + 1 in args.snapshot_steps:
            save_snapshot(args.output, pipe.unet, step + 1, manifest, schedule)
    return schedule


def train(args, pipe, rows, official):
    targets, target_hashes = load_targets(args, rows)
    cache, condition_hashes = load_conditions(args, pipe, rows, official)
    initial = control.initialization_manifest(pipe, args)
    manifest = {'schema': 'prefeval-aug4-writer/v1', 'data_sha256': sha(args.data), **official, **initial,
                'asset_seal_sha256': getattr(args, 'asset_seal_sha256', None),
                'init_mode': args.init, 'learning_rate': args.learning_rate, 'steps': args.steps,
                'snapshot_steps': args.snapshot_steps, 'train_seed': args.train_seed, 'init_seed': args.init_seed,
                'conditioning_seed': args.conditioning_seed, 'effective_batch': 4, 'dtype': 'float32',
                'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-4, 'clip': 1.0},
                'ack': 'Understood.', 'writer_variants': list(VARIANTS), 'objective': 'official_full_FM',
                'runtime': control.runtime_manifest(args.device), 'implementation_sha256': sha(Path(__file__)),
                'save_interval': args.save_interval, 'audit_interval': args.audit_interval,
                'fm_probe_interval': args.fm_probe_interval, 'fm_probe_count': args.fm_probe_count,
                'dependency_sha256': {name: sha(ROOT / name) for name in (
                    'scripts/experiments/prefeval_k1_init_ablation.py',
                    'src/vision_memory/training/latent_bank_unet.py',
                    'src/vision_memory/training/checkpoint.py',
                    'src/vision_memory/dreamlite/conditioning.py')},
                'shared_training_identity': {'data_sha256': sha(args.data),
                    'targets_sha256': canonical_object_sha256(target_hashes),
                    'conditions_sha256': canonical_object_sha256(condition_hashes),
                    'schedule_sha256': canonical_object_sha256({'name': 'aug4-balanced/v1', 'seed': args.train_seed,
                        'ids': [r['base_pair_id'] for r in rows], 'variants': list(VARIANTS), 'batch': 4})}}
    if args.expected_init_audit:
        manifest['expected_init_audit_sha256'] = expected_audit(manifest, args.expected_init_audit)
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Output is bound to different run inputs')
    save_json(path, manifest)
    save_json(args.output / 'condition-hashes.json', condition_hashes)
    if not (args.output / 'initial-state.json').exists():
        save_json(args.output / 'initial-state.json', {**initial, 'fresh_optimizer': True,
                  'optimizer_state_entries': 0, 'optimizer_step': 0, 'learning_rate': args.learning_rate})
    from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler
    predictor = DifferentiableDreamLiteMobileSampler.from_pipeline(pipe, checkpoint_unet=False)
    schedule = train_loop(args, pipe, targets, cache, rows, manifest, predictor)
    control.finite_tensors(pipe.unet.named_parameters(), reason='nonfinite_final_parameter', step=args.steps)
    control.verify_frozen(pipe, manifest['frozen_module_hashes'])
    save_snapshot(args.output, pipe.unet, args.steps, manifest, schedule)
    final = args.output / 'checkpoint-final.pt'
    source = args.output / f'checkpoint-step-{args.steps:06d}.pt'
    if final.exists():
        if sha(final) != sha(source):
            raise RuntimeError('Final checkpoint changed')
    else:
        try:
            os.link(source, final)
        except OSError:
            shutil.copyfile(source, final)
    save_json(args.output / 'complete.json', {'steps': args.steps, 'checkpoint_sha256': sha(final),
              'frozen_verified': True, 'initial_state_sha256': manifest['initial_state_sha256'],
              'final_state_sha256': control.model_hash(pipe.unet),
              'shared_training_identity': manifest['shared_training_identity']})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=['train', 'precheck', 'audit-init', 'cache-conditions'])
    for name in ('data', 'base', 'official-source', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--teacher-root', type=Path)
    result.add_argument('--condition-root', type=Path)
    result.add_argument('--init', choices=['pretrained', 'random'], default='pretrained')
    result.add_argument('--learning-rate', type=float, default=5e-5)
    result.add_argument('--steps', type=int, default=93440)
    result.add_argument('--snapshot-steps', nargs='+', type=int, default=list(SNAPSHOTS))
    result.add_argument('--train-seed', type=int, default=20261009)
    result.add_argument('--init-seed', type=int, default=20261009)
    result.add_argument('--conditioning-seed', type=int, default=0)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--expected-init-audit', type=Path)
    result.add_argument('--save-interval', type=int, default=128)
    result.add_argument('--audit-interval', type=int, default=128)
    result.add_argument('--fm-probe-interval', type=int, default=512)
    result.add_argument('--fm-probe-count', type=int, default=32)
    result.add_argument('--shard-index', type=int, default=0)
    result.add_argument('--shard-count', type=int, default=1)
    return result


def main(args):
    args.asset_seal_sha256 = verify_asset_seal(args)
    if args.mode == 'precheck':
        args.steps, args.snapshot_steps = 128, [128]
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Invalid learning rate')
    if args.steps < 1 or args.fm_probe_interval < 0 or args.fm_probe_count < 1:
        raise ValueError('Invalid training/probe budget')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid cache shard')
    if args.mode in {'train', 'precheck'}:
        if args.teacher_root is None or args.save_interval < 1 or args.audit_interval < 1:
            raise ValueError('Teacher root and positive intervals required')
        if args.shard_count != 1 or any(not 0 < n <= args.steps for n in args.snapshot_steps):
            raise ValueError('Invalid training snapshots/shards')
        if not args.resume:
            control.ensure_fresh_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    rows = load_data(args.data)
    try:
        pipe, official = load_pipe(args)
        if args.mode == 'cache-conditions':
            cache_conditions(args, pipe, rows, official)
        elif args.mode == 'audit-init':
            save_json(args.output / 'initial-state.json', {**official, **control.initialization_manifest(pipe, args)})
        else:
            train(args, pipe, rows, official)
    except control.NumericFailure as error:
        control.write_numeric_failure(args.output, error)
        raise


if __name__ == '__main__':
    main(parser().parse_args())
