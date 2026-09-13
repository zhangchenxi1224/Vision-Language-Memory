"""Explicit parameter-only initialization for a new official-FM experiment."""
from __future__ import annotations
import json
from pathlib import Path

from vision_memory.dreamlite.writer_package import file_sha, inspect_package, load_parameter_export


def initial_writer_binding(args):
    package = getattr(args, 'initial_writer_package', None)
    digest = getattr(args, 'initial_writer_package_sha256', None)
    if package is None and digest is None:
        return None
    if package is None or not digest:
        raise ValueError('Initial Writer requires its explicit package manifest SHA256')
    if (args.model_variant != 'base' or args.trainable_scope != 'full_unet'
            or args.flow_protocol != 'official' or getattr(args, 'baseline_reference', None)):
        raise ValueError('Parameter initialization requires official Base/full-U-Net and a new measured baseline')
    package = Path(package).resolve()
    if file_sha(package / 'manifest.json') != digest:
        raise ValueError('Initial Writer manifest differs from the requested SHA256')
    manifest = inspect_package(package)
    return {'kind': 'sealed_parameter_export', 'package': str(package), 'manifest_sha256': digest,
        'weights_sha256': manifest['weights_sha256'], 'parent_commit': manifest['parent_commit'],
        'parent_result_sha256': manifest['parent_result_sha256'],
        'parent_checkpoint_sha256': manifest['parent_checkpoint_sha256'],
        'parent_optimizer_steps': manifest['optimizer_steps'],
        'parent_status': manifest['status'], 'parameter_count': manifest['parameter_count'],
        'optimizer_and_rng': 'new experiment: fresh AdamW and configured seed; no parent optimizer or RNG restored',
        'baseline': 'measured from the initialized parameters before any new optimizer update'}


def apply_initial_writer(args, runtime, binding):
    if initial_writer_binding(args) != binding:
        raise ValueError('Initial Writer changed between binding and loading')
    package = Path(binding['package'])
    manifest = inspect_package(package)
    protocol = runtime['protocol_binding']
    base = json.loads((package / 'base-seal.json').read_text())
    if ({k: v for k, v in base.items() if k != 'model_dir'}
            != {k: v for k, v in protocol['base_snapshot'].items() if k != 'model_dir'}):
        raise ValueError('Initial Writer Base snapshot differs from this runtime')
    if (manifest['official_source_commit'] != protocol['official_source_commit']
            or manifest['native_steps'] != protocol['inference_steps']
            or manifest['guidance_scale'] != protocol['inference_guidance_scale']):
        raise ValueError('Initial Writer inference protocol differs from the new baseline')
    reader = runtime['snapshots']['qwen_reader']
    if any(reader[key] != value for key, value in manifest['reader_snapshot'].items()):
        raise ValueError('Initial Writer Reader snapshot differs from this runtime')
    unet = runtime['pipe'].unet
    parameters = list(unet.named_parameters())
    if not parameters or any(not p.requires_grad or 'lora_' in name for name, p in parameters):
        raise ValueError('Initialization expects every base U-Net parameter trainable without added LoRA')
    was_training = unet.training
    load_parameter_export(unet, package, manifest)
    # The shared export loader freezes inference weights. Restore the already
    # validated training scope and module mode before the training frozen audit.
    unet.requires_grad_(True).train(was_training)
