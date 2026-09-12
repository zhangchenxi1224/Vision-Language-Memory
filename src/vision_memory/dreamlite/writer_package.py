"""Inference-only full-U-Net packages with no runtime teacher-bank dependency."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

SCHEMA = 'dreamlite-official-rgb-writer/v1'
UPSTREAM = 'a6e20c8cc94027f37dd7c5a81b0b3b472aa18409'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_completed_writer(parent_run, output):
    """Export a completed experimental endpoint; do not certify it as usable."""
    parent_run, output = Path(parent_run), Path(output)
    train = parent_run / 'train'
    result = json.loads((train / 'result.json').read_text())
    terminal = json.loads((parent_run / 'terminal.json').read_text())
    identity = json.loads((train / 'identity.json').read_text())
    runtime = json.loads((train / 'runtime.json').read_text())
    if (terminal.get('state') != 'completed' or terminal['training_result_sha256'] != file_sha(train / 'result.json')
            or identity.get('model_variant') != 'base' or identity.get('trainable_scope') != 'full_unet'
            or identity.get('flow_protocol') != 'official'):
        raise ValueError('Require a sealed completed official Base full-U-Net endpoint')
    checkpoint = train / 'checkpoint-final.pt'
    if file_sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from the completed result')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload.get('schema_version') != 1 or payload['manifest'] != {**identity, **runtime}
            or payload['optimizer_step'] != identity['steps'] or result['optimizer_steps'] != identity['steps']):
        raise ValueError('Checkpoint cursor or manifest differs from the completed run')
    state = payload['trainable_state']
    if not state or any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                        or not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError('Expected finite full-U-Net FP32 parameters')
    count = sum(tensor.numel() for tensor in state.values())
    if count != result['unet_trainable_parameters']:
        raise ValueError('Checkpoint parameter count differs from its training record')
    binding = runtime['additional_protocol_binding']
    if binding['official_source_commit'] != UPSTREAM or binding['inference_steps'] != 28:
        raise ValueError('Unexpected source or sampling protocol')
    output.mkdir(parents=True, exist_ok=False)
    torch.save({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, output / 'unet-parameters.pt')
    base = binding['base_snapshot']
    (output / 'base-seal.json').write_text(json.dumps(base, indent=2, sort_keys=True) + '\n')
    reader = runtime['snapshots']['qwen_reader']
    manifest = {'schema': SCHEMA, 'status': 'experimental_endpoint_requires_independent_functional_validation',
        'parent_commit': identity['git_commit'], 'parent_result_sha256': file_sha(train / 'result.json'),
        'parent_checkpoint_sha256': result['checkpoint_sha256'], 'optimizer_steps': payload['optimizer_step'],
        'training_bank_sha256': identity['bank_manifest_sha256'], 'semantic_question_count': identity['semantic_question_count'],
        'conditional_group_count': identity['conditional_group_count'], 'official_source_commit': UPSTREAM,
        'weights': 'unet-parameters.pt', 'weights_sha256': file_sha(output / 'unet-parameters.pt'),
        'parameter_count': count, 'base_seal_sha256': file_sha(output / 'base-seal.json'),
        'reader_snapshot': {key: reader[key] for key in ('repo_id', 'revision', 'manifest_sha256', 'snapshot_payload_sha256')},
        'guidance_scale': binding['inference_guidance_scale'], 'image_guidance_scale': 1., 'native_steps': 28,
        'writer_dtype': 'float32', 'reader_dtype': 'bfloat16',
        'persistent_state': 'only the previous generated1024x1024 RGB uint8 image',
        'runtime_inputs': 'base snapshot, pinned official source, optional frozen Reader, event/query stream; no teacher bank'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    if file_sha(checkpoint) != result['checkpoint_sha256'] or file_sha(train / 'result.json') != terminal['training_result_sha256']:
        raise RuntimeError('Parent changed during export; exported package is not sealed')
    (output / 'complete.json').write_text(json.dumps({'manifest_sha256': file_sha(output / 'manifest.json')}, indent=2) + '\n')
    return manifest


def inspect_package(package):
    package = Path(package)
    complete = json.loads((package / 'complete.json').read_text())
    if file_sha(package / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('Package manifest changed')
    manifest = json.loads((package / 'manifest.json').read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('weights') != 'unet-parameters.pt'
            or manifest.get('official_source_commit') != UPSTREAM or manifest.get('native_steps') != 28
            or manifest.get('image_guidance_scale') != 1. or not 1. <= manifest.get('guidance_scale', 0) <= 100.):
        raise ValueError('Unsupported writer package protocol')
    for name, key in (('unet-parameters.pt', 'weights_sha256'), ('base-seal.json', 'base_seal_sha256')):
        if file_sha(package / name) != manifest[key]:
            raise ValueError('Package file changed: ' + name)
    return manifest


def load_parameter_export(module, package, manifest):
    """Validate every tensor before mutating the destination, including frozen modules."""
    path = Path(package) / 'unet-parameters.pt'
    if file_sha(path) != manifest['weights_sha256']:
        raise ValueError('Exported weights changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    parameters = dict(module.named_parameters())
    if set(state) != set(parameters) or sum(p.numel() for p in parameters.values()) != manifest['parameter_count']:
        raise ValueError('Full parameter coverage differs from the destination U-Net')
    for name, parameter in parameters.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                or tensor.dtype != torch.float32 or parameter.dtype != torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError('Parameter shape, dtype, or finiteness mismatch: ' + name)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device))
    module.eval().requires_grad_(False)


def load_writer_package(package, *, base_model, official_source, device='cuda:0'):
    from vision_memory.repro.hf_snapshot import inspect_download
    manifest = inspect_package(package)
    source = Path(official_source).resolve()
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != UPSTREAM
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip()):
        raise ValueError('The pinned official DreamLite source changed')
    expected = json.loads((Path(package) / 'base-seal.json').read_text())
    actual = inspect_download(Path(base_model), expected['revision'])
    # Permit moving the exact same snapshot; contents and HF hashes remain fixed.
    if {k: v for k, v in expected.items() if k != 'model_dir'} != {k: v for k, v in actual.items() if k != 'model_dir'}:
        raise ValueError('Base snapshot differs from the training snapshot')
    sys.path.insert(0, str(source))
    from dreamlite import DreamLitePipelineLoRA
    pipe = DreamLitePipelineLoRA.from_pretrained(base_model, local_files_only=True, torch_dtype=torch.float32).to(device)
    load_parameter_export(pipe.unet, package, manifest)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    return pipe, manifest
