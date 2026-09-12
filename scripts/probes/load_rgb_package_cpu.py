"""Actually load an exported Writer on CPU; this is not a functional image test."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'base-model', 'official-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('CPU loading evidence already exists')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from vision_memory.dreamlite.writer_package import file_sha, load_writer_package
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    started = time.monotonic()
    manifest_sha = file_sha(a.package / 'manifest.json')
    print('Loading the sealed official pipeline and parameter-only package on CPU', flush=True)
    pipe, manifest = load_writer_package(a.package, base_model=a.base_model,
                                        official_source=a.official_source, device='cpu')
    modules = {}
    for name in ('unet', 'vae', 'text_encoder'):
        module = getattr(pipe, name)
        parameters = list(module.parameters())
        if module.training or any(p.requires_grad or p.grad is not None or p.device.type != 'cpu'
                                  or p.dtype != torch.float32 for p in parameters):
            raise ValueError('Loaded model violated the frozen CPU FP32 contract: ' + name)
        modules[name] = {'parameter_tensors': len(parameters), 'parameter_values': sum(p.numel() for p in parameters),
                         'device': 'cpu', 'dtype': 'float32', 'frozen_eval': True}
    state = torch.load(a.package / 'unet-parameters.pt', map_location='cpu', weights_only=True)
    parameters = dict(pipe.unet.named_parameters())
    if set(state) != set(parameters) or any(not torch.equal(value, parameters[name]) for name, value in state.items()):
        raise ValueError('The actually loaded official U-Net differs from the export')
    del state
    memory = OfficialRGBMemory(pipe, guidance_scale=manifest['guidance_scale'])
    if memory.image.size != (1024, 1024) or memory.image.mode != 'RGB' or torch.cuda.is_initialized():
        raise ValueError('Unexpected initial state or GPU initialization')
    if file_sha(a.package / 'manifest.json') != manifest_sha:
        raise ValueError('Package identity changed during loading')
    result = {'package_manifest_sha256': manifest_sha, 'package_status': manifest['status'],
              'package_weights_sha256': manifest['weights_sha256'], 'modules': modules,
              'loaded_unet_bitwise_equal_export': True, 'cuda_initialized': False,
              'initial_memory': 'RGB1024x1024', 'writer_calls': 0, 'reader_calls': 0,
              'elapsed_seconds': time.monotonic() - started,
              'inputs': {'package': str(a.package), 'base_model': str(a.base_model), 'official_source': str(a.official_source)},
              'scope': 'Real pipeline/package loading and parameter equality only. No bank/parent checkpoint supplied; no image generation, Reader inference or functional success claim.'}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
