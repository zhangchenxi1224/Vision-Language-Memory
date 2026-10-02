"""Read back an exported parameter file and compare every value to its sealed checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src')]
import torch
from vision_memory.dreamlite.writer_package import file_sha, inspect_package

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent-run', type=Path, required=True)
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = inspect_package(a.package)
checkpoint = a.parent_run / 'train/checkpoint-final.pt'
if file_sha(checkpoint) != manifest['parent_checkpoint_sha256']:
    raise ValueError('Original checkpoint differs from the export binding')
original = torch.load(checkpoint, map_location='cpu', weights_only=False)
exported = torch.load(a.package / 'unet-parameters.pt', map_location='cpu', weights_only=True)
if set(exported) != set(original['trainable_state']):
    raise ValueError('Export parameter names changed')
for name, value in exported.items():
    parent = original['trainable_state'][name]
    if value.dtype != parent.dtype or not torch.equal(value, parent):
        raise ValueError('Export changed parameter values: ' + name)
summary = {'package_manifest_sha256': file_sha(a.package / 'manifest.json'),
           'parent_checkpoint_sha256': manifest['parent_checkpoint_sha256'],
           'weights_sha256': manifest['weights_sha256'], 'parameter_tensors': len(exported),
           'parameter_values': sum(value.numel() for value in exported.values()),
           'all_parameter_values_bitwise_equal': True,
           'export_has_only_named_parameters': True,
           'export_bytes': (a.package / 'unet-parameters.pt').stat().st_size,
           'training_checkpoint_bytes': checkpoint.stat().st_size,
           'functional_status': manifest['status'],
           'scope': 'Serialization/stripping verification only; does not prove native inference parity or memory functionality'}
a.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
