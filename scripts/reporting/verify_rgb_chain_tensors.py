"""Cross-check captured chain tensors against PNG pixels and native trajectory endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
done = json.loads((a.run / 'complete.json').read_text())
rows = [json.loads(line) for line in (a.run / 'generations.jsonl').read_text().splitlines()]
selected = [row for row in rows if row['prompt_id'] == 'original_open']
if len(selected) != 96:
    raise ValueError('Require all96 registered writes')
checked = []
for row in selected:
    image_path = a.run / row['image_artifact']
    tensor_path = image_path.with_suffix('.pt')
    if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != done['artifact_hashes'][tensor_path.name]:
        raise ValueError('Tensor file changed')
    payload = torch.load(tensor_path, map_location='cpu', weights_only=True)
    with Image.open(image_path) as image:
        if image.mode != 'RGB' or image.size != (1024, 1024):
            raise ValueError('Unexpected stored memory format')
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
    if not torch.equal(payload['image'], pixels):
        raise ValueError('Reader tensor differs from actual saved PNG pixels')
    if (len(payload['trajectory']) != 29 or not torch.equal(payload['trajectory'][0], payload['noise'])
            or not torch.equal(payload['trajectory'][-1], payload['latent'])):
        raise ValueError('Native trajectory initial or final state differs')
    noise = torch.randn(payload['noise'].shape, generator=torch.Generator().manual_seed(row['noise_seed']), dtype=torch.float32)
    if not torch.equal(noise, payload['noise']):
        raise ValueError('Recorded noise is not the preregistered independent Gaussian')
    checked.append({'image': image_path.name, 'reader_pixels_equal_png': True,
                    'native_pure_noise_start': True, 'trajectory_states': 29})
a.output.write_text(json.dumps({'complete_sha256': hashlib.sha256((a.run / 'complete.json').read_bytes()).hexdigest(),
    'writes_checked': len(checked), 'checks': checked,
    'source_encoding_scope': 'Actual official source encoding equality is asserted by the immutable native sampler at every write; this audit checks captured output pixels and noise/trajectory tensors'}, indent=2) + '\n')
print(json.dumps({'writes_checked': len(checked), 'pixel_noise_trajectory_checks': 'passed'}))
