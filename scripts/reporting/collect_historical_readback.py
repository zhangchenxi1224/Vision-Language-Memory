"""Reverify all historical oracle readback cells and persisted pixel tensors."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.probes.historical_fp32_readback import PANEL_SHA, summarize, load_historical_latent
from scripts.reporting.collect_transition_endpoint import sha, read, jsonl


def collect(run, panel_path, *, text_only=False):
    import torch
    import numpy as np
    from PIL import Image
    from vision_memory.repro import canonical_tensor_sha256
    if sha(panel_path) != PANEL_SHA:
        raise ValueError('Historical panel differs')
    panel, complete = read(panel_path), read(run / 'complete.json')
    identity = complete['identity']
    if (read(run / 'identity.json') != identity or identity['commit'] != '5bcf6d0990e6e428c536ff21d58c0b4af69ce66b'
            or identity['panel_sha256'] != PANEL_SHA or identity['optimizer_updates'] != 0 or identity['writer_calls'] != 0):
        raise ValueError('Historical readback identity differs')
    expected = {'identity.json', 'generations.jsonl', 'blank.png'}
    for member in panel['members']:
        label = f"target-{member['target_index']:03d}-seed-{member['seed']:02d}"
        expected.update((label + '.pt', label + '.png'))
    if set(complete['artifact_hashes']) != expected:
        raise ValueError('Historical readback artifact coverage differs')
    omitted = []
    for name, digest in complete['artifact_hashes'].items():
        if text_only and name.endswith('.pt') and not (run / name).exists():
            omitted.append(name)
        elif sha(run / name) != digest:
            raise ValueError('Historical readback artifact changed: ' + name)
    rows = jsonl(run / 'generations.jsonl')
    summary = summarize(rows, panel)
    if summary != complete['summary']:
        raise ValueError('Raw answers differ from the sealed summary')
    checked = []
    for member in panel['members']:
        label = f"target-{member['target_index']:03d}-seed-{member['seed']:02d}"
        image = Image.open(run / (label + '.png'))
        if image.mode != 'RGB' or image.size != (1024, 1024):
            raise ValueError('Historical readback PNG shape/mode differs')
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
        hashes = {'rgb_uint8': canonical_tensor_sha256(pixels)}
        if not text_only:
            payload = torch.load(run / (label + '.pt'), map_location='cpu', weights_only=True)
            if not torch.equal(payload['latent'], load_historical_latent(member)) or not torch.equal(payload['rgb_uint8'], pixels):
                raise ValueError('Historical latent or persisted PNG pixels differ')
            fp32 = payload['fp32_vae_decoded']
            if fp32.dtype != torch.float32 or fp32.shape != pixels.shape or not torch.isfinite(fp32).all():
                raise ValueError('Invalid FP32 readback tensor')
            if not torch.equal((fp32 * 255).round().clamp(0, 255).byte().float() / 255., pixels):
                raise ValueError('RGB quantization differs from FP32 decode')
            hashes['fp32_vae_decoded'] = canonical_tensor_sha256(fp32)
        matching = [row for row in rows if row['target_index'] == member['target_index'] and row['seed'] == member['seed']]
        for row in matching:
            if row['image_artifact'] != label + '.pt' or (row['image_form'] in hashes and row['image_sha256'] != hashes[row['image_form']]):
                raise ValueError('Raw answer does not reference the actual persisted image')
        checked.append(label)
    blank = torch.full((1, 3, 1024, 1024), 128 / 255., dtype=torch.float32)
    blank_png = torch.from_numpy(np.asarray(Image.open(run / 'blank.png')).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
    if not torch.equal(blank, blank_png) or any(row['image_sha256'] != canonical_tensor_sha256(blank)
            or row['image_artifact'] != 'blank.png' for row in rows if row['condition'] == 'blank'):
        raise ValueError('Blank control pixels differ')
    return {'complete_sha256': sha(run / 'complete.json'), 'panel_sha256': PANEL_SHA,
        'summary': summary, 'checked_members': checked, 'png_count': 65, 'raw_rows': len(rows),
        'artifacts_omitted_locally': omitted, 'actual_tensor_reverification': not text_only,
        'scope': 'Direct historical oracle compatibility only; no shared Writer result or validation-based replacement of teachers.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--text-only', action='store_true')
    a = p.parse_args()
    result = collect(a.run, a.panel, text_only=a.text_only)
    summary = Path(str(a.output_prefix) + '-summary.json')
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        archive = Path(str(a.output_prefix) + '-evidence.tgz')
        with tarfile.open(archive, 'w:gz') as stream:
            for path in sorted(a.run.iterdir()):
                if path.suffix in ('.json', '.jsonl', '.png'):
                    stream.add(path, arcname=path.name)
            stream.add(a.panel, arcname='preregistered-panel.json')
            stream.add(summary, arcname='verified-summary.json')
        result['archive_sha256'] = sha(archive)
        result['archive_bytes'] = archive.stat().st_size
    print(json.dumps(result))


if __name__ == '__main__':
    main()
