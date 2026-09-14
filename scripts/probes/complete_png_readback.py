"""Read every persisted PNG with the frozen CLI Reader, then apply the locked scorer."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'reader-model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--validation-set', choices=('registered', 'fresh_wording_v1'), required=True)
    p.add_argument('--lane', choices=('confirmation', 'chains', 'prefix0', 'prefix1'), required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--continuation-validation-commit')
    a = p.parse_args()
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    if not a.worker:
        return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            env={**os.environ, **REQUIRED_DETERMINISM_ENV, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    from scripts.experiments.png_readback_protocol import source_spec, LANES, plan, png_pixels, compare
    sources, parent_commit = source_spec(a.continuation_validation_commit)
    from scripts.reporting.collect_transition_endpoint import read, sha, jsonl
    from scripts.reporting.collect_broader_validation import collect
    from scripts.reporting.collect_broader_endpoint import GOLD_IDS
    from scripts.train.train_latent_bank_unet import write_json, append_jsonl
    if (len(a.expected_commit) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the fixed clean readback source')
    if not math.isfinite(a.deadline_unix) or time.time() >= a.deadline_unix:
        raise ValueError('Require a finite future deadline')
    commit, prefix = sources[a.validation_set]
    source = a.runs / (prefix + '-' + a.lane)
    parent = a.runs / ('4fbc857-clear-retention-full4832' if a.continuation_validation_commit else 'b9f90e9-historical-wording-full4832')
    # Full prior suite already audits trajectories. This independently rechecks all
    # source file hashes, fixed cases, raw tokens, parent checkpoint and identities.
    source_summary = collect(source, parent, parent / 'bank/manifest.json', commit, text_only=True,
        logical_sampling_commit=parent_commit, validation_set=a.validation_set)
    if source_summary['artifacts_omitted_locally']:
        raise ValueError('Actual source tensors must exist for independent PNG binding')
    mode, lane, raw_count, _, image_count = LANES[a.lane]
    if (source_summary['mode'], source_summary['prefix_lane']) != (mode, lane):
        raise ValueError('Readback lane identity differs')
    complete = read(source / 'complete.json')
    original = jsonl(source / 'generations.jsonl')
    package_path = a.runs / (prefix + '-package/manifest.json')
    package = read(package_path)
    expected = package['reader_snapshot']
    if package['parent_checkpoint_sha256'] != source_summary['identity']['checkpoint_sha256']:
        raise ValueError('CLI package belongs to a different checkpoint')
    trained_reader = read(parent / 'train/runtime.json')['snapshots']['qwen_reader']
    if expected != {key: trained_reader[key] for key in ('repo_id', 'revision', 'manifest_sha256', 'snapshot_payload_sha256')}:
        raise ValueError('CLI Reader differs from the original training snapshot')
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.reader.open_answer import generate_short_answer
    from vision_memory.reader.open_eos import generation_diagnostics
    from scripts.inspire.model_snapshot_manifest import verify_snapshot_manifest
    from scripts.train.r11_new_frozen_dreamlite_oracle import _load_reader
    configure_strict_cuda_determinism(20260913)
    if a.device not in range(torch.cuda.device_count()) or torch.cuda.mem_get_info(a.device)[0] < 16 * 1024**3:
        raise RuntimeError('Requested Reader GPU must have 16 GiB free')
    actual = verify_snapshot_manifest(manifest_path=a.reader_model / '.snapshot_manifest.json', model_dir=a.reader_model,
        expected_repo_id=expected['repo_id'], expected_revision=expected['revision'])
    if any(actual[key] != expected[key] for key in ('manifest_sha256', 'snapshot_payload_sha256')):
        raise ValueError('Actual Reader snapshot differs from the CLI package')
    device = torch.device(f'cuda:{a.device}')
    processor, reader = _load_reader(SimpleNamespace(reader=a.reader_model), device, torch.bfloat16)
    reader.eval().requires_grad_(False)
    versions = {name: int(value._version) for name, value in reader.named_parameters()}
    a.output.mkdir(parents=True, exist_ok=False)
    identity = {'readback_commit': a.expected_commit, 'plan': plan(a.continuation_validation_commit), 'validation_set': a.validation_set,
        'lane': a.lane, 'source_complete_sha256': sha(source / 'complete.json'),
        'source_generations_sha256': sha(source / 'generations.jsonl'), 'source_path': str(source),
        'package_manifest_sha256': sha(package_path), 'reader_snapshot': expected,
        'checkpoint_sha256': source_summary['identity']['checkpoint_sha256'], 'optimizer_updates': 0,
        'inference_cli_source_sha256': sha(ROOT / 'scripts/inference/rgb_memory.py')}
    write_json(a.output / 'identity.json', identity)
    for src, dst in (('complete.json', 'source-complete.json'), ('generations.jsonl', 'source-generations.jsonl')):
        (a.output / dst).write_bytes((source / src).read_bytes())
    (a.output / 'bank.json').write_bytes((parent / 'bank/manifest.json').read_bytes())
    checks, rows = {}, []
    last_name, pixels = None, None
    with torch.no_grad():
        for old in original:
            if time.time() >= a.deadline_unix:
                raise TimeoutError('PNG readback deadline; retain partial evidence')
            name = old['image_artifact']
            if name != last_name:
                pt_path = (source / name).with_suffix('.pt')
                if sha(pt_path) != complete['artifact_hashes'][pt_path.name]:
                    raise ValueError('Captured source tensor file changed')
                payload = torch.load(pt_path, map_location='cpu', weights_only=True)
                fp = payload['image']
                if (fp.dtype != torch.float32 or tuple(fp.shape) != (1, 3, 1024, 1024)
                        or not torch.isfinite(fp).all() or fp.min() < 0 or fp.max() > 1
                        or canonical_tensor_sha256(fp) != old['image_sha256']):
                    raise ValueError('Source Reader tensor binding changed')
                png = (source / name).with_suffix('.png')
                if old['condition'] != 'matched':
                    png = a.output / (Path(name).stem + '.png')
                    Image.fromarray((fp[0].permute(1, 2, 0) * 255).round().byte().numpy()).save(png)
                elif sha(png) != complete['artifact_hashes'][png.name]:
                    raise ValueError('Persisted source PNG changed')
                with Image.open(png) as image:
                    pixels = png_pixels(image)
                if not torch.equal((fp * 255).round().byte(), (pixels * 255).round().byte()):
                    raise ValueError('PNG does not encode the captured source image')
                if mode == 'rgb_chains' and not torch.equal(fp, pixels):
                    raise ValueError('RGB chain readback pixels differ')
                check = {'source_image_sha256': old['image_sha256'], 'source_pt_sha256': sha(pt_path),
                    'png_file_sha256': sha(png), 'png_name': png.name,
                    'png_location': 'source' if old['condition'] == 'matched' else 'readback',
                    'image_sha256': canonical_tensor_sha256(pixels),
                    'pixel_max_abs_change': float((pixels - fp).abs().max())}
                if name in checks and checks[name] != check:
                    raise ValueError('Repeated image binding changed')
                checks[name] = check
                last_name = name
                del payload, fp
            # No target-conditioned forward, tokenization or scoring before generation.
            generated = generate_short_answer(model=reader, processor=processor, image=pixels.to(device),
                query=old['query'], device=device, max_new_tokens=32, do_sample=False)
            scorer = generation_diagnostics(generated, old['gold'], GOLD_IDS[old['gold']])
            excluded = {'raw', 'scorer', 'generated_token_ids', 'image_sha256'}
            meta = {key: value for key, value in old.items() if key not in excluded and key not in generated}
            row = {**meta, **generated, 'scorer': scorer, **checks[name]}
            rows.append(row)
            append_jsonl(a.output / 'generations.jsonl', row)
    comparison = compare(rows, original, source_summary['identity']['registered_plan'], read(parent / 'bank/manifest.json'), mode, lane)
    if len(rows) != raw_count or len(checks) != image_count:
        raise ValueError('Incomplete PNG readback coverage')
    if versions != {name: int(value._version) for name, value in reader.named_parameters()}:
        raise ValueError('Frozen Reader parameters changed')
    if sha(source / 'complete.json') != identity['source_complete_sha256'] or sha(package_path) != identity['package_manifest_sha256']:
        raise ValueError('Source or package changed during readback')
    for name, check in checks.items():
        path = (source if check['png_location'] == 'source' else a.output) / check['png_name']
        if sha(path) != check['png_file_sha256']:
            raise ValueError('PNG changed while Reader was running')
    write_json(a.output / 'image-checks.json', checks)
    write_json(a.output / 'comparison.json', comparison)
    write_json(a.output / 'complete.json', {'identity': identity, 'raw_rows': len(rows), 'images_checked': len(checks),
        'all_matched_correct_eos': comparison['png']['all_generated_correct_eos'],
        'chain_parity_passed': comparison['chain_parity_passed'],
        'artifact_hashes': {path.name: sha(path) for path in a.output.iterdir() if path.is_file()}})
    print(json.dumps({'raw_rows': len(rows), 'matched_correct_eos': comparison['png']['matched_correct_eos'],
        'chain_parity_passed': comparison['chain_parity_passed']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
