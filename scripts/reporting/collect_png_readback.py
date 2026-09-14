"""Recount complete PNG evidence, with optional independent local PNG pixel verification."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def collect(run, *, expected_commit, source=None, continuation_validation_commit=None):
    from scripts.reporting.collect_transition_endpoint import sha, read, jsonl
    from scripts.experiments.png_readback_protocol import plan, source_spec, LANES, compare, png_pixels
    sources, expected_parent = source_spec(continuation_validation_commit)
    from scripts.reporting.collect_broader_endpoint import BANK_SHA, registered_protocol
    run = Path(run)
    complete, identity = read(run / 'complete.json'), read(run / 'identity.json')
    if complete['identity'] != identity or identity['readback_commit'] != expected_commit or identity['plan'] != plan(continuation_validation_commit):
        raise ValueError('PNG identity or plan differs')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name or sha(run / name) != digest:
            raise ValueError('Changed PNG evidence artifact')
    if sha(run / 'source-complete.json') != identity['source_complete_sha256']:
        raise ValueError('Source seal differs')
    prior = read(run / 'source-complete.json')
    if (sha(run / 'source-generations.jsonl') != identity['source_generations_sha256']
            or identity['source_generations_sha256'] != prior['artifact_hashes']['generations.jsonl']
            or sha(run / 'bank.json') != BANK_SHA):
        raise ValueError('Source raw records or bank differ')
    source_identity = prior['identity']
    if (source_identity['probe_commit'] != sources[identity['validation_set']][0]
            or source_identity['checkpoint_sha256'] != identity['checkpoint_sha256']
            or source_identity.get('validation_set', 'registered') != identity['validation_set']):
        raise ValueError('Wrong source experiment')
    bank = read(run / 'bank.json')
    parent_commit, registered, _ = registered_protocol(bank, expected_parent)
    if identity['validation_set'] == 'fresh_wording_v1':
        from scripts.experiments.fresh_wording_validation import plan as fresh_plan
        registered = fresh_plan(registered, bank)
    if registered != source_identity['registered_plan'] or parent_commit != source_identity['parent_commit']:
        raise ValueError('Source matrix differs from the complete registered plan')
    original, rows, checks = jsonl(run / 'source-generations.jsonl'), jsonl(run / 'generations.jsonl'), read(run / 'image-checks.json')
    mode, lane, raw_count, _, image_count = LANES[identity['lane']]
    if (source_identity['mode'], source_identity['prefix_lane']) != (mode, lane):
        raise ValueError('Source lane differs')
    comparison = compare(rows, original, registered, bank, mode, lane)
    if (comparison != read(run / 'comparison.json') or len(rows) != raw_count or len(checks) != image_count
            or complete['raw_rows'] != raw_count or complete['images_checked'] != image_count
            or complete['all_matched_correct_eos'] != comparison['png']['all_generated_correct_eos']
            or complete['chain_parity_passed'] != comparison['chain_parity_passed']):
        raise ValueError('PNG recount differs from recorded results')
    matched_names = {row['image_artifact'] for row in rows if row['condition'] == 'matched'}
    if set(checks) != {row['image_artifact'] for row in rows}:
        raise ValueError('PNG image coverage differs')
    verified = 0
    for name, check in checks.items():
        if Path(name).name != name or Path(check['png_name']).name != check['png_name']:
            raise ValueError('Invalid PNG evidence path')
        if check['source_pt_sha256'] != prior['artifact_hashes'][str(Path(name).with_suffix('.pt'))]:
            raise ValueError('Source PT seal differs')
        expected_location = 'source' if name in matched_names else 'readback'
        if check['png_location'] != expected_location or check['png_name'] != str(Path(name).with_suffix('.png')):
            raise ValueError('Wrong PNG artifact binding')
        if expected_location == 'source' and check['png_file_sha256'] != prior['artifact_hashes'][check['png_name']]:
            raise ValueError('Original matched PNG seal differs')
        pixel_path = (Path(source) / check['png_name'] if source is not None else None) if expected_location == 'source' else run / check['png_name']
        if pixel_path is not None:
            from PIL import Image
            from vision_memory.repro import canonical_tensor_sha256
            if sha(pixel_path) != check['png_file_sha256']:
                raise ValueError('Actual PNG differs from recorded readback')
            with Image.open(pixel_path) as image:
                digest = canonical_tensor_sha256(png_pixels(image))
            if digest != check['image_sha256']:
                raise ValueError('Actual PNG pixels differ from Reader records')
            verified += 1
    for row in rows:
        if any(row.get(key) != value for key, value in checks[row['image_artifact']].items()):
            raise ValueError('Row differs from its independently bound PNG')
    return {'identity': identity, **comparison, 'png_images_verified_here': verified,
        'all_png_pixels_verified_here': verified == image_count, 'complete_sha256': sha(run / 'complete.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--archive', action='store_true')
    p.add_argument('--continuation-validation-commit')
    a = p.parse_args()
    from scripts.reporting.collect_transition_endpoint import sha
    summary = collect(a.run, expected_commit=a.expected_commit, source=a.source,
        continuation_validation_commit=a.continuation_validation_commit)
    if a.archive:
        archive = Path(str(a.output_prefix) + '-evidence.tgz')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in sorted(a.run.iterdir()):
                if path.is_file():
                    tar.add(path, arcname=path.name)
        summary['archive_sha256'] = sha(archive)
    target = Path(str(a.output_prefix) + '-summary.json')
    target.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({key: value for key, value in summary.items() if key not in ('identity', 'original', 'png')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
