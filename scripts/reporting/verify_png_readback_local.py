"""Verify observed archive hashes, source seals, all PNG pixels, and all raw readback rows."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def verify(readback_archive, readback_sha256, source_archive, source_sha256, expected_commit, *, continuation_validation_commit=None):
    from scripts.reporting.verify_broader_outputs_local import unpack
    from scripts.reporting.collect_transition_endpoint import sha, read
    from scripts.reporting.collect_png_readback import collect
    import torch
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix='png-readback-evidence-', dir=ROOT / '.cache') as directory:
            root = Path(directory)
            run, source = root / 'readback', root / 'source'
            unpack(readback_archive, readback_sha256, run)
            unpack(source_archive, source_sha256, source)
            identity = read(run / 'identity.json')
            if (sha(source / 'complete.json') != identity['source_complete_sha256']
                    or sha(source / 'generations.jsonl') != identity['source_generations_sha256']):
                raise ValueError('Downloaded source is not the experiment actually used by PNG readback')
            result = collect(run, expected_commit=expected_commit, source=source,
                continuation_validation_commit=continuation_validation_commit)
            if not result['all_png_pixels_verified_here']:
                raise ValueError('Incomplete local PNG pixel verification')
            return {'readback_archive_sha256': readback_sha256, 'source_archive_sha256': source_sha256,
                'recount': result,
                'scope': 'Every raw record and actual persisted PNG pixel checked locally. Original FP32 PT tensors remain remote; no local FP32 trajectory claim.'}
    finally:
        torch.set_num_threads(old_threads)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('readback-archive', 'source-archive', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('readback-sha256', 'source-sha256', 'expected-commit'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--continuation-validation-commit')
    a = p.parse_args()
    result = verify(a.readback_archive, a.readback_sha256, a.source_archive, a.source_sha256, a.expected_commit,
        continuation_validation_commit=a.continuation_validation_commit)
    a.output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    value = result['recount']
    print(json.dumps({'raw_rows': value['png']['raw_rows'], 'matched_correct_eos': value['png']['matched_correct_eos'],
        'png_images_verified_here': value['png_images_verified_here'], 'paired_matched': value['paired_matched'],
        'chain_parity_passed': value['chain_parity_passed']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
