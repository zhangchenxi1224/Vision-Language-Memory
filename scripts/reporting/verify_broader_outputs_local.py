"""Recount a downloaded151 endpoint and optional validation archive from observed SHA256s.

Supply the full archive digests observed on shared storage, never a digest derived
only from the local download. Large tensors remain remote and omissions are explicit.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import read, sha
from scripts.reporting.collect_broader_endpoint import collect as endpoint_collect
from scripts.reporting.collect_broader_validation import collect as validation_collect

PROBE_COMMIT = 'c2ec407fe9f9600a23dc8009b657b7391b9aff6d'


def unpack(archive_path, digest, directory):
    if len(digest) != 64 or sha(archive_path) != digest:
        raise ValueError('Downloaded archive differs from the observed remote SHA256')
    directory = directory.resolve()
    seen = set()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = (directory / member.name).resolve()
            if (not member.isfile() or not path.is_relative_to(directory) or path in seen
                    or path.suffix not in ('.json', '.jsonl', '.log', '.png')):
                raise ValueError('Invalid or duplicated portable evidence archive member')
            seen.add(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())


def compare_recount(remote, local, excluded):
    # JSON-normalize Counters and integer dictionary keys produced by the recount.
    normalize = lambda value: json.loads(json.dumps({key: item for key, item in value.items() if key not in excluded}))
    if normalize(remote) != normalize(local):
        raise ValueError('Local full-raw recount differs from remote artifact verification')


def verify(endpoint_archive, endpoint_digest, *, validation_archive=None, validation_digest=None):
    with tempfile.TemporaryDirectory(prefix='broader-evidence-', dir=ROOT / '.cache') as temporary:
        root = Path(temporary)
        parent = root / 'parent'
        unpack(endpoint_archive, endpoint_digest, parent)
        endpoint = endpoint_collect(parent, parent / 'bank/manifest.json', text_only=True)
        compare_recount(read(parent / 'verified-summary.json'), endpoint,
                        {'artifacts_omitted_locally', 'all_remote_artifacts_verified_here'})
        result = {'endpoint_archive_sha256': endpoint_digest, 'endpoint_recount': endpoint}
        if validation_archive is not None:
            run = root / 'validation'
            unpack(validation_archive, validation_digest, run)
            if sha(run / 'preregistered-plan.json') != sha(parent / 'preregistered-experiment.json'):
                raise ValueError('Validation archive carries a different original plan')
            validation = validation_collect(run, parent, parent / 'bank/manifest.json', PROBE_COMMIT, text_only=True)
            compare_recount(read(run / 'verified-summary.json'), validation,
                {'artifacts_omitted_locally', 'all_files_verified_here', 'pixel_noise_trajectory_tensors_checked'})
            from PIL import Image
            images = list(run.glob('*.png'))
            if len(images) != validation['generated_images']:
                raise ValueError('Missing generated PNGs in validation archive')
            for path in images:
                with Image.open(path) as image:
                    if image.mode != 'RGB' or image.size != (1024, 1024):
                        raise ValueError('Unexpected generated image format')
                    image.verify()
            result.update(validation_archive_sha256=validation_digest, validation_recount=validation,
                pngs_verified=len(images), scope='All raw query/event/seed/token/EOS cells and PNG seals recounted locally. PT tensors remain remote; endpoint-only PNGs do not establish FP32 tensor equality locally.')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-archive', type=Path, required=True)
    parser.add_argument('--endpoint-sha256', required=True)
    parser.add_argument('--validation-archive', type=Path)
    parser.add_argument('--validation-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.validation_archive) != bool(args.validation_sha256):
        parser.error('Validation archive and independently observed SHA256 must be supplied together')
    result = verify(args.endpoint_archive, args.endpoint_sha256,
        validation_archive=args.validation_archive, validation_digest=args.validation_sha256)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    concise = {'endpoint': {phase: summary['correct_eos'] for phase, summary in result['endpoint_recount']['phases'].items()},
        'actual_draws_replayed': result['endpoint_recount']['exact_draws_replayed']}
    if 'validation_recount' in result:
        value = result['validation_recount']
        concise['validation'] = {key: value[key] for key in ('mode', 'prefix_lane', 'matched_correct_eos', 'matched_rows', 'all_generated_correct_eos')}
    print(json.dumps(concise))
