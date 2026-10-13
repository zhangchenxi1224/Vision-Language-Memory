"""Offline, CPU-only verification of a published B730 supplementary snapshot.

Supply the extracted Release snapshot to --snapshot. Original remote paths are
mapped to immutable archive objects; no platform connection or GPU is used.
"""
import argparse
import fnmatch
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(snapshot, hashes_only=False):
    snapshot = snapshot.resolve()
    receipt = load(snapshot / 'LOCAL-ARCHIVE-RECEIPT.json')
    if receipt['status'] != 'verified':
        raise ValueError('Unverified source archive')
    for item in receipt['files']:
        p = snapshot / item['path']
        if not p.resolve().is_relative_to(snapshot):
            raise ValueError('Unsafe archive path')
        raw = p.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Archive hash mismatch: ' + item['path'])
    if len(receipt['files']) != receipt['file_count']:
        raise ValueError('Archive file count mismatch')
    manifest = load(snapshot / 'archive-manifest.json')
    inventory = load(snapshot / 'inventory.json')
    config = load(snapshot / 'audit-tools/runtime-attempt2.json')
    mapping = {item['remote_path']: snapshot / item['local_object'] for item in manifest['files']}
    for item in manifest['files']:
        raw = mapping[item['remote_path']].read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('Remote-to-archive binding mismatch')
    if hashes_only:
        return {'status': 'hashes_verified_only', 'files': receipt['file_count']}
    spec = importlib.util.spec_from_file_location('published_b730_summarizer', mapping[config['summarizer']])
    summarizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarizer)
    api = summarizer.load_api(ROOT)
    parser_root = snapshot / 'audit-parser'
    api.mcq_source_verification = summarizer.validate_mcq_source(parser_root / 'utils/utils_mcq.py')
    api.mcq_source_raw_sha256 = api.mcq_source_verification['raw_sha256']
    api.mcq = api.official_mcq(parser_root)

    class ArchivePath:
        def __init__(self, value):
            self.logical = str(value)

        def __str__(self):
            return self.logical

        def __truediv__(self, other):
            return type(self)(self.logical.rstrip('/') + '/' + str(other))

        def physical(self):
            if self.logical in mapping:
                return mapping[self.logical]
            candidate = Path(self.logical)
            if candidate.resolve().is_relative_to(snapshot) and candidate.is_file():
                return candidate
            raise ValueError('Unarchived source: ' + self.logical)

        def __fspath__(self):
            return str(self.physical())

        @property
        def name(self):
            return PurePosixPath(self.logical).name

        def read_bytes(self):
            return self.physical().read_bytes()

        def read_text(self, encoding='utf-8'):
            return self.read_bytes().decode(encoding)

        def open(self, mode='r', **kwargs):
            if mode == 'rb':
                return io.BytesIO(self.read_bytes())
            if mode == 'r':
                return io.StringIO(self.read_text(kwargs.get('encoding', 'utf-8')))
            raise ValueError('Read-only audit')

        def glob(self, pattern):
            prefix = self.logical.rstrip('/') + '/'
            for logical in sorted(mapping):
                suffix = logical[len(prefix):]
                if logical.startswith(prefix) and '/' not in suffix and fnmatch.fnmatchcase(suffix, pattern):
                    yield type(self)(logical)

    summarizer.Path = ArchivePath
    new_rows = logical_rows = 0
    for item in inventory['summaries']:
        source = item['summary']['source_sets']
        raw = mapping[item['path']].read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256'] or load(snapshot / 'summaries' / (item['name'] + '.json')) != item['summary']:
            raise ValueError('Summary identity mismatch')
        actual = summarizer.summarize(config['data'], source['new_b730']['root'],
                                     source['reused_aug4']['root'], config['inventory'], api,
                                     controls_checkpoint_sha256=config['controls_checkpoint_sha256'])
        if actual != item['summary']:
            raise ValueError('Recomputed summary differs: ' + item['name'])
        new_rows += actual['new_gpu_readback_rows']
        logical_rows += actual['measurement_count']
        print(json.dumps({'verified': item['name']}), flush=True)
    audit = load(snapshot / 'strict-audit.json')
    if new_rows != audit['new_gpu_readback_rows'] or logical_rows != audit['logical_measurement_references']:
        raise ValueError('Audited denominator mismatch')
    return {'status': 'all_published_summaries_recomputed_exactly',
            'main_summaries': len(inventory['summaries']), 'new_gpu_readback_rows': new_rows,
            'logical_measurement_references': logical_rows,
            'full_experiment_complete': audit['full_experiment_complete'],
            'gpu_inference_performed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--hashes-only', action='store_true')
    args = p.parse_args()
    sys.dont_write_bytecode = True
    print(json.dumps(verify(args.snapshot, args.hashes_only), indent=2))
