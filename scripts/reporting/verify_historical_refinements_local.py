"""Verify both downloaded raw archives and recount every endpoint and training record."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reporting.collect_historical_refinement import collect, sha

ARCHIVES = {'three': '21ba76b1235410f7c5603231d2caae7f0f568e59beb4a4ec335fb6b9c433d7f6',
            'five': '14bbaaa17ae4f85ac90c3e30d9ed97a342eb84517087fea7314f08ae748e8ee1'}


def main():
    results = ROOT / 'reports/official-alignment-results-20260913'
    for arm, digest in ARCHIVES.items():
        archive_path = results / f'historical-{arm}-query-full-evidence.tgz'
        if sha(archive_path) != digest:
            raise ValueError('Incomplete or changed downloaded archive: ' + arm)
        with tempfile.TemporaryDirectory(prefix='refinement-check-', dir=ROOT / '.cache') as temporary:
            directory = Path(temporary).resolve()
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    path = (directory / member.name).resolve()
                    if not member.isfile() or not path.is_relative_to(directory):
                        raise ValueError('Unexpected archive member')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
            summary = collect(directory, results / 'historical-writer-bank-manifest.json', arm, text_only=True)
            remote = json.loads((directory / 'verified-summary.json').read_bytes())
            except_fields = {'verified_artifact_files', 'artifacts_omitted_locally', 'all_artifacts_verified_here'}
            if {key: value for key, value in remote.items() if key not in except_fields} != {
                    key: value for key, value in summary.items() if key not in except_fields}:
                raise ValueError('Local raw recount differs from full remote verification')
            from PIL import Image
            images = list(directory.rglob('*.png'))
            if len(images) != 32:
                raise ValueError('Missing FP32 visualization or RGB endpoint PNG')
            for path in images:
                with Image.open(path) as image:
                    if image.mode != 'RGB' or image.size != (1024, 1024):
                        raise ValueError('Wrong endpoint image format')
                    image.verify()
            summary.update(archive_sha256=digest, archive_bytes=archive_path.stat().st_size,
                endpoint_pngs_verified=32, full_remote_artifact_verification=remote['all_artifacts_verified_here'],
                scope='All480 endpoint raw rows,176 checkpoint raw rows,4096 optimizer records and4112 trajectory index entries recounted locally. Large PT and intermediate PNGs remain remote; teacher success is not Writer success.')
            (results / f'historical-{arm}-query-full-local-verification.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            print(json.dumps({key: value for key, value in summary.items() if key not in ('results', 'artifacts_omitted_locally')}))


if __name__ == '__main__':
    main()
