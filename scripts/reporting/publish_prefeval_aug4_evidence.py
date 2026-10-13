"""Build a byte-preserving scientific subset of the completed private run archive.

No GPU work or mutation of the source archive. The public subset deliberately
excludes platform/session/control-plane records; original archive identity is
retained. Use verify_prefeval_aug4_publication.py to independently rescore it.
"""
import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/prefeval-aug4x3-init-ablation-exposure512-20261009'
SOURCE_SHA = '6e2a3c8f012ccac939013f95e33db3d80b721dbf0ae7c10eb544ef90cf1ae015'
ASSET = 'prefeval-aug4x3-scientific-evidence-20261013.tar.gz'
RELEASE = 'https://github.com/zhangchenxi1224/Vision-Language-Memory/releases/tag/prefeval-aug4x3-final-20261013'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def selected(name):
    return (name.startswith(('frozen-repo/', 'frozen-data/', 'run/evaluation/',
                             'run/teacher-evaluation/', 'run/formal/', 'run/teachers/',
                             'run/conditions/', 'run/precheck/', 'run/tasks/', 'run/logs/'))
            or name in {'archive/retained-artifacts.jsonl', 'run/teacher-summary.json',
                        'run/teacher-accepted.json', 'run/results-index.json',
                        'run/identity.json', 'run/precheck-decision.json',
                        'deployment/model-seal.json',
                        'deployment/initialization-audits-passed.json'}
            or (name.startswith('deployment/audit-host') and name.endswith('/initial-state.json'))
            or name.startswith('deployment/pooled-eval-20261013/')
            or name in {'deployment/job-migration-20261011/prefeval_aug4_two_gpu_writer.py',
                        'deployment/job-migration-20261011/TWO_GPU_WRITER.md'})


def scrub_operational(value):
    """Public audit view; source bytes and their digest remain in local custody."""
    if isinstance(value, dict):
        return {k: scrub_operational(v) for k, v in value.items()
                if k not in {'owner_token', 'token', 'ssh_command', 'ssh_password', 'password'}}
    if isinstance(value, list):
        return [scrub_operational(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/prefeval-final-publication-20261013')
    args = ap.parse_args()
    if sha(args.source) != SOURCE_SHA:
        raise ValueError('Source archive checksum mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / 'extracted'
    source_manifest = None
    with tarfile.open(args.source, 'r|gz') as src:
        for member in src:
            if member.name == 'archive/SHA256-MANIFEST.json':
                source_manifest = json.load(src.extractfile(member))
                break
    if not source_manifest:
        raise ValueError('Missing original per-file hash manifest')
    inventory = {}
    secret_pattern = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    with (args.output / ASSET).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as dst:
                with tarfile.open(args.source, 'r|gz') as src:
                    for member in src:
                        if not member.isfile() or not selected(member.name):
                            continue
                        name = PurePosixPath(member.name)
                        if name.is_absolute() or '..' in name.parts:
                            raise ValueError('Unsafe source member')
                        data = src.extractfile(member).read()
                        digest = hashlib.sha256(data).hexdigest()
                        expected = source_manifest[member.name]
                        if digest != expected['sha256'] or len(data) != expected['bytes']:
                            raise ValueError('Original member digest mismatch: ' + member.name)
                        if secret_pattern.search(data):
                            raise ValueError('Potential credential in: ' + member.name)
                        inventory[member.name] = {'bytes': len(data), 'sha256': digest}
                        info = tarfile.TarInfo(member.name)
                        info.size, info.mode, info.mtime = len(data), 0o644, 0
                        dst.addfile(info, io.BytesIO(data))
                        # Full subset on disk enables independent verification with frozen code.
                        target = extracted / member.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        if member.name.startswith('run/evaluation/') and member.name.endswith('/summary.json'):
                            local = REPORT / 'evidence/summaries' / name.parts[-2] / 'summary.json'
                            local.parent.mkdir(parents=True, exist_ok=True)
                            local.write_bytes(data)
                        if member.name in {'run/teacher-summary.json', 'run/teacher-accepted.json'}:
                            local = REPORT / 'evidence' / name.name
                            local.parent.mkdir(parents=True, exist_ok=True)
                            local.write_bytes(data)
                manifest_bytes = (json.dumps(inventory, sort_keys=True, indent=2) + '\n').encode()
                info = tarfile.TarInfo('PUBLIC-SHA256-MANIFEST.json')
                info.size, info.mode, info.mtime = len(manifest_bytes), 0o644, 0
                dst.addfile(info, io.BytesIO(manifest_bytes))
                (extracted / info.name).write_bytes(manifest_bytes)
    receipt_sources = [
        'mixed16-20261013-r4/final-completion-audit.json',
        'mixed16-20261013-r4/final-checkpoints-verified.json',
        'mixed16-20261013-r4/completion-resource-cleanup.json',
        'final-archive-20261013/FINAL-ARCHIVE.json',
        'COMPLETED-20261013.json',
    ]
    provenance = []
    for rel in receipt_sources:
        source = REPORT / rel
        destination = REPORT / 'evidence/audit' / source.name
        save(destination, scrub_operational(json.loads(source.read_text(encoding='utf-8-sig'))))
        provenance.append({'source_local_relative_path': rel, 'source_sha256': sha(source),
                           'public_view': destination.relative_to(REPORT).as_posix(),
                           'public_view_sha256': sha(destination),
                           'transformation': 'JSON formatting; remove credential/control-token fields recursively; scientific fields unchanged'})
    manifest = {
        'schema': 'prefeval-aug4-publication/v1', 'release_url': RELEASE,
        'asset': ASSET, 'asset_sha256': sha(args.output / ASSET),
        'asset_bytes': (args.output / ASSET).stat().st_size,
        'member_count': len(inventory), 'uncompressed_member_bytes': sum(v['bytes'] for v in inventory.values()),
        'member_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'original_archive': {'filename': args.source.name, 'sha256': SOURCE_SHA,
                             'bytes': args.source.stat().st_size, 'retained_locally_and_on_project_storage': True},
        'selection_implementation': 'scripts/reporting/publish_prefeval_aug4_evidence.py:selected',
        'audit_views': provenance,
        'summary_files': {k: v for k, v in inventory.items() if k.startswith('run/evaluation/') and k.endswith('/summary.json')},
        'limitations': ['This asset is a scientific subset, not the original operational archive.',
                       'Model weights, PNG pixels and caches remain on project storage; retained-artifacts.jsonl records paths and hashes.',
                       'Rescoring uses already generated raw Reader outputs; it does not reproduce GPU inference or training.'],
    }
    assert len(manifest['summary_files']) == 84
    snapshot = REPORT / 'evidence/original-final-report-20261013.md'
    if snapshot.exists():
        manifest['original_report_snapshot'] = {'path': snapshot.relative_to(REPORT).as_posix(),
                                               'sha256': sha(snapshot)}
    save(REPORT / 'PUBLICATION-MANIFEST.json', manifest)
    (args.output / 'SHA256SUMS').write_text(manifest['asset_sha256'] + '  ' + ASSET + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest[k] for k in ('asset', 'asset_sha256', 'asset_bytes', 'member_count')}))


if __name__ == '__main__':
    main()
