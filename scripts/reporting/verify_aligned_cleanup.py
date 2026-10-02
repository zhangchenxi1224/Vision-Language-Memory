"""Check cleanup boundaries and retained scientific implementation."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / 'docs/alignment-cleanup-manifest.json').read_text(encoding='utf-8'))
    registry = json.loads((ROOT / 'experiments/registry.json').read_text(encoding='utf-8'))
    errors = []
    paths = [r['path'] for r in manifest['files']]
    if len(paths) != len(set(paths)) or len(paths) != manifest['removed_files']:
        errors.append('Removal manifest has duplicate paths or an incorrect count')
    for name in paths:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.exists():
            errors.append(f'Retired artifact still present or unsafe: {name}')
    for row in registry['families']:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_dir():
            errors.append(f'Missing registered directory: {row["path"]}')
        if row['role'] == 'current_performance' and row['path'].startswith('reports/official-'):
            errors.append('Engineering controls cannot enter current PrefEval score tables')
    result = subprocess.run(['git', 'diff', '--name-only', manifest['pre_cleanup_commit'], '--',
                             'src', 'configs', 'third_party', 'models.lock.json', 'data.lock.json',
                             'scripts/experiments', 'scripts/eval', 'scripts/train', 'scripts/inspire',
                             'reports/prefeval-k1-l0-l2-20260924',
                             'reports/prefeval-multitarget-20260927',
                             'reports/prefeval-official-alignment-20260923'],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        errors.append(result.stderr.strip())
    elif result.stdout.strip():
        errors.append('Scientific runtime changed during cleanup: ' + result.stdout.strip())
    result = {'removed_files': len(paths), 'removed_bytes': manifest['removed_bytes'],
              'retired_dataset_contracts': len(manifest['retired_dataset_tests']),
              'errors': errors, 'passed': not errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
