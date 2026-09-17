"""Read the two complete historical RGB chain matrices; preserve exact scoring."""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reporting.compare_broader_validation_raw import read_rows


def main():
    cache = ROOT / '.cache'
    bundle = cache / '2c5189a-results.tar'
    with bundle.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == 'dbd7bcb9bc43a6412623effc0ef543d4c1e4a7d2600745693df29f154d078516'
    report = {'scope': 'Historical observed RGB chains only; unchanged exact tokens plus immediate EOS.',
              'bundle_sha256': digest, 'matrices': {}}
    with tarfile.open(bundle) as outer:
        for lane in ('logical', 'fresh-wording'):
            directory = cache / f'e372f3c-{lane}-chains-parts'
            manifest = json.loads((directory / 'manifest.json').read_bytes())
            before = read_rows(directory / manifest['file'], manifest['sha256'])
            payload = outer.extractfile(f'2c5189a-{lane}-chains-evidence.tgz').read()
            target = cache / f'prefeval-parent-{lane}-chains.tgz'
            target.write_bytes(payload)
            after = read_rows(target, hashlib.sha256(payload).hexdigest())
            assert before.keys() == after.keys() and len(after) == 480
            changes = Counter()
            failures = []
            for key, (old, old_pass) in before.items():
                new, new_pass = after[key]
                for field in ('query', 'gold', 'event_text'):
                    assert old.get(field) == new.get(field)
                changes[f'{int(old_pass)}->{int(new_pass)}'] += 1
                if not old_pass or not new_pass:
                    failures.append({'cell': key, 'event': new.get('event_text'),
                                     'gold': new['gold'], 'before': old['raw'], 'after': new['raw'],
                                     'before_tokens': old['generated_token_ids'],
                                     'after_tokens': new['generated_token_ids'],
                                     'before_pass': old_pass, 'after_pass': new_pass})
            report['matrices'][lane] = {
                'before_archive_sha256': manifest['sha256'],
                'after_archive_sha256': hashlib.sha256(payload).hexdigest(),
                'changes': dict(changes), 'failures': failures,
                'before_correct': sum(p for _, p in before.values()),
                'after_correct': sum(p for _, p in after.values())}
    output = Path(__file__).with_name('parent-chain-comparison.json')
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for lane, result in report['matrices'].items():
        print(lane, result['changes'])
        print(Counter((r['cell'][0], r['gold'], r['before'], r['after']) for r in result['failures']))


if __name__ == '__main__':
    main()
