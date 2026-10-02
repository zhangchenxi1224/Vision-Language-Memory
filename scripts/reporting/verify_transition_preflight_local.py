"""Verify the downloaded immutable preflight archive without claiming training success."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/official-alignment-results-20260913'
def sha(data):
    return hashlib.sha256(data).hexdigest()

archive = OUT / 'transition-wording-preflight-evidence.tgz'
summary_path = OUT / 'transition-wording-preflight-summary.json'
assert sha(archive.read_bytes()) == 'f8e78e0db220f45c03872a27249dfa2f60eb4cbe9dd37584f8e76f09f15662ba'
assert sha(summary_path.read_bytes()) == 'df05f300b7587c681fce0b71dd7366e04973d200c2b8d4a4031b06172fb42132'
with tarfile.open(archive) as stream:
    files = {member.name: stream.extractfile(member).read() for member in stream.getmembers() if member.isfile()}
summary = json.loads(summary_path.read_bytes())
assert files['verified-summary.json'] == summary_path.read_bytes()
complete = json.loads(files['train/baseline/complete.json'])
assert sha(files['train/baseline/complete.json']) == summary['baseline_complete_sha256']
local, remote_only = [], []
for name, digest in complete['artifact_hashes'].items():
    key = 'train/baseline/' + name
    if key in files:
        assert sha(files[key]) == digest, key
        local.append(name)
    else:
        remote_only.append(name)
bank = json.loads((OUT / 'transition-wording-bank-manifest.json').read_bytes())
groups = {group['question_id']: group for group in bank['groups']}
rows = [json.loads(line) for line in files['train/baseline/generations.jsonl'].splitlines()]
assert len(rows) == 1350
assert len({(r['question_id'], r['condition'], r['noise_seed'], r['prompt_id']) for r in rows}) == 1350
correct = 0
matched = 0
for row in rows:
    group = groups[row['question_id']]
    assert row['query'] == group['question_variants'][row['prompt_id']]
    assert row['gold'] == group['answer']
    passed = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [group['termination_contract']['assistant_end_token_id']]
    assert passed == bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos'])
    if row['condition'] == 'matched':
        matched += 1
        correct += int(passed)
assert matched == 900 and correct == summary['baseline_correct_eos']
result = {'archive_sha256': sha(archive.read_bytes()), 'summary_sha256': sha(summary_path.read_bytes()),
          'raw_rows_verified': len(rows), 'matched_rows': matched, 'matched_correct_eos': correct,
          'local_phase_files_verified': local, 'phase_files_verified_remotely_only': remote_only,
          'scope': 'Untrained baseline only; no completed training or functional validation claim'}
(OUT / 'transition-wording-preflight-local-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'phase_files_verified_remotely_only'}))
