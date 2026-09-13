"""Recheck the downloaded broader bank against its actual readback and original panel."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.build_historical_writer_bank import positive_controls, donor_member, PANEL_SHA, BANK_SHA
from scripts.reporting.collect_transition_endpoint import sha, read

directory = ROOT / 'reports/official-alignment-results-20260913'
manifest_path = directory / 'historical-writer-bank-manifest.json'
manifest_sha = 'd54895adb15b91c3befd52c57f6248c76cf2abadb44916af830726a0736e688c'
if sha(manifest_path) != manifest_sha or read(directory / 'historical-writer-bank-complete.json')['manifest_sha256'] != manifest_sha:
    raise ValueError('Downloaded broader bank differs from its actual remote seal')
bank = read(manifest_path)
panel_path = directory / 'historical-fp32-readback-panel.json'
reference_path = directory / 'transition-wording-bank-manifest.json'
archive = directory / 'historical-fp32-readback-evidence.tgz'
if sha(panel_path) != PANEL_SHA or sha(reference_path) != BANK_SHA or sha(archive) != 'b002f07d9ab7ca37f7dadebf8282a0ac0fb312f8a38d37b21bcfbe04b5ddc5c5':
    raise ValueError('Source evidence changed')
panel, reference = read(panel_path), read(reference_path)
with tarfile.open(archive) as stream:
    rows = [json.loads(line) for line in stream.extractfile('generations.jsonl').read().splitlines()]
positive, summary = positive_controls(rows, panel)
if summary != bank['provenance']['readback_summary_including_failures'] or bank['models'] != reference['models'] or bank['snapshots'] != reference['snapshots']:
    raise ValueError('Model identity or historical failures changed')
targets = {target['target_index']: target for target in panel['targets']}
teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
members = {(member['target_index'], member['seed']): member for member in panel['members']}
if len(bank['groups']) != 16 or len(teachers) != 64 or len(bank['teachers']) != 64:
    raise ValueError('Broader bank coverage differs')
seen = set()
gray = next(group for group in reference['groups'] if group['source_kind'] == 'blank_gray_1024')
for group in bank['groups']:
    target = targets[group['historical_target_index']]
    if (group['semantic_question_id'] != target['semantic_group_id'] or group['answer'] != target['gold']
            or group['question_variants'] != target['question_variants']
            or group['question_instruction_contract'] != 'historical-r11-five-prompts/v1'
            or group['event_text'] != '\n'.join(event['event_text'] for event in target['event_stream'])
            or group['source_event_stream'] != target['event_stream'] or len(group['teacher_ids']) != 4):
        raise ValueError('Original event/query text or group metadata changed')
    if group['source_kind'] != 'blank_gray_1024' or any(group[key] != value for key, value in gray.items() if key.startswith('source_latent_')):
        raise ValueError('Gray source identity changed')
    for seed, tid in enumerate(group['teacher_ids']):
        if tid in seen:
            raise ValueError('Duplicate teacher membership')
        seen.add(tid)
        teacher, member = teachers[tid], members[target['target_index'], seed]
        expected_sha = hashlib.sha256(json.dumps(positive[target['target_index'], seed], sort_keys=True).encode()).hexdigest()
        if (teacher['question_id'] != group['question_id'] or teacher['answer'] != target['gold']
                or teacher['latent_sha256'] != member['latent_sha256']
                or teacher['historical_checkpoint_sha256'] != member['checkpoint_sha256']
                or teacher['original_positive_control_sha256'] != expected_sha):
            raise ValueError('A target was replaced or detached from its actual original readback')
    donor, owner = donor_member(target, panel['targets'], panel['members'])
    expected_donor = next(t for t in teachers.values() if t['source_run'] == donor['historical_run'])
    actual = group['donor_control']
    if any(actual[key] != expected_donor[key] for key in ('latent_path', 'latent_file_sha256', 'latent_sha256')) or actual['answer'] != owner['gold']:
        raise ValueError('Fixed donor rule changed')
if seen != set(teachers):
    raise ValueError('Some original teacher is missing')
result = {'manifest_sha256': manifest_sha, 'groups_checked': len(bank['groups']), 'teacher_records_checked': len(teachers),
    'original_query_and_event_bytes_preserved': True, 'all_original_endpoints_preserved': True,
    'original_positive_controls_reverified': 128, 'full_readback_rows_reverified': len(rows),
    'historical_all_five_passing_members': summary['members_passing_both_forms_all_five'],
    'actual_latent_payloads_checked_locally': False, 'remote_latent_files': [teacher['latent_path'] for teacher in bank['teachers']],
    'scope': 'Downloaded bank and all source text/selection bindings checked locally; raw64 latent copies were verified by the remote CPU-only builder.'}
(directory / 'historical-writer-bank-local-verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'remote_latent_files'}))
