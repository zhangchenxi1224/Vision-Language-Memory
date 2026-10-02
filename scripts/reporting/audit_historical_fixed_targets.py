"""Audit the existing hash-only single-target rule before broader Writer fitting."""
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from vision_memory.training.latent_bank_unet import member_split
from scripts.reporting.collect_transition_endpoint import sha, read

directory = ROOT / 'reports/official-alignment-results-20260913'
bank_path = directory / 'historical-writer-bank-manifest.json'
archive = directory / 'historical-fp32-readback-evidence.tgz'
if sha(bank_path) != 'd54895adb15b91c3befd52c57f6248c76cf2abadb44916af830726a0736e688c':
    raise ValueError('Broader bank changed')
if sha(archive) != 'b002f07d9ab7ca37f7dadebf8282a0ac0fb312f8a38d37b21bcfbe04b5ddc5c5':
    raise ValueError('Actual readback archive changed')
bank = read(bank_path)
teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
with tarfile.open(archive) as stream:
    rows = [json.loads(line) for line in stream.extractfile('generations.jsonl').read().splitlines()]
selected = []
for group in bank['groups']:
    teacher_id = member_split(group['teacher_ids'])[0][0]
    teacher = teachers[teacher_id]
    matching = [row for row in rows if row['condition'] == 'matched' and row['target_index'] == group['historical_target_index']
                and f"-seed-{row['seed']}-" in teacher_id]
    if len(matching) != 10 or {(row['image_form'], row['prompt_id']) for row in matching} != {
            (form, prompt) for form in ('fp32_vae_decoded', 'rgb_uint8') for prompt in group['question_variants']}:
        raise ValueError('Selected target lacks both forms of all five queries')
    failures = []
    for row in matching:
        passed = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645]
        if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Raw target readback differs from the score')
        if not passed:
            failures.append({key: row[key] for key in ('image_form', 'prompt_id', 'query', 'gold', 'raw', 'generated_token_ids')})
    selected.append({'question_id': group['question_id'], 'target_index': group['historical_target_index'],
        'teacher_id': teacher_id, 'latent_sha256': teacher['latent_sha256'],
        'historical_checkpoint_sha256': teacher['historical_checkpoint_sha256'], 'correct_eos': 10 - len(failures),
        'rows': 10, 'failures': failures})
result = {'selection_rule': 'existing member_split hash order, first training member; no outcome-based substitution',
    'bank_sha256': sha(bank_path), 'readback_archive_sha256': sha(archive), 'selected': selected,
    'selected_questions': len(selected), 'questions_passing_both_forms_all_five': sum(not item['failures'] for item in selected),
    'correct_eos': sum(item['correct_eos'] for item in selected), 'raw_rows': sum(item['rows'] for item in selected),
    'writer_calls': 0, 'optimizer_updates': 0,
    'interpretation': 'Two selected target images already fail some Reader queries. Exact latent imitation cannot by itself certify all-five-query success. This is not a measured or theoretical accuracy bound for a stochastic Writer.',
    'next_action': 'Preserve all original data; refine the fixed selection for all sixteen questions with the previously verified three-prompt FP32 oracle recipe, then requalify both image forms. Do not drop or swap only the failed questions.'}
(directory / 'historical-fixed-target-readback.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'selected'}))
