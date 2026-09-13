"""Preserve all sixteen historical questions and64 qualified original-EOS targets.

This is a broader shared-Writer training bank, not a claim of broad Writer
success. All previously observed rewrite failures remain in its provenance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.probes.historical_fp32_readback import PANEL_SHA, load_historical_latent, summarize
from scripts.reporting.collect_transition_endpoint import BANK_SHA, read, jsonl, sha

READBACK_SHA = '091eb3d68eba1553ff7f34dab7a620e044003443ac811064ccd3730dee890a0e'
QUALIFICATION_SHA = '3ea31c28bcf985c914af17bc5bd11c182feb7ebf06365fd4a9d9448cf5f630eb'


def positive_controls(rows, panel):
    """Require both actual image forms for every original question; drop none."""
    summary = summarize(rows, panel)
    positive = {}
    for member in panel['members']:
        selected = [r for r in rows if r['target_index'] == member['target_index'] and r['seed'] == member['seed']
                    and r['condition'] == 'matched' and r['prompt_id'] == 'original_open']
        if len(selected) != 2 or {r['image_form'] for r in selected} != {'fp32_vae_decoded', 'rgb_uint8'}:
            raise ValueError('Missing original-question positive control')
        if not all(r['generated_token_ids'] == r['scorer']['gold_token_ids'] + [151645] for r in selected):
            raise ValueError('A preregistered target failed its original question; do not silently discard it')
        positive[member['target_index'], member['seed']] = selected
    return positive, summary


def donor_member(target, targets, members):
    # Fixed metadata rule; never choose based on a current Reader's answer.
    candidates = [other for other in targets if other['gold'] != target['gold']]
    candidates.sort(key=lambda other: (other['topic'] != target['topic'], other['target_index']))
    selected = candidates[0]
    return min((member for member in members if member['target_index'] == selected['target_index']), key=lambda m: m['seed']), selected


def build(panel_path, readback, qualification_path, reference_bank, output, expected_commit):
    import torch
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256
    from vision_memory.repro import canonical_tensor_sha256
    if torch.cuda.is_initialized():
        raise ValueError('Bank assembly must not initialize CUDA or interrupt active training')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the exact clean bank assembly source')
    if sha(panel_path) != PANEL_SHA or sha(readback / 'complete.json') != READBACK_SHA or sha(qualification_path) != QUALIFICATION_SHA:
        raise ValueError('Historical panel, actual readback or full tensor qualification changed')
    if sha(reference_bank) != BANK_SHA:
        raise ValueError('Gray source/model reference bank changed')
    panel, complete, qualified = read(panel_path), read(readback / 'complete.json'), read(qualification_path)
    if qualified['actual_tensor_reverification'] is not True or qualified['artifacts_omitted_locally']:
        raise ValueError('Require actual full remote tensor verification')
    if qualified['complete_sha256'] != READBACK_SHA:
        raise ValueError('Qualification refers to another readback')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name or sha(readback / name) != digest:
            raise ValueError('Readback artifact changed: ' + name)
    rows = jsonl(readback / 'generations.jsonl')
    controls, summary = positive_controls(rows, panel)
    if summary != complete['summary'] or summary != qualified['summary']:
        raise ValueError('Readback raw coverage differs from its full audit')
    reference, _ = load_teacher_bank(reference_bank)
    for old_key, current_key in (('mobile', 'dreamlite_mobile'), ('reader', 'qwen_reader')):
        if complete['identity']['snapshots'][old_key] != reference['snapshots'][current_key]:
            raise ValueError('Reference and historical readback models differ')
    gray = next(group for group in reference['groups'] if group['source_kind'] == 'blank_gray_1024')
    gray_path = Path(gray['source_latent_path'])
    if sha(gray_path) != gray['source_latent_file_sha256']:
        raise ValueError('Gray source latent changed')
    gray_tensor = torch.load(gray_path, map_location='cpu', weights_only=True)
    if canonical_tensor_sha256(gray_tensor) != gray['source_latent_sha256']:
        raise ValueError('Gray source tensor changed')
    targets = sorted(panel['targets'], key=lambda target: target['target_index'])
    members = sorted(panel['members'], key=lambda member: (member['target_index'], member['seed']))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'latents').mkdir()
    teachers, lookup, groups = [], {}, []
    for member in members:
        target = targets[member['target_index']]
        if target['target_index'] != member['target_index']:
            raise ValueError('Historical target indexing changed')
        qid = target['semantic_group_id'] + '-historical-prefix'
        latent = load_historical_latent(member)
        path = output / 'latents' / f"target-{member['target_index']:02d}-seed-{member['seed']}.pt"
        torch.save(latent, path)
        tid = qid + f"-seed-{member['seed']}-" + member['latent_sha256'][:16]
        teacher = {'teacher_id': tid, 'question_id': qid, 'answer': target['gold'], 'endpoint_step': 256,
            'evaluation_generation': {'do_sample': False, 'max_new_tokens': 32}, 'gold_eos_appended': True,
            'target': {'answer': target['gold'], 'strict_correct': True},
            'latent_path': str(path.resolve()), 'latent_file_sha256': file_sha256(path), 'latent_sha256': member['latent_sha256'],
            'source_run': member['historical_run'], 'historical_checkpoint_sha256': member['checkpoint_sha256'],
            'generation_file_sha256': complete['artifact_hashes']['generations.jsonl'],
            'qualification': 'both FP32 and RGB original-open answer plus immediateEOS; rewrites are not filtered',
            'original_positive_control_sha256': hashlib.sha256(json.dumps(controls[member['target_index'], member['seed']], sort_keys=True).encode()).hexdigest()}
        teachers.append(teacher)
        lookup[member['target_index'], member['seed']] = teacher
    for target in targets:
        qid = target['semantic_group_id'] + '-historical-prefix'
        donor, donor_target = donor_member(target, targets, members)
        donor_record = lookup[donor['target_index'], donor['seed']]
        group = {key: copy.deepcopy(value) for key, value in gray.items() if key.startswith('source_latent_')}
        group.update(question_id=qid, semantic_question_id=target['semantic_group_id'], source_kind='blank_gray_1024',
            event_text='\n'.join(event['event_text'] for event in target['event_stream']),
            source_event_stream=copy.deepcopy(target['event_stream']),
            event_input_scope='one gray-source write of the full ordered event-only prefix; not a live sequential-memory test',
            answer=target['gold'], question_variants=copy.deepcopy(target['question_variants']),
            question_instruction_contract='historical-r11-five-prompts/v1', termination_contract=copy.deepcopy(gray['termination_contract']),
            teacher_ids=[lookup[target['target_index'], seed]['teacher_id'] for seed in range(4)],
            planned_count=4, successful_run_count=4, historical_target_index=target['target_index'],
            donor_control={'latent_path': donor_record['latent_path'], 'latent_file_sha256': donor_record['latent_file_sha256'],
                'latent_sha256': donor_record['latent_sha256'], 'answer': donor_target['gold'],
                'owner_question_id': donor_record['question_id'],
                'interpretation': 'fixed different-answer donor owner; response to this target query is measured, not assumed'})
        groups.append(group)
    bank = {'schema': 'latent-teacher-bank/v1', 'bank_status': 'sealed', 'route': 'direct',
        'models': reference['models'], 'snapshots': reference['snapshots'], 'groups': groups, 'teachers': teachers,
        'semantic_question_count': 16, 'conditional_group_count': 16,
        'provenance': {'builder_commit': expected_commit, 'panel_sha256': PANEL_SHA, 'readback_complete_sha256': READBACK_SHA,
            'qualification_sha256': QUALIFICATION_SHA, 'gray_reference_bank_sha256': BANK_SHA,
            'selection': 'all16 preregistered targets times all4 original seeds; zero outcome filtering or replacement',
            'readback_summary_including_failures': summary,
            'inference_success_claim': False, 'question_exposure': 'allfive queries observed in teacher qualification; none is a fresh Writer research holdout',
            'scope': 'broader seen-question shared-Writer training input only; no claim of unseen entities, simultaneous facts, or sequential editing'}}
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(bank, indent=2, sort_keys=True) + '\n')
    loaded, tensors = load_teacher_bank(manifest)
    if len(loaded['groups']) != 16 or len(tensors) != 64 or torch.cuda.is_initialized():
        raise ValueError('Assembled bank coverage or CPU-only contract differs')
    if any(sha(path) != expected for path, expected in ((panel_path, PANEL_SHA), (readback / 'complete.json', READBACK_SHA),
           (qualification_path, QUALIFICATION_SHA), (reference_bank, BANK_SHA))):
        raise ValueError('Input binding changed during bank assembly')
    seal = {'manifest_sha256': sha(manifest), 'groups': 16, 'teachers': 64, 'optimizer_updates': 0,
            'cuda_initialized': False, 'readback_failures_preserved': True, 'scope': bank['provenance']['scope']}
    (output / 'complete.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    return seal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'readback', 'qualification', 'reference-bank', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.panel, a.readback, a.qualification, a.reference_bank, a.output, a.expected_commit)))
