"""Combine the unchanged45 transition conditions with all16 refined questions."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import sha, read, BANK_SHA as TRANSITION_SHA
from scripts.experiments.refine_historical_writer_targets import plan, write, BANK_SHA as HISTORICAL_SHA

REFINER_COMMIT = '918e7d21ded6bff05133aa856f445ae6a4c089a0'
REFINEMENT_COMPLETE_SHA = 'e52d4777e99ea6f2dbf8614ce0b0acd2afc2e305382adb2fd781bd596179688d'
REFINED_BANK_SHA = '76ecd98f2bea422b756e7691425973ec9694f44fabca5905731c963a0f38b7dd'


def merge(transition, refined, provenance):
    if (len(transition['groups']) != 45 or len(transition['teachers']) != 45
            or len(refined['groups']) != 16 or len(refined['teachers']) != 16
            or transition['models'] != refined['models'] or transition['snapshots'] != refined['snapshots']):
        raise ValueError('Require the complete45+16 conditions with identical model coordinates')
    groups = copy.deepcopy(transition['groups'] + refined['groups'])
    teachers = copy.deepcopy(transition['teachers'] + refined['teachers'])
    if (len({group['question_id'] for group in groups}) != 61
            or len({teacher['teacher_id'] for teacher in teachers}) != 61
            or {group['historical_target_index'] for group in refined['groups']} != set(range(16))):
        raise ValueError('Question/target collision or missing historical question')
    if any(len(group['teacher_ids']) != 1 for group in groups):
        raise ValueError('Require one fixed raw target per condition')
    return {'schema': 'latent-teacher-bank/v1', 'bank_status': 'sealed', 'route': 'direct',
        'models': copy.deepcopy(transition['models']), 'snapshots': copy.deepcopy(transition['snapshots']),
        'groups': groups, 'teachers': teachers, 'semantic_question_count': 17, 'conditional_group_count': 61,
        'transition_conditions': 45, 'historical_prefix_conditions': 16, 'provenance': provenance}


def build(transition_path, historical_path, refinement, output, expected_commit, *, expand_wordings=False):
    import torch
    from vision_memory.training.latent_bank_unet import load_teacher_bank
    if torch.cuda.is_initialized():
        raise ValueError('Bank assembly must stay on CPU')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require a clean fixed assembly source')
    if sha(transition_path) != TRANSITION_SHA or sha(historical_path) != HISTORICAL_SHA:
        raise ValueError('Original transition or historical bank changed')
    original = read(historical_path)
    if sha(refinement / 'complete.json') != REFINEMENT_COMPLETE_SHA:
        raise ValueError('The measured complete five-query refinement changed')
    complete = read(refinement / 'complete.json')
    plan_path = refinement / 'preregistered-plan.json'
    if (read(plan_path) != plan(original, all_five=True) or sha(plan_path) != complete['plan_sha256']
            or complete.get('state') != 'completed' or complete.get('bank_sealed') is not True
            or complete.get('questions') != 16 or complete.get('correct_eos') != 160
            or complete.get('additional_latent_updates') != 4096 or complete.get('writer_updates') != 0):
        raise ValueError('All16 fixed refined targets must actually qualify before merging')
    for lane in (0, 1):
        directory = refinement / 'lanes' / f'lane-{lane}'
        identity, terminal = read(directory / 'identity.json'), read(directory / 'complete.json')
        if (identity['commit'] != REFINER_COMMIT or identity['plan_sha256'] != sha(plan_path)
                or terminal['state'] != 'completed' or terminal['all_passed'] is not True
                or set(terminal['results']) != {str(index) for index in range(16) if index % 2 == lane}):
            raise ValueError('Refinement lane identity, coverage or terminal differs')
    refined_path = refinement / 'bank/manifest.json'
    if sha(refined_path) != complete['bank_manifest_sha256'] or sha(refined_path) != REFINED_BANK_SHA:
        raise ValueError('Refined bank differs from completed qualification')
    transition, _ = load_teacher_bank(transition_path)
    refined, _ = load_teacher_bank(refined_path)
    if refined['provenance']['refinement_commit'] != REFINER_COMMIT:
        raise ValueError('Unexpected refinement implementation')
    provenance = {'builder_commit': expected_commit, 'transition_bank': str(transition_path),
        'transition_bank_sha256': TRANSITION_SHA, 'original_historical_bank': str(historical_path),
        'original_historical_bank_sha256': HISTORICAL_SHA, 'refined_bank': str(refined_path),
        'refined_bank_sha256': sha(refined_path), 'refinement_complete_sha256': sha(refinement / 'complete.json'),
        'selection': 'all45 original transitions plus all16 fixed refined questions; no outcome filtering',
        'scope': '17 seen semantic questions. The16 historical conditions are full-prefix gray-source writes; broader sequential and multi-fact memory are not established by this bank.'}
    bank = merge(transition, refined, provenance)
    if expand_wordings:
        from scripts.experiments.broader_writer_protocol import augment
        bank = augment(bank)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'manifest.json'
    write(manifest, bank)
    loaded, tensors = load_teacher_bank(manifest)
    expected_groups = 151 if expand_wordings else 61
    if loaded['groups'][:45] != transition['groups'] or len(tensors) != expected_groups or torch.cuda.is_initialized():
        raise ValueError('Merge changed old conditions or failed CPU verification')
    seal = {'manifest_sha256': sha(manifest), 'conditional_groups': expected_groups, 'semantic_questions': 17,
        'unchanged_transition_groups': 45, 'historical_groups': 16, 'teacher_records': expected_groups,
        'writer_updates': 0, 'cuda_initialized': False, 'scope': provenance['scope']}
    write(output / 'complete.json', seal)
    return seal


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('transition-bank', 'historical-bank', 'refinement', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--expand-wordings', action='store_true')
    a = p.parse_args()
    print(json.dumps(build(a.transition_bank, a.historical_bank, a.refinement, a.output, a.expected_commit,
                           expand_wordings=a.expand_wordings)))
