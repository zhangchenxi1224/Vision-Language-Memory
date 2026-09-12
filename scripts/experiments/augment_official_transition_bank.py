"""Add fixed event wordings to the verified source-transition bank without changing tensors."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

SOURCE_BANK_SHA = '3d89168c9d36e6df5ded067e54d13128dda1a3cf2c5516600901b017ebe377f2'
ENTITY = 'the indigo desk train 001123'


def extra_events(operation, state):
    if operation == 'noop':
        return (f'A package was delivered to {ENTITY}. Do not alter the music preference already saved.',
                f'There is a routine desk status update for {ENTITY}. Preserve whichever music preference the memory currently holds.')
    if operation == 'clear':
        return (f'Delete the saved music choice for {ENTITY}. Leave the music preference unset.',
                f'Erase the remembered music preference for {ENTITY}; no preference should remain saved.')
    if operation != 'set' or state not in ('ambient', 'jazz'):
        raise ValueError('Unexpected state transition')
    return (f'Record the current preferred music for {ENTITY}: {state}.',
            f'Replace any earlier music choice for {ENTITY} with {state}.')


def assemble(parent):
    if (len(parent['groups']) != 15 or len(parent['teachers']) != 15
            or parent['conditional_group_count'] != 15 or parent['semantic_question_count'] != 1):
        raise ValueError('Expected the exact fifteen-group source transition bank')
    result = copy.deepcopy(parent)
    result['groups'], result['teachers'] = [], []
    teachers = {t['teacher_id']: t for t in parent['teachers']}
    for original in parent['groups']:
        events = (original['event_text'], *extra_events(original['operation'], original['target_state']))
        if len(original['teacher_ids']) != 1 or len(set(events)) != 3:
            raise ValueError('Each condition needs one target and three distinct events')
        for index, event in enumerate(events):
            group = copy.deepcopy(original)
            teacher = copy.deepcopy(teachers[original['teacher_ids'][0]])
            qid = original['question_id'] + '-wording-' + str(index)
            tid = qid + '-' + teacher['latent_sha256'][:16]
            group.update(question_id=qid, teacher_ids=[tid], event_text=event, wording_index=index,
                         parent_transition_group=original['question_id'])
            teacher.update(teacher_id=tid, question_id=qid,
                           parent_transition_teacher=original['teacher_ids'][0])
            result['groups'].append(group)
            result['teachers'].append(teacher)
    result.update(conditional_group_count=45, event_wordings_per_transition=3)
    result['provenance'] = {'parent_transition_bank_sha256': SOURCE_BANK_SHA,
        'parent_provenance': copy.deepcopy(parent['provenance']), 'optimizer_updates': 0,
        'target_and_source_tensors': 'unchanged sealed parent tensors; repeated conditional identities are not independent targets',
        'development_observations': 'CFG1 confirmation clear rewordings failed and early recurrent no-ops lost readable state',
        'scope': 'one entity, one semantic question, three target states, fifteen source/operation transitions, three wordings each'}
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = a.parent_run / 'manifest.json'
    complete = json.loads((a.parent_run / 'complete.json').read_text())
    if (sha(manifest) != SOURCE_BANK_SHA or complete['bank_manifest_sha256'] != SOURCE_BANK_SHA
            or complete['bank_sealed'] is not True):
        raise ValueError('Source bank seal differs')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name or sha(a.parent_run / name) != digest:
            raise ValueError('Source preparation artifact changed: ' + name)
    parent = json.loads(manifest.read_text())
    for teacher in parent['teachers']:
        if sha(Path(teacher['latent_path'])) != teacher['latent_file_sha256']:
            raise ValueError('Target changed')
    result = assemble(parent)
    result['provenance']['parent_complete_sha256'] = sha(a.parent_run / 'complete.json')
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output / 'manifest.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest_sha256': sha(out), 'conditional_groups': 45,
                      'semantic_questions': 1, 'unique_target_tensors': len({t['latent_sha256'] for t in result['teachers']})}))


if __name__ == '__main__':
    main()
