"""Fixed condition diversity and new validation for the broader official Writer."""
import copy
import hashlib
import json
from pathlib import Path
import re
from scripts.experiments.transition_warm_start_plan import plan as warm_plan
from scripts.probes.transition_validation_plan import ENTITY, plan as transition_plan
from vision_memory.training.latent_bank_unet import stable_seed

SEED = 20260915
PARENT_CHECKPOINT = 'e04a3d5c90abbe37f230d4282129057240f4db4bee5ffe4b92936b294029d053'
PARENT_PACKAGE = 'd1536d754dc3770f3e715b1c0bfaa3278ff61e68b7bad70dce385b3b38f8a2d7'


def additional_events(operation, state):
    # The first two expressions have already been observed in the failed warm
    # confirmation. They explicitly become regression/development training data.
    if operation == 'noop':
        return (
            f'Record that {ENTITY} passed a routine inspection, keeping the saved music preference as it was.',
            f'Inventory information changed for {ENTITY}; preserve its stored music selection.',
            f'Keep the music preference for {ENTITY} unchanged after this routine update.',
            f'This notice concerns maintenance of {ENTITY}; leave its remembered music choice intact.',
            f'No change has been requested to the music preference for {ENTITY}. Retain the existing value.',
            f'A delivery arrived for {ENTITY}. Its saved music selection stays the same.')
    if operation == 'clear':
        return (
            f'Remove the music preference stored for {ENTITY}; discard the previous selection.',
            f"Forget {ENTITY}'s saved music choice so that no music option is selected.",
            f'The music preference for {ENTITY} has been withdrawn. Remove it from memory.',
            f'Clear all saved music choices for {ENTITY}, leaving no active music preference.',
            f'The previous music selection for {ENTITY} is no longer valid. Delete that selection.',
            f'Unset the music preference for {ENTITY}; it should have no stored value.')
    if operation != 'set' or state not in ('ambient', 'jazz'):
        raise ValueError('Unexpected transition semantics')
    return (
        f'Store {state} as the music preference for {ENTITY}, replacing any earlier choice.',
        f"Set {ENTITY}'s current music preference to {state}.",
        f'The music preference for {ENTITY} is now {state}. Save this selection.',
        f'Update the remembered music choice for {ENTITY} to {state}.',
        f'For {ENTITY}, use {state} as the saved music preference from now on.',
        f'{ENTITY.capitalize()} now prefers {state} music; replace the previously stored choice.')


def augment(bank):
    if len(bank['groups']) != 61 or len(bank['teachers']) != 61:
        raise ValueError('Require all45 original transitions and all16 refined questions first')
    result = copy.deepcopy(bank)
    teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
    originals = [group for group in bank['groups'][:45] if group['wording_index'] == 0]
    if len(originals) != 15:
        raise ValueError('Missing original transition bases')
    for original in originals:
        for index, event in enumerate(additional_events(original['operation'], original['target_state']), 3):
            group, teacher = copy.deepcopy(original), copy.deepcopy(teachers[original['teacher_ids'][0]])
            qid = original['question_id'] + '-expanded-' + str(index)
            tid = qid + '-' + teacher['latent_sha256'][:16]
            group.update(question_id=qid, teacher_ids=[tid], event_text=event, wording_index=index,
                wording_exposure='observed_warm_validation_now_training' if index < 5 else 'new_training_expression')
            teacher.update(question_id=qid, teacher_id=tid, parent_transition_teacher=original['teacher_ids'][0])
            result['groups'].append(group)
            result['teachers'].append(teacher)
    result.update(conditional_group_count=151, transition_conditions=135, historical_prefix_conditions=16)
    result['provenance'].update(condition_expansion='All15 source/operation transitions now have9 expressions; first45 groups retained byte-for-byte.',
        prior_validation_exposure='All8 event expressions from1201 warm confirmation now enter training; their old failure remains valid and they cannot count as new holdouts.')
    if len(result['groups']) != 151 or len({group['question_id'] for group in result['groups']}) != 151:
        raise ValueError('Expanded condition coverage differs')
    return result


def fresh_transition_validation():
    value = transition_plan(SEED)
    writes = {state: (
        f'Please remember the following music selection for {ENTITY}: {state}. This supersedes the old selection.',
        f'Going forward, {state} is the preferred music for {ENTITY}; retain that information.') for state in ('ambient', 'jazz')}
    writes['clear'] = (
        f'Cancel the remembered music selection for {ENTITY}. There is no replacement preference.',
        f'The music entry for {ENTITY} should be empty now; remove its previously remembered choice.')
    noops = (
        f'The address record for {ENTITY} was checked. Its remembered music preference must remain unchanged.',
        f'Leave the stored music information for {ENTITY} exactly as before; this update concerns scheduling only.')
    for case in value['single_writes']:
        if case['style'].startswith('new_wording_'):
            case['event_text'] = writes[case['state']][int(case['style'][-1]) - 1]
    for chain in value['rgb_chains']:
        for step in chain['steps']:
            if step['event_style'].startswith('new_wording_'):
                index = int(step['event_style'][-1]) - 1
                step['event_text'] = noops[index] if step['operation'] == 'noop' else writes[step['expected_state']][index]
    return value


def prefix_validation():
    from scripts.probes.historical_fp32_readback import PANEL_SHA
    path = Path(__file__).resolve().parents[2] / 'reports/official-alignment-results-20260913/historical-fp32-readback-panel.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PANEL_SHA:
        raise ValueError('Historical event/query panel changed')
    panel = json.loads(path.read_bytes())
    cases = []
    for target in sorted(panel['targets'], key=lambda target: target['target_index']):
        entity, topic = target['entity'], target['topic']
        parsed = []
        for event in target['event_stream']:
            text = re.sub(r'^R3 Train Standard Templates [0-9]+: ', '', event['event_text'])
            patterns = (
                rf'For {re.escape(entity)}, remember that the preferred {topic} is ([^.]+)\.',
                rf'Save ([^.]+) as the current {topic} preference for {re.escape(entity)}\.',
                rf'Replace the earlier {topic} preference for {re.escape(entity)} with ([^.]+)\.',
                rf'The {topic} preference for {re.escape(entity)} is now ([^.]+), not the previous value\.')
            matches = [match for pattern in patterns if (match := re.fullmatch(pattern, text))]
            if event['event_kind'] in ('set', 'overwrite') and len(matches) == 1:
                parsed.append({'operation': event['event_kind'], 'value': matches[0][1]})
            elif event['event_kind'] == 'clear' and text in (
                    f'Clear the saved {topic} preference for {entity}.',
                    f'Forget the current {topic} choice for {entity}; none is active now.'):
                parsed.append({'operation': 'clear', 'value': 'no active preference'})
            else:
                raise ValueError('Cannot preserve the semantics of a historical event: ' + text)
        if parsed[-1]['value'] != target['gold']:
            raise ValueError('Full ordered prefix does not end in the recorded state')
        for style in range(3):
            events = []
            for source, semantic in zip(target['event_stream'], parsed, strict=True):
                if style == 0:
                    event = source['event_text']
                elif semantic['operation'] == 'clear':
                    event = (f'Remove the saved {topic} selection for {entity}.' if style == 1
                        else f'There is no longer an active {topic} choice for {entity}; erase it.')
                else:
                    value = semantic['value']
                    event = (f'For {entity}, the {topic} preference should now be {value}; store this choice.' if style == 1
                        else f"Update {entity}'s remembered {topic} selection to {value}.")
                events.append(event)
            cases.append({'case': f"target-{target['target_index']:02d}-style-{style}",
                'target_index': target['target_index'], 'question_id': target['semantic_group_id'] + '-historical-prefix',
                'style': 'original_prefix' if style == 0 else f'new_prefix_wording_{style}',
                'event_text': '\n'.join(events), 'event_semantics': parsed,
                'gold': target['gold'], 'question_variants': target['question_variants'],
                'noise_seeds': [stable_seed(SEED, 'broader-prefix-confirmation', target['target_index'] * 12 + style * 4 + i) for i in range(4)]})
    return {'cases': cases, 'matched_images': 192, 'matched_rows': 960,
        'scope': 'One gray-source write per full ordered prefix; all16 targets, original plus two reworded prefixes. Query strings were teacher-training-seen.'}


def training_plan(bank_sha, training_commit):
    return {'schema': 'broader-official-writer/v1', 'training_commit': training_commit,
        'bank_sha256': bank_sha, 'parent_checkpoint_sha256': PARENT_CHECKPOINT, 'initial_package_manifest_sha256': PARENT_PACKAGE,
        'training_seed': SEED, 'optimizer_steps': 4832, 'global_batch': 4, 'draws': 19328,
        'conditional_groups': 151, 'draws_per_group': 128, 'semantic_questions': 17,
        'model_variant': 'base', 'trainable_scope': 'full_unet', 'flow_protocol': 'official',
        'optimizer': {'kind': 'fresh AdamW', 'lr': 5e-5, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-4, 'clip_norm': 1.},
        'inference': {'native_steps': 28, 'guidance_scale': 1., 'pure_gaussian_initialization': True},
        'eval_seeds': 2, 'development_matched_rows_per_phase': 1510, 'development_raw_rows_per_phase': 3020,
        'development_noise_seeds': [stable_seed(SEED, 'heldout-evaluation-noise', i) for i in range(2)],
        'endpoint_selection': 'fixed4832 updates; no best-checkpoint selection',
        'budget_change': '128 draws per expanded condition, versus256 in the45-condition experiment. This increases total draws while sharing budget across more expressions; not an equal-exposure ablation.',
        'target_queries': 'Allfive historical queries supervised during teacher construction; not untouched holdouts.',
        'transition_validation': fresh_transition_validation(),
        'prefix_validation': prefix_validation(),
        'scope': 'Broader seen-question full-prefix writes plus continuous transitions for one music entity. Does not establish unseen entities or simultaneous multi-fact retention.'}
