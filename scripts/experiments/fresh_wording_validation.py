"""Freeze new event wording and noise before observing the b9 training endpoint."""
import copy
import hashlib
import json

from scripts.experiments.broader_writer_protocol import SEED
from scripts.probes.transition_validation_plan import ENTITY
from vision_memory.training.latent_bank_unet import stable_seed

VALIDATION_SET = 'fresh_wording_v1'
PARENT_COMMIT = 'b9f90e956eea7bda15f638c8877919941ce4fec5'
EXPECTED_PLAN_SHA = 'ba77e8c2ab8742bdcda703ae12071b504ea99c33173d0ebad314ddc10292fb6b'
NATIVE_BASELINE_COMMIT = '03f8467e5a1201c2dbd9d12484bf2338d7837727'


def event(entity, topic, operation, value, style):
    if operation == 'noop':
        options = (
            f'Only the delivery date for {entity} changed. Its remembered {topic} preference is still valid.',
            f'This update concerns {entity}\'s stock record, so carry its existing {topic} preference forward.',
            f'Preserve whichever {topic} choice {entity} already has; the notice concerns an unrelated repair.',
        )
    elif operation == 'clear':
        options = (
            f'Withdraw {entity}\'s {topic} choice entirely, leaving no preferred option on record.',
            f'{entity.capitalize()} has opted out of its {topic} preference. Remove the remembered selection.',
            f'There should be no preferred {topic} assigned to {entity} anymore; discard the saved value.',
        )
    elif operation in ('set', 'overwrite'):
        options = (
            f'{entity.capitalize()} has selected {value} for {topic}. Treat this as its latest preference.',
            f'For the record, {entity}\'s preferred {topic} has changed to {value}; keep the latest value.',
            f'Apply this new preference to {entity}: {value} for {topic}, superseding the choice on record.',
        )
    else:
        raise ValueError('Unsupported fresh validation operation')
    if style not in range(3):
        raise ValueError('Fresh validation requires one of three fixed styles')
    return options[style]


def all_events(value):
    items = [case.get('event_text') for case in value['transition_validation']['single_writes']]
    items += [step.get('event_text') for sequence in value['transition_validation']['rgb_chains'] for step in sequence['steps']]
    items += [case['event_text'] for case in value['prefix_validation']['cases']]
    return {text for text in items if text is not None}


def all_seeds(value):
    seeds = {seed for case in value['transition_validation']['single_writes'] for seed in case['noise_seeds']}
    seeds.update(step['noise_seed'] for sequence in value['transition_validation']['rgb_chains'] for step in sequence['steps'])
    seeds.update(seed for case in value['prefix_validation']['cases'] for seed in case['noise_seeds'])
    return seeds


def plan(training, bank):
    if training['training_commit'] == NATIVE_BASELINE_COMMIT:
        from scripts.experiments.native_condition_protocol import plan as native_plan
        from scripts.experiments.historical_wording_protocol import plan as wording_plan
        if training != native_plan(bank, NATIVE_BASELINE_COMMIT):
            raise ValueError('Native baseline differs from its fixed training registration')
        original = plan(wording_plan(bank, PARENT_COMMIT), bank)
        value = copy.deepcopy(training)
        for key in ('transition_validation', 'prefix_validation', 'validation_set', 'validation_scope'):
            value[key] = copy.deepcopy(original[key])
        value['validation_exposure'] = 'Previously observed fresh_wording_v1 cases and noise reused in full to measure the fixed 03 parent baseline; not a new holdout.'
        value['reused_validation_plan_sha256'] = EXPECTED_PLAN_SHA
        value['diagnostic_purpose'] = 'Measure all original observed-expression cells on the unchanged 03 endpoint before attributing 4f errors to continuation training.'
        return value
    continuation_commit = '4fbc85725d78427235757ace2661d086b896a97f'
    generated_source_commit = 'b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36'
    if training['training_commit'] in (continuation_commit, generated_source_commit):
        if training['training_commit'] == generated_source_commit:
            from scripts.experiments.generated_source_training_protocol import plan as continuation_plan
        else:
            from scripts.experiments.clear_retention_protocol import plan as continuation_plan
        from scripts.experiments.historical_wording_protocol import plan as wording_plan
        if training != continuation_plan(bank, training['training_commit']):
            raise ValueError('Continuation differs from its fixed training registration')
        # Reuse exactly the fully sealed b9 validation cases and noise. Their
        # prior observation must remain explicit when selecting this repair.
        original = plan(wording_plan(bank, PARENT_COMMIT), bank)
        value = copy.deepcopy(training)
        for key in ('transition_validation', 'prefix_validation', 'validation_set', 'validation_scope'):
            value[key] = copy.deepcopy(original[key])
        value['validation_exposure'] = 'Previously observed fresh_wording_v1 cases and noise reused in full as continuation regression tests; not a new holdout.'
        value['reused_validation_plan_sha256'] = EXPECTED_PLAN_SHA
        return value
    if training['training_commit'] != PARENT_COMMIT:
        raise ValueError('Fresh validation is fixed to the b9 training endpoint')
    from pathlib import Path
    from scripts.probes.historical_fp32_readback import PANEL_SHA
    panel_path = Path(__file__).resolve().parents[2] / 'reports/official-alignment-results-20260913/historical-fp32-readback-panel.json'
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != PANEL_SHA:
        raise ValueError('Historical event provenance changed')
    targets = {target['target_index']: target for target in json.loads(panel_path.read_bytes())['targets']}
    value = copy.deepcopy(training)
    single_seeds = [stable_seed(SEED, 'b9-fresh-wording-single-v1', index) for index in range(16)]
    for index, case in enumerate(value['transition_validation']['single_writes']):
        style = index % 3
        case['style'] = f'fresh_wording_{style}'
        operation = 'clear' if case['state'] == 'clear' else 'set'
        case['event_text'] = event(ENTITY, 'music', operation, case['state'], style)
        case['noise_seeds'] = single_seeds if style == 0 else single_seeds[:4]
    for sequence in value['transition_validation']['rgb_chains']:
        style = sequence['repetition'] % 3
        for step in sequence['steps']:
            operation = 'noop' if step['operation'] == 'noop' else ('clear' if step['expected_state'] == 'clear' else 'set')
            step['event_style'] = f'fresh_wording_{style}'
            step['event_text'] = event(ENTITY, 'music', operation, step['expected_state'], style)
            step['noise_seed'] = stable_seed(SEED, 'b9-fresh-wording-chain-v1', sequence['repetition'] * 6 + step['step'])
    for case in value['prefix_validation']['cases']:
        style = int(case['case'].rsplit('-', 1)[1])
        target = targets[case['target_index']]
        case['case'] = f"fresh-target-{case['target_index']:02d}-style-{style}"
        case['style'] = f'fresh_prefix_wording_{style}'
        case['event_text'] = '\n'.join(event(target['entity'], target['topic'], item['operation'], item['value'], style)
                                        for item in case['event_semantics'])
        case['noise_seeds'] = [stable_seed(SEED, 'b9-fresh-wording-prefix-v1', case['target_index'] * 12 + style * 4 + index)
                               for index in range(4)]
    forbidden_events = all_events(training) | {group['event_text'] for group in bank['groups']}
    forbidden_events.update(text for texts in training['training_augmentation']['events'].values() for text in texts)
    if all_events(value) & forbidden_events:
        raise ValueError('Fresh event wording overlaps training or observed validation text')
    forbidden_seeds = all_seeds(training) | {stable_seed(SEED, 'training-noise', index) for index in range(training['draws'])}
    forbidden_seeds.update(stable_seed(SEED, 'heldout-evaluation-noise', index) for index in range(2))
    if all_seeds(value) & forbidden_seeds or len(all_seeds(value)) != 16 + 24 + 192:
        raise ValueError('Fresh noise stream overlaps prior exposure or has unexpected collisions')
    value['validation_exposure'] = 'All event wording and noise are new to this training and its prior functional matrices. The semantic questions, target states and five Reader queries remain seen.'
    value['validation_set'] = VALIDATION_SET
    value['validation_scope'] = 'Complete1990 raw rows,1800 matched answers,360 generated images; all16 historical targets and16 actual six-write RGB chains. Not unseen entities or simultaneous multi-fact memory.'
    if digest(value) != EXPECTED_PLAN_SHA:
        raise ValueError('Fresh validation differs from the sealed pre-endpoint plan')
    return value


def digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
