"""Fixed historical event augmentation; source/target sampling stays unchanged."""
from collections import Counter
import hashlib
import torch

from scripts.experiments.broader_writer_protocol import prefix_validation, SEED
from scripts.experiments.native_condition_protocol import plan as native_plan
from vision_memory.training.latent_bank_unet import balanced_draw, stable_seed

POLICY = 'historical-nine-expressions-v1'
REFERENCE_COMMIT = '03f8467e5a1201c2dbd9d12484bf2338d7837727'
REFERENCE_RESULT = 'bc9ddf6a5b72fe0a00f2f1fa3c74715a6531ba56fabb8c43cb73ba6e143754bb'


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def render(entity, topic, operation, value, style):
    if operation == 'clear':
        values = (
            f'The saved {topic} preference belonging to {entity} is withdrawn; delete it.',
            f'Unset {entity}\'s {topic} preference. No choice remains active.',
            f'Erase the existing {topic} preference associated with {entity}.',
            f'{entity.capitalize()} no longer has a preferred {topic}. Clear the stored choice.',
            f'Discard the previous {topic} choice for {entity} without selecting a replacement.',
            f'The {topic} choice recorded for {entity} has been cancelled. Forget that choice.',
            f'Delete {entity}\'s remembered {topic} preference and leave it unset.',
            f'Revoke the current {topic} selection belonging to {entity}; retain no active selection.',
        )
    elif operation in ('set', 'overwrite'):
        values = (
            f'Record {value} as {entity}\'s preferred {topic}, superseding any prior choice.',
            f'{entity.capitalize()} now chooses {value} for {topic}. Remember the new choice.',
            f'The saved {topic} choice belonging to {entity} must be {value}.',
            f'From this point onward, {entity} prefers {value} for {topic}; replace the previous entry.',
            f'Save this selection: {entity}, {topic}, {value}. It takes precedence over earlier selections.',
            f'Keep {value} in memory as the latest {topic} preference of {entity}.',
            f'Change {entity}\'s recorded {topic} choice to {value} and discard the earlier choice.',
            f'{value.capitalize()} is the current preferred {topic} for {entity}. Store the updated preference.',
        )
    else:
        raise ValueError('Unsupported historical event operation')
    if not 1 <= style <= 8:
        raise ValueError('Training style must be one of the fixed eight additions')
    return values[style - 1]


def variants(bank):
    # Parse the sealed event stream using the existing strict semantic parser.
    # Values originate in the events, never in a query or the teacher's answer.
    from scripts.probes.historical_fp32_readback import PANEL_SHA
    import json
    from pathlib import Path
    panel_path = Path(__file__).resolve().parents[2] / 'reports/official-alignment-results-20260913/historical-fp32-readback-panel.json'
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == PANEL_SHA
    panel = {target['target_index']: target for target in json.loads(panel_path.read_bytes())['targets']}
    cases = prefix_validation()['cases']
    originals = {case['question_id']: case for case in cases if case['style'] == 'original_prefix'}
    historical = [group for group in bank['groups'] if 'historical_target_index' in group]
    if len(historical) != 16 or set(originals) != {group['question_id'] for group in historical}:
        raise ValueError('Require all sixteen unchanged historical conditions')
    result = {}
    observed = {case['event_text'] for case in cases if case['style'] != 'original_prefix'}
    for group in historical:
        case = originals[group['question_id']]
        target = panel[group['historical_target_index']]
        if case['event_text'] != group['event_text']:
            raise ValueError('Historical training prefix differs from the sealed original')
        texts = [group['event_text']]
        texts += ['\n'.join(render(target['entity'], target['topic'], event['operation'], event['value'], style)
                    for event in case['event_semantics']) for style in range(1, 9)]
        if len(set(texts)) != 9 or set(texts[1:]) & observed:
            raise ValueError('Duplicate training expression or observed validation text included')
        result[group['question_id']] = texts
    return result


def wording_index(seed, draw_index, question_id):
    if draw_index < 0:
        raise ValueError('Negative draw index')
    # Each historical stratum occurs exactly once per 31-draw logical cycle.
    cycle = draw_index // 31
    expression_cycle, offset = divmod(cycle, 9)
    generator = torch.Generator().manual_seed(stable_seed(seed, 'historical-expression-order:' + question_id, expression_cycle))
    return int(torch.randperm(9, generator=generator)[offset])


def plan(bank, commit):
    value = native_plan(bank, commit)
    texts = variants(bank)
    counts = {qid: Counter() for qid in texts}
    for index in range(value['draws']):
        group, _, _, _ = balanced_draw(bank['groups'], SEED, index, sampling_strategy='logical_condition')
        qid = group['question_id']
        if qid in texts:
            counts[qid][str(wording_index(SEED, index, qid))] += 1
    if any(set(count) != {str(index) for index in range(9)} or max(count.values()) - min(count.values()) > 1 for count in counts.values()):
        raise ValueError('Historical expression cycles do not cover all fixed variants evenly')
    value.update(schema='historical-wording-training-comparison/v1',
        reference_commit=REFERENCE_COMMIT, reference_result_sha256=REFERENCE_RESULT,
        training_augmentation={'policy': POLICY, 'events': texts, 'exact_draws_per_expression': counts,
            'condition_inputs': ['current source image', 'selected event text'],
            'selection_metadata_is_writer_input': False},
        budget_change='Same original initialization, fresh AdamW,4832 updates,19328 source/teacher/noise/sigma draws,31 equally weighted logical strata,native condition training and native28 CFG1 inference. Only historical training event expression changes: original plus eight additions.',
        baseline_requirement='Full302-image/3020-raw native baseline must equal03f8467 bitwise. Augmented training embeddings are sealed separately and cannot change inference runtime or original baseline.',
        alignment_difference='Official FM/source-only/native inference unchanged; the explicit native training condition choice follows03f8467.',
        endpoint_selection='Fixed4832. Evaluate all original1510 development and1800 registered functional cells; preserve all failures. Observed-case regression, no fresh holdout or unseen-entity claim.')
    return value
