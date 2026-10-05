"""Training-side question coverage; no evaluation labels or held-out forms.

The weights are a registered DreamLite hypothesis, not an Image Hijacks recipe.
"""
from collections import Counter
import hashlib
import json

SCHEMA = 'dreamlite.context-coverage.v1'
RECALL = (
    'What preference did the user express? State it and any conditions in at most two sentences.',
    'Recover the user-specific preference from memory, including qualifications. Use at most two sentences.',
    'Summarize what should be remembered about this user in at most two sentences.',
    'Which user preference should guide future assistance? Include any exceptions; use at most two sentences.',
)
NEUTRAL = (
    'Compute 17 + 26. Return only the number.',
    'Translate the word "sunrise" into French. Return only the translation.',
    'Sort the numbers 8, 3, 11 in ascending order. Return only the sorted numbers.',
    'How many sides does a triangle have? Answer briefly.',
)
VALIDATION = (
    ('recall', 'List the user preference and its stated limitations as a compact factual note. Do not invent details.'),
    ('recall', 'What did you learn about the user that remains relevant? Answer with one short sentence.'),
    ('application', 'Describe one recommendation that would respect the remembered preference and briefly explain why.'),
    ('application', 'Describe one recommendation that should be avoided given this user preference, and briefly explain why.'),
    ('neutral', 'Compute 14 + 19. Return only the number.'),
    ('neutral', 'Translate the word "moon" into French. Return only the translation.'),
)


def training_context(row, step, mcq):
    """12-step cycle: 4 recall, 4 MCQ application, 2 open application, 2 neutral.

    MCQ draw index advances only on MCQ slots, retaining all T1-T3/position pairs.
    Returns query, family, order. Targets always come from the history teacher.
    """
    from scripts.experiments.prefeval_k1_data import option_order
    slot, cycle = step % 12, step // 12
    if slot < 4:
        return RECALL[slot], 'recall', None
    if slot < 8:
        draw = cycle * 4 + slot - 4
        order, _ = option_order(row['base_pair_id'], draw)
        query = row['forms'][f'T{draw % 3 + 1}']
        return query + mcq['get_mcq_question_format']([row['options'][i] for i in order]), 'application_mcq', order
    if slot < 10:
        draw = cycle * 2 + slot - 8
        query = row['forms'][f'T{draw % 3 + 1}']
        return query + ' Answer in at most two sentences without a list of options.', 'application_open', None
    return NEUTRAL[(cycle * 2 + slot - 10) % len(NEUTRAL)], 'neutral', None


def context_manifest(rows, steps, mcq):
    entries = {r['base_pair_id']: [training_context(r, s, mcq) for s in range(steps)] for r in rows}
    return {'schema': SCHEMA, 'schedule': entries, 'validation': VALIDATION,
            'weights_per_12_updates': {'recall': 4, 'application_mcq': 4, 'application_open': 2, 'neutral': 2},
            'teacher_history': 'initial observed exchange only',
            'selection': 'equal-family mean token KL on six disjoint validation queries; earliest tie',
            'heldout_scope': 'query templates within trained history; not unseen-history generalization',
            'sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}


def family_mean(scores):
    if len(scores) != len(VALIDATION):
        raise ValueError('Incomplete validation denominator')
    groups = Counter(f for f, _ in VALIDATION)
    return sum(sum(v for (f, _), v in zip(VALIDATION, scores) if f == family) / count
               for family, count in groups.items()) / len(groups)


def choose_checkpoint(results, expected_steps):
    import math
    if sorted(r['step'] for r in results) != sorted(expected_steps):
        raise ValueError('Missing or duplicate candidate checkpoint')
    if any(not math.isfinite(r['score']) for r in results):
        raise ValueError('Nonfinite validation score')
    return min(results, key=lambda r: (r['score'], r['step']))
