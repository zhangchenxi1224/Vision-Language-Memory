"""Validate the fixed historical condition encoding seal and every selected draw."""
from scripts.experiments.historical_wording_protocol import POLICY, text_sha, wording_index
from scripts.experiments.broader_writer_protocol import SEED


def identity_binding(registered):
    value = registered['training_augmentation']
    if value['policy'] != POLICY:
        raise ValueError('Unknown historical training augmentation')
    return {'policy': POLICY,
        'event_text_sha256': {qid: [text_sha(event) for event in events] for qid, events in value['events'].items()},
        'writer_input': False}


def verify_seal(bank, identity, runtime, seal, registered):
    expected = identity_binding(registered)
    historical = {group['question_id'] for group in bank['groups'] if 'historical_target_index' in group}
    if (identity.get('training_augmentation') != expected or set(seal) != historical
            or set(expected['event_text_sha256']) != historical or len(historical) != 16):
        raise ValueError('Historical training identity or complete encoding coverage differs')
    for qid, entries in seal.items():
        if len(entries) != 9:
            raise ValueError('Require all nine encodings for each historical condition')
        for index, entry in enumerate(entries):
            if (set(entry) != {'index', 'event_text_sha256', 'prompt_embeds_sha256', 'attention_mask_sha256'}
                    or entry['index'] != index or entry['event_text_sha256'] != expected['event_text_sha256'][qid][index]):
                raise ValueError('Condition encoding is not bound to the fixed event expression')
            for key in ('prompt_embeds_sha256', 'attention_mask_sha256'):
                digest = entry[key]
                if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
                    raise ValueError('Malformed actual condition tensor digest')
        if entries[0]['prompt_embeds_sha256'] != runtime['condition_sha256'][qid]:
            raise ValueError('Original historical training encoding changed')
    return expected


def verify_draw(group, draw, draw_index, seal):
    if 'historical_target_index' in group:
        qid = group['question_id']
        index = wording_index(SEED, draw_index, qid)
        if draw.get('training_condition_variant') != seal[qid][index]:
            raise ValueError('Actual historical draw used a different expression or encoding')
        return qid, str(index)
    if 'training_condition_variant' in draw:
        raise ValueError('Music training condition was augmented unexpectedly')
    return None
