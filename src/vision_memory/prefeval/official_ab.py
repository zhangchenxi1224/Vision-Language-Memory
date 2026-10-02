"""Paired official-task supervision; no model-dependent sample selection."""
from __future__ import annotations
import gzip
import hashlib
import json
import random
from pathlib import Path

SEED = 20260924
TRAIN_FORMS = ('T1', 'T2', 'T3')
EVAL_FORMS = ('T1', 'T2', 'T3', 'O1', 'O2')
# Verbatim released get_mcq_question_format layout (PrefEval 50795054).
MCQ_TEMPLATE = '''
    I'm trying to decide on this and here are 4 options for my query: \n{options}\nNow, I'd like you to pick one of them as your top recommendation for me.
    Important instructions for your response:
    1. Choose only one option (A, B, C, or D) that best matches my preferences.
    2. Your answer must be one of these options.
    3. Don't say things like "I can't choose" or suggest alternatives not listed.
    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:
       <choice>[A/B/C/D]</choice>
    '''

def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_for(*values):
    return int(hashlib.sha256(json.dumps(values).encode()).hexdigest()[:15], 16)

def records(report):
    report = Path(report)
    split = json.loads((report / 'WRITER_IMPLEMENTATION_SPLIT.json').read_text())
    ids = set(split['pilot_train_ids'] + split['internal_dev_ids'])
    benchmark = {r['base_pair_id']: r for r in read_gz(report / 'data/benchmark-disclosures.jsonl.gz')
                 if r['form'] == 'explicit' and r['base_pair_id'] in ids}
    result = []
    for r in read_gz(report / 'data/sft-train-10interturn.jsonl.gz'):
        key = r['base_pair_id']
        if key not in ids:
            continue
        b = benchmark[key]
        result.append(dict(id=key, split='train' if key in split['pilot_train_ids'] else 'dev',
            question=r['query']['content'], answer=r['target']['content'], history=r['history'],
            options=b['evaluation_only']['classification_task_options'],
            benchmark=b, source=r['source']))
    assert len(result) == 154 and sum(r['split'] == 'train' for r in result) == 64
    return sorted(result, key=lambda r: r['id'])

def option_order(record, step, *, namespace='training'):
    # Each form occurs at each correct-answer position once every 12 updates.
    correct = (step // 3) % 4
    wrong = [1, 2, 3]
    random.Random(seed_for(SEED, namespace, record['id'], step)).shuffle(wrong)
    wrong.insert(correct, 0)
    return wrong, correct

def mcq(record, question, step, *, namespace='training'):
    order, gold = option_order(record, step, namespace=namespace)
    options = '\n'.join(f'{chr(65+i)}. {record["options"][j]}' for i, j in enumerate(order))
    return question + MCQ_TEMPLATE.format(options=options), f'<choice>{chr(65+gold)}</choice>', order

def training_item(record, forms, arm, step):
    if record['split'] != 'train':
        raise ValueError('No latent optimization on dev/evaluation examples')
    family = TRAIN_FORMS[step % 3]
    question = forms[family]
    if arm == 'A':
        return dict(family=family, query=question, target=record['answer'], order=None)
    if arm != 'B':
        raise ValueError(arm)
    query, target, order = mcq(record, question, step)
    return dict(family=family, query=query, target=target, order=order)

def writer_event(record, index=0):
    history = record['history']
    return '\n'.join(f'{m["role"]}: {m["content"]}' for m in history[2*index:2*index+2])

def initial_ack_event(record, reader_history, variant):
    """Two train-only initial exchanges, sharing the unchanged user disclosure."""
    if record['split'] != 'train' or variant not in (0, 1):
        raise ValueError('Acknowledgment augmentation is restricted to training states')
    original = record['history'][:2]
    candidate = reader_history[:2]
    if (len(original) != 2 or len(candidate) != 2 or
            [x['role'] for x in candidate] != ['user', 'assistant'] or
            candidate[0] != original[0]):
        raise ValueError('Require identical disclosure and one complete acknowledgment')
    return writer_event(dict(history=original if variant == 0 else candidate))

def validate_forms(question, forms):
    if set(forms) != set(EVAL_FORMS) or forms['T1'] != question:
        raise ValueError('Require unchanged T1 and exactly four declared paraphrases')
    if any(not isinstance(q, str) or len(q.strip()) < 8 for q in forms.values()):
        raise ValueError('Missing question')
    if len(set(q.strip().casefold() for q in forms.values())) != 5:
        raise ValueError('Duplicate question forms')
    return forms
