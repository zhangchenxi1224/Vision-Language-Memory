"""K1 official-record adapter; questions never enter the Writer input."""
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
ALIGN = ROOT / 'reports/prefeval-official-alignment-20260923'
REPORT = ROOT / 'reports/prefeval-k1-l0-l2-20260924'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def official_mcq(upstream):
    """Execute the unchanged pure upstream functions, without importing API clients."""
    import re
    from bs4 import BeautifulSoup
    path = Path(upstream) / 'utils/utils_mcq.py'
    names = {'format_options', 'get_mcq_question_format', 'extract_choice'}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(selected) == 3
    env = {'re': re, 'BeautifulSoup': BeautifulSoup}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env

def option_order(pair_id, step):
    # Each 12 updates: 3 train forms x 4 correct positions. Shuffle distractors too.
    position = (step // 3) % 4
    seed = int.from_bytes(hashlib.sha256(f'{pair_id}:{step}:k1'.encode()).digest()[:8], 'big')
    distractors = [1, 2, 3]
    random.Random(seed).shuffle(distractors)
    distractors.insert(position, 0)
    return distractors, position

def load_records(split='pilot'):
    plan = json.loads((ALIGN / 'WRITER_IMPLEMENTATION_SPLIT.json').read_text())
    ids = plan[{'pilot': 'pilot_train_ids', 'train': 'train_ids', 'dev': 'internal_dev_ids'}[split]]
    records = {x['base_pair_id']: x for x in read_rows(ALIGN / 'data/sft-train-10interturn.jsonl.gz')}
    bench = {x['base_pair_id']: x for x in read_rows(ALIGN / 'data/benchmark-disclosures.jsonl.gz') if x['form'] == 'explicit'}
    forms_path = REPORT / ('dev-question-forms.json' if split == 'dev' else 'question-forms.json')
    forms = json.loads(forms_path.read_text()) if forms_path.exists() else {}
    result = []
    for pid in ids:
        row = records[pid]
        row['benchmark'] = bench[pid]
        row['forms'] = forms.get(pid, {'T1': row['query']['content']})
        row['options'] = bench[pid]['evaluation_only']['classification_task_options']
        result.append(row)
    return result

def event_text(exchange):
    assert len(exchange) == 2 and [m['role'] for m in exchange] == ['user', 'assistant']
    return '\n'.join(m['role'] + ': ' + m['content'] for m in exchange)

def make_forms(split='pilot'):
    specs = list(csv.DictReader((REPORT / f'{split}-paraphrase-specs.tsv').open(encoding='utf-8'), delimiter='\t'))
    records = {r['base_pair_id']: r for r in load_records(split)}
    assert set(records) == {s['base_pair_id'] for s in specs}
    forms = {}
    for s in specs:
        pid, obj, verb = s['base_pair_id'], s['request_object'], s['request_action']
        forms[pid] = {
            'T1': records[pid]['query']['content'],
            'T2': f'What would you {verb} as {obj}?',
            'T3': f'Please {verb} {obj}.',
            'O1': f"{s['context']} Which {obj} would you suggest?",
            'O2': f'If I asked you to {verb} {obj}, what would you suggest?',
        }
    # Fix article/quantifier grammar before any OOD readback, without changing T1/T2/T3.
    fixes = {
        'travel_hotel:0052': 'I am choosing a hotel in New York. Which hotel would you suggest for me?',
        'lifestyle_beauty:0038': 'I am considering face masks for a home spa day. Could you suggest a few good ones?',
        'lifestyle_health:0040': 'I am interested in trying yoga for stress relief. Which style or type of class would you suggest for me?',
        'lifestyle_beauty:0020': 'I am choosing foundation and concealer for everyday use. Could you suggest a reliable foundation and concealer?',
        'pet_ownership:0020': 'I am considering a pet for myself. Which pet would you suggest as suitable for me?',
    }
    for pid, question in fixes.items():
        if pid in forms:
            forms[pid]['O1'] = question
    path = REPORT / ('dev-question-forms.json' if split == 'dev' else 'question-forms.json')
    path.write_text(json.dumps(forms, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['pilot','dev'], default='pilot')
    make_forms(parser.parse_args().split)
