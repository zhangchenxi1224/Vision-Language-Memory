"""Describe exact preference-text exposure without changing official splits."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import ALIGN, REPORT, load_records, official_eval_disclosures, sha


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    splits = {name: load_records(name) for name in ['pilot', 'train', 'dev']}
    splits['official'] = [{'base_pair_id': r['base_pair_id'], 'topic': r['topic'],
                          'history': r['input']['disclosure'], 'query': r['input']['query']}
                         for r in official_eval_disclosures()]
    comparisons = []
    for left, right in [('pilot', 'dev'), ('train', 'dev'), ('pilot', 'official'), ('train', 'official')]:
        index = defaultdict(list)
        for row in splits[left]:
            index[normalized(row['history'][0]['content'])].append(row)
        overlaps = []
        for row in splits[right]:
            matches = index.get(normalized(row['history'][0]['content']), [])
            if matches:
                overlaps.append({'left_ids': [r['base_pair_id'] for r in matches],
                    'right_id': row['base_pair_id'], 'preference': row['history'][0]['content'],
                    'exact_raw_text_match': any(r['history'][0]['content'] == row['history'][0]['content'] for r in matches),
                    'same_normalized_T1_question': any(normalized(r['query']['content']) == normalized(row['query']['content']) for r in matches)})
        comparisons.append({'left': left, 'right': right, 'left_n': len(splits[left]), 'right_n': len(splits[right]),
            'shared_ids': sorted({r['base_pair_id'] for r in splits[left]} & {r['base_pair_id'] for r in splits[right]}),
            'shared_topics': sorted({r['topic'] for r in splits[left]} & {r['topic'] for r in splits[right]}),
            'overlapping_preferences': len(overlaps), 'overlaps': overlaps})
    sources = [ALIGN/'WRITER_IMPLEMENTATION_SPLIT.json', ALIGN/'data/sft-train-10interturn.jsonl.gz',
               ALIGN/'data/benchmark-disclosures.jsonl.gz']
    result = {'normalization': 'Unicode NFKC, casefold, whitespace collapse; no semantic matching',
        'comparisons': comparisons, 'sources': [{'file': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sources],
        'interpretation': 'No splits, targets, or scores changed. An exact-text nonmatch does not establish semantic novelty. Full 730 training is prepared, not yet executed. Official 180 remains the primary denominator; exposure strata, if reported after full training, are supplementary.'}
    output = REPORT/'evidence/preference-text-overlap.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
