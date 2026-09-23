"""Describe frozen T1 free outputs without treating output changes as adherence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def summarize(snapshot):
    result = {}
    for arm in ['A', 'B']:
        source = snapshot / arm / 'readback-0.jsonl'
        payload = source.read_bytes()
        rows = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
        by_id = defaultdict(dict)
        for row in rows:
            assert row['family'] == 'T1' and row['task'] == 'free'
            assert row['chain'] == row['prefix'] == 0
            assert row['control'] not in by_id[row['pair_id']]
            by_id[row['pair_id']][row['control']] = row
        pairs = []
        for pid, controls in sorted(by_id.items()):
            memory, mismatch = [controls[k] for k in ['memory', 'mismatch']]
            assert memory['reader_query'] == mismatch['reader_query']
            assert memory['png_sha256'] != mismatch['png_sha256']
            first, second = [r['generated'] for r in [memory, mismatch]]
            shared = 0
            for x, y in zip(first['generated_token_ids'], second['generated_token_ids']):
                if x != y:
                    break
                shared += 1
            pairs.append({'pair_id': pid, 'raw_exact_match': first['raw'] == second['raw'],
                          'common_initial_tokens': shared})
        result[arm] = {
            'expected_preferences': 64, 'observed_paired_preferences': len(pairs),
            'complete': len(pairs) == 64,
            'exact_same_raw_answers': sum(p['raw_exact_match'] for p in pairs),
            'median_common_initial_tokens': statistics.median(p['common_initial_tokens'] for p in pairs),
            'controls': {control: {
                'expected': 64, 'observed': sum(r['control'] == control for r in rows),
                'truncated': sum(r['generated']['truncated'] for r in rows if r['control'] == control),
            } for control in sorted({r['control'] for r in rows})},
            'source_sha256': hashlib.sha256(payload).hexdigest(), 'pairs': pairs,
        }
    return {'arms': result, 'scope': 'Frozen teacher T1 free outputs only; A snapshot is partial. '
            'Exact output differences measure image sensitivity, not correct preference retrieval. '
            'No judge accuracy, OOD selection, or decoding-budget change.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.snapshot)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({arm: {k: v for k, v in data.items() if k != 'pairs'} for arm, data in result['arms'].items()}))
