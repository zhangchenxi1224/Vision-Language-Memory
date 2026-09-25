"""Recompute complete position and paired C/I pilot statistics from archived rows."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = ROOT / 'runs/ci64-full-pilot-20260926-0042'
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records, official_mcq

parse = official_mcq(ROOT / 'third_party/prefeval_reference')['extract_choice']
ids = [r['base_pair_id'] for r in load_records('pilot')]
prefixes = [0, 1, 5, 10]
families = ['T1', 'T2', 'T3', 'O1', 'O2']
main = {}
positions = {}
for arm, lane in [('C', 'canonical64'), ('I', 'identity64')]:
    rows = []
    for directory in [RAW / lane / 'readback-pilot-T1', RAW / 'readback-queue' / (lane + '-pilot')]:
        rows += [json.loads(s) for s in (directory / 'readback-0.jsonl').read_text(encoding='utf8').splitlines()]
    index = {tuple(r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family']): r for r in rows}
    assert len(index) == len(rows) == 6720
    main[arm] = index
    positions[arm] = {}
    for prefix in [0, 10]:
        path = RAW / f'readback-queue/{lane}-pilot-positions-{prefix}.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
        idx = {(r['pair_id'], r['chain'], r['position']): r for r in rows}
        assert len(idx) == len(rows) == 640
        assert set(idx) == {(pid, c, p) for pid in ids for c in range(2) for p in ['official', 0, 1, 2, 3]}
        for row in rows:
            pred = parse(row['generated']['raw'])
            assert pred == row['predicted_letter']
            assert row['correct_letter'] == 'ABCD'[row['order'].index(0)]
            assert row['correct'] == (pred == row['correct_letter'])
            assert row['prefix'] == prefix
            assert row['png_sha256'] == index[row['pair_id'], row['chain'], prefix, 'memory', 'T1']['png_sha256']
        joint = lambda pid, c: all(idx[pid, c, p]['correct'] for p in range(4))
        semantic = lambda row: row['order']['ABCD'.index(row['predicted_letter'])] if row['predicted_letter'] in list('ABCD') else None
        positions[arm][prefix] = {
            'rows': 640,
            'position_correct_of_128': {str(p): sum(idx[pid, c, p]['correct'] for pid in ids for c in range(2)) for p in ['official', 0, 1, 2, 3]},
            'four_positions_correct_of_128': sum(joint(pid, c) for pid in ids for c in range(2)),
            'four_positions_two_noises_correct_of_64': sum(all(joint(pid, c) for c in range(2)) for pid in ids),
            'same_semantic_option_of_128': sum(len({semantic(idx[pid, c, p]) for p in range(4)}) == 1 and semantic(idx[pid, c, 0]) is not None for pid in ids for c in range(2)),
            'parse_failures': sum(parse(r['generated']['raw']) is None for r in rows),
            'truncations': sum(r['generated']['truncated'] for r in rows),
        }

paired = []
joint_initial = []
for family in families:
    for prefix in prefixes:
        values = [(main['C'][pid, c, prefix, 'memory', family]['correct'], main['I'][pid, c, prefix, 'memory', family]['correct']) for pid in ids for c in range(2)]
        paired.append({'family': family, 'prefix': prefix, 'total': 128,
            'I_correct_C_wrong': sum(i and not c for c, i in values),
            'C_correct_I_wrong': sum(c and not i for c, i in values)})
        cohort = [(pid, c) for pid in ids for c in range(2) if all(main[a][pid, c, 0, 'memory', family]['correct'] for a in ['C', 'I'])]
        joint_initial.append({'family': family, 'prefix': prefix, 'total': len(cohort),
            **{a: sum(main[a][pid, c, prefix, 'memory', family]['correct'] for pid, c in cohort) for a in ['C', 'I']}})

out = {'positions': positions, 'paired_C_I': paired, 'conditional_on_both_initially_correct': joint_initial,
       'note': 'Full denominators are primary; two seeds and five forms are repeated measurements, not independent preferences.'}
(HERE / 'paired-and-positions.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf8')
print(json.dumps(out))
