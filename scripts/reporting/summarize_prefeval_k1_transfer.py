"""Describe pilot transfer errors against teacher loss; no selection or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(training, paired):
    output = {}
    for arm in ['A', 'B']:
        losses = {r['pair_id']: r['last_12_mean_ce']
                  for r in training['records'] if r['arm'] == arm}
        rows = [dict(r, teacher_ce=losses[r['pair_id']])
                for r in paired[arm]['per_preference']]
        assert len(rows) == len(losses) == 64
        assert {r['pair_id'] for r in rows} == set(losses)
        ordered = sorted(rows, key=lambda r: (r['teacher_ce'], r['pair_id']))
        quartiles = []
        for i in range(4):
            group = ordered[i * 16:(i + 1) * 16]
            quartiles.append({
                'quartile_low_to_high_ce': i + 1, 'preferences': len(group),
                'ce_min': group[0]['teacher_ce'], 'ce_max': group[-1]['teacher_ce'],
                'teacher_mcq_correct': sum(r['teacher_correct'] for r in group),
                'student_both_chains_correct': sum(r['student_correct_chains'] == 2 for r in group),
                'student_correct_of_32': sum(r['student_correct_chains'] for r in group),
                'pair_ids': [r['pair_id'] for r in group],
            })
        groups = {}
        for label, selected in [
            ('teacher_correct_student_both_correct', [r for r in rows if r['teacher_correct'] and r['student_correct_chains'] == 2]),
            ('teacher_correct_student_any_error', [r for r in rows if r['teacher_correct'] and r['student_correct_chains'] < 2]),
            ('teacher_wrong', [r for r in rows if not r['teacher_correct']]),
        ]:
            groups[label] = {'preferences': len(selected),
                'median_teacher_ce': statistics.median(r['teacher_ce'] for r in selected) if selected else None,
                'pair_ids': [r['pair_id'] for r in selected]}
        output[arm] = {'loss_quartiles': quartiles, 'transfer_groups': groups}
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-training', type=Path, required=True)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sources = [args.teacher_training, args.paired]
    result = {'analysis': summarize(*(json.loads(path.read_text(encoding='utf-8')) for path in sources)),
        'sources': [{'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
        'scope': 'Post-hoc descriptive analysis of all 64 trained preferences, T1 MCQ only. Quartiles are formed separately within each arm. Two chains are repeated measurements. No causal test, checkpoint selection, data filtering, or OOD analysis.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({arm: {**data, 'loss_quartiles': [{k: v for k, v in q.items() if k != 'pair_ids'} for q in data['loss_quartiles']],
        'transfer_groups': {label: {k: v for k, v in group.items() if k != 'pair_ids'} for label, group in data['transfer_groups'].items()}}
        for arm, data in result['analysis'].items()}, indent=2))
