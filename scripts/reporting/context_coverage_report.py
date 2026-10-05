"""Paired teacher-only MCQ report. No promotion or generic free-answer claims."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import official_mcq


def paired_interval(values):
    rng = random.Random(20261006)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000))
    return {'mean': sum(values)/len(values), 'ci95': [means[249], means[9749]], 'independent_n': len(values)}


def read_checked(paths, ids, controls=None):
    extract = official_mcq(ROOT/'third_party/prefeval_reference')['extract_choice']
    table, checked = {}, {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['pair_id'] not in ids or (controls and row['control'] not in controls):
                continue
            key = tuple(row[k] for k in ('pair_id','chain','prefix','control','family','task'))
            if key in table:
                raise ValueError('Duplicate readback key')
            pred = extract(row['generated']['raw'])
            if pred != row['predicted_letter'] or (pred == row['correct_letter']) != row['correct']:
                raise ValueError('Saved score disagrees with raw-answer parser')
            png = row.get('png_path')
            if png:
                if png not in checked:
                    checked[png] = hashlib.sha256(Path(png).read_bytes()).hexdigest()
                if checked[png] != row['png_sha256']:
                    raise ValueError('PNG bytes changed')
            table[key] = row
    return table


def summarize(table, ids):
    expected = {(pid,0,0,c,f,'mcq') for pid in ids for c in ('memory','blank','mismatch','text')
                for f in ('T1','T2','T3','O1','O2')}
    if set(table) != expected:
        raise ValueError(f'Incomplete denominator: expected {len(expected)}, got {len(table)}')
    values = {c: [int(all(table[pid,0,0,c,f,'mcq']['correct'] for f in ('O1','O2'))) for pid in ids]
              for c in ('memory','blank','mismatch','text')}
    return {'rows': len(table), 'joint_o1_o2': {c: sum(v)/len(v) for c,v in values.items()},
            'memory_minus_mismatch': paired_interval([a-b for a,b in zip(values['memory'],values['mismatch'])]),
            'memory_minus_blank': paired_interval([a-b for a,b in zip(values['memory'],values['blank'])]),
            'truncations': sum(bool(r['generated'].get('truncated')) for r in table.values()),
            'parse_failures': sum(bool(r.get('parse_failure')) for r in table.values())}, values


def main(args):
    ids = json.loads(args.ids_file.read_text())['ids']
    result = {'schema':'dreamlite.context-coverage-report.v1', 'ids':ids, 'arms':{}, 'contrasts':{},
              'scope':'teacher on training histories; MCQ only; no Writer/dev/retention/free-judge conclusion',
              'promotion':'not_authorized_by_this_metric'}
    vectors = {}
    for mode in ('history_hard','prompt_matching'):
        for pid in ids:
            name = pid.replace(':','_')
            old_done = json.loads((args.reference/mode/'teachers'/name/'complete.json').read_text())
            new_done = json.loads((args.run/mode/'teachers'/name/'complete.json').read_text())
            for key in ('arm','steps','lr','supervision','temperature','reader_weights_sha256',
                        'reader_config_sha256','target_pipeline_sha256','reader_objective_sha256',
                        'mcq_source_sha256','teacher_max_new_tokens','quantization'):
                if old_done['binding'][key] != new_done['binding'][key]:
                    raise ValueError('Unpaired teacher setting: ' + key)
        old = read_checked(sorted((args.reference/mode/'teacher-readback').glob('readback-*.jsonl')), ids,
                           controls={'memory','blank','text'})
        old.update(read_checked(sorted((args.run/mode/'reference-mismatch').glob('readback-*.jsonl')), ids))
        label = 'original/' + mode
        result['arms'][label], vectors[label] = summarize(old, ids)
        for endpoint in ('teachers','selected'):
            label = 'diverse/' + mode + '/' + endpoint
            table = read_checked(sorted((args.run/mode/('readback-'+endpoint)).glob('readback-*.jsonl')), ids)
            result['arms'][label], vectors[label] = summarize(table, ids)
    for mode in ('history_hard','prompt_matching'):
        original, final, selected = [vectors[k]['memory'] for k in
            ('original/'+mode,'diverse/'+mode+'/teachers','diverse/'+mode+'/selected')]
        result['contrasts'][mode+'/coverage_at_final'] = paired_interval([a-b for a,b in zip(final,original)])
        result['contrasts'][mode+'/selection_vs_final'] = paired_interval([a-b for a,b in zip(selected,final)])
    result['contrasts']['soft_minus_hard_diverse_final'] = paired_interval([a-b for a,b in zip(
        vectors['diverse/prompt_matching/teachers']['memory'], vectors['diverse/history_hard/teachers']['memory'])])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','reference','ids-file','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
