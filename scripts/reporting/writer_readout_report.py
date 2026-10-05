"""Complete-denominator, preference-paired report for shared Writer readout."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_writer_readout import ENDPOINTS, expected_keys, cohort_rows, read
from scripts.experiments.prefeval_context_readout import validate_protocol
from scripts.experiments.prefeval_k1_data import sha
from scripts.reporting.context_coverage_report import paired_interval


def summarize(rows, cohorts, spec):
    validate_protocol(spec)
    ids = [pid for group in cohorts.values() for pid in group]
    table = {}
    target_bindings = {}
    for row in rows:
        key = tuple(row[k] for k in ('pair_id', 'query_id', 'endpoint', 'control', 'chain'))
        if key in table or not math.isfinite(row['kl']):
            raise ValueError('Duplicate or nonfinite row')
        table[key] = row
        pair = key[:2]
        binding = (tuple(row['target_ids']), row['teacher_logits_sha256'])
        if pair in target_bindings and target_bindings[pair] != binding:
            raise ValueError('Unpaired teacher prefix/distribution')
        target_bindings[pair] = binding
    if set(table) != expected_keys(ids, spec['queries']):
        raise ValueError('Incomplete readout denominator')
    families = defaultdict(list)
    for q in spec['queries']:
        families[q['family']].append(q['id'])
    result = {'rows': len(rows), 'metric': 'Conditional teacher-prefix token KL; not free-generation accuracy',
              'scope': 'Exploratory existing Writer comparison; internal dev exposed to prior research; no default promotion',
              'cohorts': {}}
    for split, group in cohorts.items():
        out = {'independent_n': len(group), 'families': {}}
        for family, qids in families.items():
            def values(e, c):
                chains = [0] if c in ('text', 'blank') else [0, 1]
                return [sum(table[pid, q, e, c, chain]['kl'] for q in qids for chain in chains)
                        / (len(qids) * len(chains)) for pid in group]
            baseline = values('b730', 'memory')
            blank, text = values('blank', 'blank'), values('text', 'text')
            current = {'blank_kl': sum(blank) / len(group), 'text_self_consistency_kl': sum(text) / len(group), 'endpoints': {}}
            for e in ENDPOINTS:
                mem, wrong = values(e, 'memory'), values(e, 'mismatch')
                current['endpoints'][e] = {'memory_kl': sum(mem) / len(group), 'mismatch_kl': sum(wrong) / len(group),
                    'baseline_minus_memory': paired_interval([a-b for a,b in zip(baseline,mem)]),
                    'mismatch_minus_memory': paired_interval([a-b for a,b in zip(wrong,mem)]),
                    'blank_minus_memory': paired_interval([a-b for a,b in zip(blank,mem)])}
            out['families'][family] = current
        result['cohorts'][split] = out
    return result


def report(run, protocol):
    spec = read(protocol)
    rows = []
    for shard in range(2):
        done = read(run / f'finished-{shard}.json')
        path, ident = run / f'readout-{shard}.jsonl', run / f'identity-{shard}.json'
        if done['readout_sha256'] != sha(path) or done['identity_sha256'] != sha(ident):
            raise ValueError('Readout receipt mismatch')
        identity = read(ident)
        if identity['protocol_sha256'] != sha(protocol):
            raise ValueError('Query protocol changed')
        values = [json.loads(s) for s in path.read_text().splitlines()]
        if len(values) != done['rows'] or {r['pair_id'] for r in values} != set(identity['assignment']):
            raise ValueError('Shard denominator mismatch')
        for asset in identity['assets'].values():
            if sha(asset['path']) != asset['sha256']:
                raise ValueError('PNG changed')
        rows.extend(values)
    groups = {s: [r['base_pair_id'] for r in rs] for s,rs in cohort_rows().items()}
    result = summarize(rows, groups, spec)
    (run / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps({'rows': report(args.run, args.protocol)['rows']}))
