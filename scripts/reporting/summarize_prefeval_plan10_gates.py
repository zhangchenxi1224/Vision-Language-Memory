"""Same-PNG state and overwrite results for the fixed V/R/D panel."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.experiments import prefeval_attribute_generalization as x
from scripts.reporting.verify_prefeval_semantic_transfer import raw_score


def read_rows(root, name, arm):
    result = {}
    png_hashes = {}
    for path in sorted((root / 'evaluation').glob(f'shard-*/{name}.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['condition'] != arm:
                continue
            key = row['target'], row['panel'], row['query']['id']
            assert key not in result, ('duplicate', arm, key)
            png = root / 'training' / arm / row['target'] / 'memory.png'
            if png not in png_hashes:
                png_hashes[png] = x.s.file_sha(png)
            assert png_hashes[png] == row['png_sha']
            result[key] = row
    return result


def summarize(root, arm, evaluation, gates):
    reads = read_rows(root, 'reads', arm)
    ranks = read_rows(root, 'ranking', arm)
    def correct(sid, panel, item):
        row = reads[sid, panel, item['id']]
        assert row['query'] == item
        return raw_score(row, item, panel == 'mcq')
    def ranked(sid, item):
        row = ranks[sid, 'application_reserved', item['id']]
        scores, gold = row['score']['scores'], row['gold_index']
        assert row['query'] == item
        assert row['choices'][gold] == item['target']
        return scores[gold] > max(v for i, v in enumerate(scores) if i != gold)
    groups = defaultdict(list)
    capacity = defaultdict(lambda: [0, 0])
    states = {}
    mcq = []
    transfer = []
    for sid, target in evaluation['targets'].items():
        recovery = [correct(sid, 'qualification', q) for q in target['qualification'] if q['kind'] == 'recovery']
        assert len(recovery) == 2 * len(target['state'])
        apps = [ranked(sid, q) for q in target['application_reserved']]
        official = [correct(sid, 'mcq', q) for q in target['mcq']]
        for q, ok in zip(target['mcq'], official):
            groups[q['semantic_group']].append(ok)
        mcq.extend(official)
        transfer.extend(apps)
        cap = capacity['K' + str(len(target['state']))]
        cap[0] += int(all(recovery))
        cap[1] += 1
        states[sid] = dict(recovery=all(recovery), joint=all(recovery) and all(apps),
                           joint_with_mcq=all(recovery) and all(apps) and all(official))
    contrasts = []
    for contrast in evaluation['overwrite_contrasts']:
        before = evaluation['targets'][contrast['before_state']]
        after = evaluation['targets'][contrast['after_state']]
        pairs = []
        for a in before['application_reserved']:
            if a['scope'] != contrast['scope']:
                continue
            b = next(q for q in after['application_reserved'] if q['case_id'] == a['case_id'] and q['scope'] == a['scope'])
            assert a['query'] == b['query'] and a['target'] != b['target']
            pairs.append(ranked(contrast['before_state'], a) and ranked(contrast['after_state'], b))
        assert pairs
        contrasts.append(dict(contrast=contrast, correct=sum(pairs), total=len(pairs)))
    absolute = dict(recovery=all(capacity[k][0] >= threshold[0] for k, threshold in gates['recovery'].items()),
                    original_mcq=sum(mcq) >= gates['mcq_min'], transfer_ranking=sum(transfer) >= gates['reserved_ranking_min'],
                    overwrite_pairs=sum(c['correct'] for c in contrasts) >= gates['contrast_pairs_min'],
                    every_overwrite=all(c['correct'] >= gates['each_contrast_min'] for c in contrasts))
    return dict(mcq=[sum(mcq), len(mcq)], semantic_group_macro=sum(sum(v)/len(v) for v in groups.values())/len(groups),
                transfer=[sum(transfer), len(transfer)], capacity=dict(capacity), states=states,
                joint=sum(v['joint'] for v in states.values()), joint_with_mcq=sum(v['joint_with_mcq'] for v in states.values()),
                contrasts=contrasts, absolute_gates=absolute)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration, _, _ = x.load()
    evaluation = x.s.load_json(x.s.DATA / 'evaluation-payload.json')
    result = {arm: summarize(root, arm, evaluation, registration['progression'])
              for arm, root in [('V', args.source), ('R', args.output), ('D', args.output)]}
    gain = result['D']['semantic_group_macro'] - result['R']['semantic_group_macro']
    result['comparison'] = dict(mcq_macro_gain=gain, target_met=gain >= .10)
    result['all_outcome_gates_met'] = all(result['D']['absolute_gates'].values()) and gain >= .10
    result['writer_updates'] = 0
    result['usable_writer'] = False
    result['scope'] = 'Counterfactual context/proposal augmentation using previously authored attributes; substantive attribute expansion remains untested.'
    _, payload, cases = x.load()
    for arm in ('R', 'D'):
        reads = read_rows(args.output, 'reads', arm)
        ranks = read_rows(args.output, 'ranking', arm)
        pairs = {'xml': [], 'ranking': []}
        for sid, target in payload['targets'].items():
            for vid in sorted({c['value_id'] for c in target['applications']}):
                for first, second in ((0, 1), (2, 3)):
                    xml_ok, ranking_ok = [], []
                    for case in (cases[vid][first], cases[vid][second]):
                        query, _, gold = x.q.xml_candidates(case, 0)
                        xml_ok.append(raw_score(reads[sid, 'attribute_xml', query['id']], {**query, 'target_index': gold}, True))
                        query, choices, gold = x.s.ranking_candidates(case, 0)
                        row = ranks[sid, 'attribute_full_action', query['id']]
                        assert row['choices'] == choices and row['gold_index'] == gold
                        scores = row['score']['scores']
                        ranking_ok.append(scores[gold] > max(v for i, v in enumerate(scores) if i != gold))
                    pairs['xml'].append(all(xml_ok))
                    pairs['ranking'].append(all(ranking_ok))
        result[arm]['counterfactual_pairs'] = {k: [sum(v), len(v)] for k, v in pairs.items()}
    result['compute'] = {
        arm: {'seconds': sum(d['seconds'] for d in rows),
              'tokens': sum(d['processed_input_tokens'] for d in rows),
              'gradient_forwards': sum(d['reader_forwards'] for d in rows)}
        for arm in ('R', 'D')
        for rows in [[json.loads(p.read_text()) for p in (args.output / 'training' / arm).glob('*/complete.json')]]
    }
    eval_rows = [json.loads(p.read_text()) for p in (args.output / 'evaluation').glob('shard-*/complete.json')]
    result['compute']['evaluation'] = {k: sum(d[k] for d in eval_rows) for k in ('seconds', 'processed_input_tokens', 'recovery_ce_forwards')}
    (args.output / 'same-png-gates.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'states'} if isinstance(v, dict) else v for k,v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
