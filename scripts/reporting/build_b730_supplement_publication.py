"""Rebuild the fixed partial B730 checkpoint table without rerunning evaluation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/b730-latest-eval-20261013'
SNAPSHOT = REPORT / 'evidence/partial-summaries-20261014-021530'


def generate():
    audit = json.loads((SNAPSHOT / 'strict-audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'verified_complete_sets_only' and len(audit['results']) == 12
    expected = {Path(r['summary_path']).parent.name: r['summary_sha256'] for r in audit['results']}
    assert {p.stem for p in (SNAPSHOT / 'summaries').glob('*.json')} == set(expected)
    rows = []
    sources = {}
    for path in sorted((SNAPSHOT / 'summaries').glob('*.json')):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected[path.stem]
        s = json.loads(raw)
        assert s['status'] == 'verified_complete'
        step, split = s['checkpoint']['optimizer_step'], s['split']
        rel = path.relative_to(REPORT).as_posix()
        sources[rel] = {'sha256': digest, 'bytes': len(raw)}
        units = [('aggregate', s)]
        if split == 'diagnostics':
            units.extend(s['suites'].items())
        if split == 'official':
            units.append(('official177', s['official177']))
        for suite, metrics in units:
            gain = metrics['paired_memory_gain']['mismatch']
            for control in ('memory', 'mismatch', 'blank', 'text'):
                c = metrics['controls'][control]
                rows.append(dict(step=step, split=split, suite=suite, control=control,
                                 correct=c['correct'], total=c['total'], accuracy=c['rate'],
                                 preferences=c['preference_count'], parse_failures=c['parse_failures'],
                                 memory_minus_mismatch_pp=gain['difference_percentage_points'],
                                 ci95_low_pp=gain['ci95_percentage_points'][0],
                                 ci95_high_pp=gain['ci95_percentage_points'][1],
                                 source=rel, source_sha256=digest))
    assert len(rows) == 76
    text = ['# B730 最新统一协议：已核验 checkpoint 明细', '',
            '固定快照：2026-10-14 02:15 CST。12/17 主汇总已完成本地逐行复算；未收录项标为未归档，不用其他实验或历史协议结果填补。', '',
            '| Step | Train | Dev | Diagnostics | Opposites | Official180/177 |',
            '|---|---|---|---|---|---|']
    for step in (23360, 46720, 70080, 93440):
        status = []
        for split in ('train', 'dev', 'diagnostics', 'opposites', 'official'):
            name = f'B730-{step:06d}-{split}'
            status.append('已核验' if name in expected else ('不在协议内' if split == 'official' and step != 93440 else '本快照未归档'))
        text.append('| ' + str(step) + ' | ' + ' | '.join(status) + ' |')
    text += ['', '下表每个分母为回读次数；同一偏好下的种子和答案位置不是独立内容。诊断 aggregate 与三个 suite 不得重复相加；official177 由 official180 过滤，不是额外执行。区间为偏好级配对 bootstrap（10,000次，种子20261009），不代表训练种子不确定性。', '',
             '| Step / Split / Suite | 匹配 | 错配 | 匹配−错配 pp [95% CI] | 灰图 | 全文 |',
             '|---|---:|---:|---:|---:|---:|']
    for i in range(0, len(rows), 4):
        group = rows[i:i + 4]
        r = group[0]
        cells = [f"{x['correct']}/{x['total']} ({100*x['accuracy']:.2f}%)" for x in group]
        label = f"[{r['step']} / {r['split']} / {r['suite']}]({r['source']})"
        delta = f"{r['memory_minus_mismatch_pp']:+.2f} [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}]"
        text.append('| ' + ' | '.join([label, cells[0], cells[1], delta, cells[2], cells[3]]) + ' |')
    text += ['', '[终点与 AUG4 对比](evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.md) · [解释边界](SUPPLEMENT_INTERPRETATION.md) · [历史 B730 原协议全表](../prefeval-aug4x3-init-ablation-exposure512-20261009/B730-COMPARISON-20261013.md)', '']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return {'CHECKPOINTS-PARTIAL-20261014.md': '\n'.join(text).encode(),
            'CHECKPOINTS-PARTIAL-20261014.csv': stream.getvalue().encode(),
            'CHECKPOINT-SOURCES-20261014.json': (json.dumps(sources, indent=2) + '\n').encode()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    artifacts = generate()
    for name, data in artifacts.items():
        path = REPORT / name
        if args.check:
            if path.read_bytes() != data:
                raise ValueError('Derived result differs: ' + name)
        else:
            path.write_bytes(data)
    print(json.dumps({'status': 'verified' if args.check else 'built', 'main_summaries': 12,
                      'control_rows': 76, 'full_experiment_complete': False}))
