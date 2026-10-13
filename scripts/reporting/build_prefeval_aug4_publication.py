"""Rebuild every published checkpoint table from byte-preserved summaries.

CPU-only. Old B730 is a protocol-qualified historical comparison, not a paired
augmentation ablation. No missing checkpoint or confidence interval is imputed.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/prefeval-aug4x3-init-ablation-exposure512-20261009'
OLD = ROOT / 'reports/prefeval-k1-l0-l2-20260924'
CONTROLS = ('memory', 'mismatch', 'blank', 'text')
ARMS = ('P-L', 'P-H', 'R-L', 'R-H')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    sources, old, new = {}, {}, {}

    def read(path):
        rel = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        # Historical JSONs are Git text blobs: Windows checkout uses CRLF,
        # whereas their committed bytes use LF. Bind the portable Git content.
        historical = path.is_relative_to(OLD)
        if historical:
            raw = raw.replace(b'\r\n', b'\n')
        sources[rel] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                        'hash_policy': 'Git LF text bytes' if historical else 'exact original bytes'}
        return load(path), rel

    data, rel = read(OLD / 'evidence/b730-exposure512-step93440-train-final-20260929/results.json')
    for name, summary in data['evaluations'].items():
        match = re.search(r'step-(\d+)-(\w+)-(V\d)', name)
        step, split, version = int(match[1]), match[2], match[3]
        for family, metrics in summary['families'].items():
            old[step, split, version, family] = (metrics['controls'], rel)
    data, rel = read(OLD / 'evidence/b730-exposure512-baseline23360-20260927/baseline-summary.json')
    for version, summary in data.items():
        for family, metrics in summary['families'].items():
            old[23360, 'train', version, family] = (metrics['controls'], rel)
    for split, version, path in [
        ('pilot', 'V0', 'b730-v0-v2-cbank-20260926-1300/pilot-V0-summary.json'),
        ('pilot', 'V1', 'b730-pilot-final-20260926-0640/summary.json'),
        ('dev', 'V0', 'b730-dev-v0-final-20260926-1155/summary.json'),
        ('dev', 'V1', 'b730-dev-final-20260926-0530/summary.json'),
        ('pilot', 'V2', 'b730-pilot-v2-final-20260926-1410/summary.json'),
        ('dev', 'V2', 'b730-v0-v2-cbank-20260926-1300/dev-V2-summary.json'),
    ]:
        data, rel = read(OLD / 'evidence' / path)
        for metric in data['metrics']:
            assert metric['prefix'] == 0
            key = 23360, split, version, metric['family']
            if key not in old:
                old[key] = ({}, rel)
            old[key][0][metric['control']] = metric
    for path in sorted((REPORT / 'evidence/summaries').glob('*/summary.json')):
        summary, rel = read(path)
        match = re.fullmatch(r'(P-L|P-H|R-L|R-H)-(\d+)-(\w+)', path.parent.name)
        arm, step, split = match[1], int(match[2]), match[3]
        split = 'official180' if split == 'official' else split
        new[step, split, arm, 'aggregate'] = (summary, rel)
        if split == 'official180':
            new[step, 'official177', arm, 'derived_subset'] = (summary['official177'], rel)
        if split == 'diagnostics':
            for suite, metrics in summary['suites'].items():
                new[step, split, arm, suite] = (metrics, rel)
    assert len(old) == 98 and len(new) == 148
    for controls, _ in old.values():
        assert set(controls) == set(CONTROLS)
    for summary, _ in new.values():
        assert set(summary['controls']) == set(CONTROLS)
    return old, new, sources


def pct(c):
    return 100 * c['correct'] / c['total']


def triple(c):
    return f"{pct(c['memory']):.2f} / {pct(c['mismatch']):.2f} / {pct(c['memory']) - pct(c['mismatch']):+.2f}"


def count(c):
    return f"{c['correct']}/{c['total']} ({pct(c):.2f}%)"


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='Require byte-exact agreement with committed derived tables')
    args = ap.parse_args()
    old, new, sources = collect()
    current_rows, old_rows = [], []
    text = ['# 本次全 checkpoint 匹配、错配与控制数据', '',
            '由 `scripts/reporting/build_prefeval_aug4_publication.py` 从 84 份原始汇总生成。', '',
            '四组 × 五个 checkpoint × 四个常规 split，加四组最终 official180，共 84 个原始统计单元；official177 为其中四个官方单元的去重子集，共 88 个汇总单元。另展开 60 个诊断 suite 单元，不得与 aggregate 重复相加。', '',
            'memory/mismatch 使用 2 个图像种子 × 4 个正确选项位置；blank/text 不复制图像种子。表内准确率分母为回读次数，不是独立偏好数。train/dev/diagnostics/opposites/official180/official177 分别为 730/90/96/16/180/177 个偏好。', '',
            'Δ = memory − mismatch，单位 pp；CI 为偏好级配对 bootstrap 95% 区间（10,000 次，种子 20261009），不是训练种子间区间，未作多重比较校正。诊断项是已训练偏好的新表达；opposites 的 mismatch 是对应原偏好图像在反事实标签下的成绩。', '',
            '每行链接到原始汇总；机器可读明细保留未四舍五入的比率、分母、解析失败和来源 SHA。', '',
            '[CSV 全明细](ALL-CHECKPOINTS-20261013.csv) · [旧 B730 对比](B730-COMPARISON-20261013.md) · [分析](ANALYSIS-20261013.md) · [证据及复算](EVIDENCE-20261013.md)', '']
    for split in ('train', 'dev', 'diagnostics', 'opposites', 'official180', 'official177'):
        text += ['## ' + split, '']
        rows = []
        for key in sorted(new):
            step, sp, arm, suite = key
            if sp != split or suite not in ('aggregate', 'derived_subset'):
                continue
            s, rel = new[key]
            c, gain = s['controls'], s['paired_memory_gain']['mismatch']
            ci = gain['ci95_percentage_points']
            link = Path(rel).relative_to(REPORT.relative_to(ROOT)).as_posix()
            rows.append([f'[{step} / {step * 4 // 730}]({link})', arm,
                         count(c['memory']), count(c['mismatch']),
                         f"{gain['difference_percentage_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                         count(c['blank']), count(c['text'])])
        text += [table(['Step / 曝光', '组', '匹配', '错配', 'Δ [95% CI]', 'Blank', 'Text'], rows), '']
    text += ['## Diagnostics 三种 suite 全展开', '',
             'Wstar/T1：新写入表达、原问题；W0/Tstar：原写入表达、新问题；Wstar/Tstar：两者均为新表达。仍为同一批 96 个已训练偏好。', '']
    rows = []
    for (step, split, arm, suite), (s, rel) in sorted(new.items()):
        c, gain = s['controls'], s['paired_memory_gain']['mismatch']
        for ctrl in CONTROLS:
            x = c[ctrl]
            assert 0 <= x['correct'] <= x['total']
            current_rows.append(dict(step=step, exposure=step * 4 // 730, split=split, arm=arm, suite=suite,
                control=ctrl, correct=x['correct'], total=x['total'], accuracy=x['correct'] / x['total'],
                parse_failures=x['parse_failures'], preferences=x['preference_count'],
                memory_minus_mismatch_pp=gain['difference_percentage_points'],
                gain_ci95_low_pp=gain['ci95_percentage_points'][0], gain_ci95_high_pp=gain['ci95_percentage_points'][1],
                source=rel, source_sha256=sources[rel]['sha256']))
        if suite not in ('aggregate', 'derived_subset'):
            ci = gain['ci95_percentage_points']
            rows.append([step, arm, suite, count(c['memory']), count(c['mismatch']),
                         f"{gain['difference_percentage_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", count(c['blank']), count(c['text'])])
    text += [table(['Step', '组', 'Suite', '匹配', '错配', 'Δ [95% CI]', 'Blank', 'Text'], rows)]
    # Cross-check the original completion CSV, preserving its historical bytes.
    legacy = list(csv.DictReader((REPORT / 'FINAL-METRICS-20261013.csv').open(encoding='utf-8-sig')))
    assert len(legacy) == 352
    index = {(r['step'], r['split'], r['arm'], r['control']): r for r in current_rows if r['suite'] in ('aggregate', 'derived_subset')}
    for row in legacy:
        key = int(row['step']), ('official180' if row['split'] == 'official' else row['split']), row['arm'], row['control']
        other = index[key]
        for name in ('correct', 'total'):
            assert int(row[name]) == other[name]
        for name in ('memory_minus_mismatch_pp', 'gain_ci95_low_pp', 'gain_ci95_high_pp'):
            assert abs(float(row[name]) - other[name]) < 1e-10
        assert row['summary_sha256'] == other['source_sha256']
    comp = ['# 旧 B730 exposure512 与本次 aug4x3 全量比较', '',
            '**数值并列是历史比较，不是严格单因素因果对照。** 所有已提供的 checkpoint 均保留；缺失项用 —，不插值、不挑最佳 checkpoint。', '',
            '| 维度 | 旧 B730 exposure512 | 本次 aug4x3 |', '|---|---|---|',
            '| 内容 | train730、pilot64（已训练偏好）、dev90 | train730、dev90、已训练内容诊断96、opposites16、official180/177 |',
            '| 初始化 | 4f 任务权重；包含额外历史训练 | P 为官方 DreamLite Base；R 仅 U-Net 标准随机，TE/VAE/Reader 仍预训练冻结 |',
            '| 学习率 | 5e-5 | L=5e-5，H=5e-4 |',
            '| 训练预算 | batch4、93440 更新、每内容512曝光；V0/V1各256 | batch4、93440 更新、每内容512曝光；W0–W3各128 |',
            '| 教师与问题 | 既有 B730 教师、旧问题配置 | 新730目标库，T1–T3×4位置×24更新=288步/目标；四组共用 |',
            '| 主表评分 | 2种子；原主表正确选项位置口径；V0/V1是表达条件，不是独立训练种子 | 2种子×4正确选项位置，偏好级配对区间 |',
            '| 统计限制 | 本表不补造旧记录没有的配对区间 | 仅一组配对训练种子；区间不描述训练种子方差 |', '',
            '增强扩大表达覆盖，不把730个内容变成8760个独立内容。新旧初始化、教师、表达、位置均发生变化，不能把跨代差异归因于其中一个因素。', '',
            '下列并列表每格为 **匹配% / 错配% / Δpp**；两代 train/dev 主问题 T1 并列，但分母与条件不同。', '']
    for split in ('train', 'dev'):
        rows = []
        for step in (5840, 11680, 23360, 46720, 70080, 93440):
            row = [f'{step} / {step * 4 // 730}']
            row += [triple(old[step, split, v, 'T1'][0]) if (step, split, v, 'T1') in old else '—' for v in ('V0', 'V1')]
            row += [triple(new[step, split, a, 'aggregate'][0]['controls']) if (step, split, a, 'aggregate') in new else '—' for a in ARMS]
            rows.append(row)
        comp += ['## ' + split + ' 主问题', '', table(['Step / 曝光', '旧 V0', '旧 V1', *ARMS], rows), '']
    comp += ['## 旧 B730 全 checkpoint、全问题族及控制分母', '',
             '包含 98 个分组：88 个 V0/V1 主记录及 step23360 的10个 V2 补充记录。旧 pilot 与新 diagnostics/opposites 不同，不能改名后合并。下面直接保留旧 T1/T2/T3/O1/O2 标签；具体题文以链接的旧协议与源汇总为准。', '',
             '[旧协议与部署](../prefeval-k1-l0-l2-20260924/B_MCQ_DEPLOYMENT_20260925.md) · [旧数据完整 CSV](B730-ALL-CHECKPOINTS-20261013.csv) · [本次全量表](ALL-CHECKPOINTS-20261013.md)', '']
    rows = []
    for (step, split, version, family), (c, rel) in sorted(old.items()):
        link = '../prefeval-k1-l0-l2-20260924/' + Path(rel).relative_to(OLD.relative_to(ROOT)).as_posix()
        rows.append([f'[{step}]({link})', split, version, family, *[count(c[ctrl]) for ctrl in CONTROLS],
                     f"{pct(c['memory']) - pct(c['mismatch']):+.2f}"])
        for ctrl in CONTROLS:
            x = c[ctrl]
            assert 0 <= x['correct'] <= x['total']
            old_rows.append(dict(step=step, exposure=step * 4 // 730, split=split, version=version, family=family,
                control=ctrl, correct=x['correct'], total=x['total'], accuracy=x['correct'] / x['total'],
                parse_failures=x.get('parse_failures', ''), memory_minus_mismatch_pp=pct(c['memory']) - pct(c['mismatch']),
                source=rel, source_sha256=sources[rel]['sha256']))
    comp += [table(['Step', 'Split', '表达', '问题族', '匹配', '错配', 'Blank', 'Text', 'Δ pp'], rows)]
    outputs = {
        'ALL-CHECKPOINTS-20261013.md': ('\n'.join(text).rstrip() + '\n').encode(),
        'ALL-CHECKPOINTS-20261013.csv': csv_bytes(current_rows),
        'B730-COMPARISON-20261013.md': ('\n'.join(comp).rstrip() + '\n').encode(),
        'B730-ALL-CHECKPOINTS-20261013.csv': csv_bytes(old_rows),
        'TABLE-SOURCES.json': (json.dumps({'sources': sources, 'new_aggregate_units': 88, 'diagnostic_suite_units': 60,
                                           'old_family_units': 98}, ensure_ascii=False, indent=2) + '\n').encode(),
    }
    for name, data in outputs.items():
        if args.check:
            assert (REPORT / name).read_bytes() == data, 'Derived table differs: ' + name
        else:
            (REPORT / name).write_bytes(data)
    print(json.dumps({'status': 'verified' if args.check else 'built', 'current_csv_rows': len(current_rows),
                      'old_csv_rows': len(old_rows), 'original_final_csv_rows_verified': len(legacy), 'source_files': len(sources)}))


if __name__ == '__main__':
    main()
