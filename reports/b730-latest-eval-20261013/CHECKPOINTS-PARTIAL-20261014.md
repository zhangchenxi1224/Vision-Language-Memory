# B730 最新统一协议：已核验 checkpoint 明细

固定快照：2026-10-14 02:15 CST。12/17 主汇总已完成本地逐行复算；未收录项标为未归档，不用其他实验或历史协议结果填补。

| Step | Train | Dev | Diagnostics | Opposites | Official180/177 |
|---|---|---|---|---|---|
| 23360 | 本快照未归档 | 本快照未归档 | 本快照未归档 | 本快照未归档 | 不在协议内 |
| 46720 | 已核验 | 已核验 | 本快照未归档 | 已核验 | 不在协议内 |
| 70080 | 已核验 | 已核验 | 已核验 | 已核验 | 不在协议内 |
| 93440 | 已核验 | 已核验 | 已核验 | 已核验 | 已核验 |

下表每个分母为回读次数；同一偏好下的种子和答案位置不是独立内容。诊断 aggregate 与三个 suite 不得重复相加；official177 由 official180 过滤，不是额外执行。区间为偏好级配对 bootstrap（10,000次，种子20261009），不代表训练种子不确定性。

| Step / Split / Suite | 匹配 | 错配 | 匹配−错配 pp [95% CI] | 灰图 | 全文 |
|---|---:|---:|---:|---:|---:|
| [46720 / dev / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-046720-dev.json) | 295/720 (40.97%) | 301/720 (41.81%) | -0.83 [-2.08, +0.28] | 154/360 (42.78%) | 342/360 (95.00%) |
| [46720 / opposites / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-046720-opposites.json) | 44/128 (34.38%) | 16/128 (12.50%) | +21.88 [+6.25, +39.08] | 24/64 (37.50%) | 64/64 (100.00%) |
| [46720 / train / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-046720-train.json) | 4770/5840 (81.68%) | 2451/5840 (41.97%) | +39.71 [+36.93, +42.57] | 1180/2920 (40.41%) | 2776/2920 (95.07%) |
| [70080 / dev / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-070080-dev.json) | 297/720 (41.25%) | 297/720 (41.25%) | +0.00 [-1.39, +1.39] | 154/360 (42.78%) | 342/360 (95.00%) |
| [70080 / diagnostics / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-070080-diagnostics.json) | 1394/2304 (60.50%) | 1088/2304 (47.22%) | +13.28 [+10.46, +16.06] | 563/1152 (48.87%) | 1076/1152 (93.40%) |
| [70080 / diagnostics / W0/Tstar](evidence/partial-summaries-20261014-021530/summaries/B730-070080-diagnostics.json) | 664/768 (86.46%) | 364/768 (47.40%) | +39.06 [+30.86, +47.14] | 191/384 (49.74%) | 367/384 (95.57%) |
| [70080 / diagnostics / Wstar/T1](evidence/partial-summaries-20261014-021530/summaries/B730-070080-diagnostics.json) | 366/768 (47.66%) | 363/768 (47.27%) | +0.39 [-0.52, +1.56] | 181/384 (47.14%) | 358/384 (93.23%) |
| [70080 / diagnostics / Wstar/Tstar](evidence/partial-summaries-20261014-021530/summaries/B730-070080-diagnostics.json) | 364/768 (47.40%) | 361/768 (47.01%) | +0.39 [-1.43, +2.34] | 191/384 (49.74%) | 351/384 (91.41%) |
| [70080 / opposites / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-070080-opposites.json) | 44/128 (34.38%) | 11/128 (8.59%) | +25.78 [+8.59, +45.31] | 24/64 (37.50%) | 64/64 (100.00%) |
| [70080 / train / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-070080-train.json) | 5266/5840 (90.17%) | 2465/5840 (42.21%) | +47.96 [+45.02, +50.96] | 1180/2920 (40.41%) | 2776/2920 (95.07%) |
| [93440 / dev / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-093440-dev.json) | 295/720 (40.97%) | 300/720 (41.67%) | -0.69 [-1.81, +0.42] | 154/360 (42.78%) | 342/360 (95.00%) |
| [93440 / diagnostics / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-093440-diagnostics.json) | 1433/2304 (62.20%) | 1113/2304 (48.31%) | +13.89 [+11.02, +16.67] | 563/1152 (48.87%) | 1076/1152 (93.40%) |
| [93440 / diagnostics / W0/Tstar](evidence/partial-summaries-20261014-021530/summaries/B730-093440-diagnostics.json) | 700/768 (91.15%) | 376/768 (48.96%) | +42.19 [+33.85, +50.26] | 191/384 (49.74%) | 367/384 (95.57%) |
| [93440 / diagnostics / Wstar/T1](evidence/partial-summaries-20261014-021530/summaries/B730-093440-diagnostics.json) | 364/768 (47.40%) | 371/768 (48.31%) | -0.91 [-2.86, +0.91] | 181/384 (47.14%) | 358/384 (93.23%) |
| [93440 / diagnostics / Wstar/Tstar](evidence/partial-summaries-20261014-021530/summaries/B730-093440-diagnostics.json) | 369/768 (48.05%) | 366/768 (47.66%) | +0.39 [-1.04, +1.95] | 191/384 (49.74%) | 351/384 (91.41%) |
| [93440 / official / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-093440-official.json) | 521/1440 (36.18%) | 504/1440 (35.00%) | +1.18 [-0.14, +2.99] | 250/720 (34.72%) | 666/720 (92.50%) |
| [93440 / official / official177](evidence/partial-summaries-20261014-021530/summaries/B730-093440-official.json) | 497/1416 (35.10%) | 498/1416 (35.17%) | -0.07 [-0.49, +0.42] | 247/708 (34.89%) | 654/708 (92.37%) |
| [93440 / opposites / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-093440-opposites.json) | 44/128 (34.38%) | 11/128 (8.59%) | +25.78 [+8.59, +45.31] | 24/64 (37.50%) | 64/64 (100.00%) |
| [93440 / train / aggregate](evidence/partial-summaries-20261014-021530/summaries/B730-093440-train.json) | 5463/5840 (93.54%) | 2483/5840 (42.52%) | +51.03 [+47.95, +54.08] | 1180/2920 (40.41%) | 2776/2920 (95.07%) |

[终点与 AUG4 对比](evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.md) · [解释边界](SUPPLEMENT_INTERPRETATION.md) · [历史 B730 原协议全表](../prefeval-aug4x3-init-ablation-exposure512-20261009/B730-COMPARISON-20261013.md)
