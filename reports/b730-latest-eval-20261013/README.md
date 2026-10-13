# B730 最新统一协议补测：阶段结果

**固定93,440步终点已完成并核验。B730能够应对已训练内容的新提问措辞，对新写入表达的泛化较弱；尚未显示可靠的新内容记忆收益。** 本次发布是2026-10-14 02:15 CST固定快照，严格归档12/17份主汇总；不是全实验完成声明。

| 阅读目的 | 入口 |
|---|---|
| B730与AUG4四组同协议终点对比、paired96 | [结果表](evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.md) · [完整计数CSV](evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.csv) · [配对区间JSON](evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.json) |
| 本快照全部checkpoint的匹配、错配、灰图、全文 | [明细表](CHECKPOINTS-PARTIAL-20261014.md) · [76行控制CSV](CHECKPOINTS-PARTIAL-20261014.csv) |
| 结论、解释与下一步 | [分析](ANALYSIS-20261014.md) |
| 完成范围及剩余归档 | [进度](PROGRESS-20261014.md) |
| 固定科学条件及“未见”含义 | [协议](PROTOCOL.json) · [覆盖和比较边界](SUPPLEMENT_INTERPRETATION.md) |
| 原始来源与独立离线复算 | [证据说明](EVIDENCE-20261014.md) · [发布清单](PUBLICATION-MANIFEST.json) |
| B730历史原协议成绩 | [2026-09-29最终报告](../prefeval-k1-l0-l2-20260924/B730_EXPOSURE512_FINAL_20260929.md) · [历史全checkpoint比较](../prefeval-aug4x3-init-ablation-exposure512-20261009/B730-COMPARISON-20261013.md) |

## 固定终点：同一96条已训练内容

| 条件 | B730 | AUG4 P-L | AUG4 R-L |
|---|---:|---:|---:|
| 原写入W0、原提问T1 | 93.62% | 93.10% | 86.59% |
| 新写入W*、原提问T1 | 47.40% | 72.27% | 65.10% |
| 原写入W0、新提问T* | 91.15% | 89.71% | 84.51% |
| 新写入W*、新提问T* | 48.05% | 71.22% | 62.76% |

表中第一行从train730抽取相同96个ID；不能用整个train730均值代替。B730仅换写入表达下降46.22pp，偏好级配对95%区间为[−54.69, −37.50]；P-L对应下降20.83pp。高学习率组和所有控制均保留在完整对比表中，没有按分数筛除。

dev90的B730匹配/错配为40.97%/41.67%，official177为35.10%/35.17%。这两项匹配收益的95%区间均跨零。未见提问T*是已训练内容的改写问题，不是未训练内容或新的推理任务。

## 发布与执行边界

只评测历史B730的23,360/46,720/70,080/93,440步固定权重，没有重新训练。B730保留4f任务初始化身份；AUG4 P从官方Base开始、R为随机U-Net，教师和训练覆盖也不同。因此同协议模型差异是体系比较，不是增强单因素因果估计。

本次统一确认为 `Understood.`，历史V0/V1为不同确认语；历史噪声、答案位置和donor协议也有差别，历史成绩独立保留。blank/text复用经完整审核的AUG4原始控制，保留来源绑定，不计为新B730 GPU推理。train与diagnostics的donor有88/96条不同，跨表的匹配−错配变化不能解释为纯表达影响。

复算入口是CPU只读工具；运行控制器由既有独占owner继续维护。本仓库发布不授权重放历史启动脚本或重新执行已完成分片。注册项：`prefeval-b730-latest-eval-supplement`。
