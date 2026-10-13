# PrefEval aug4x3 四组初始化实验：已完成

2026-10-13 已完成四组各93,440步、1,472项规定任务、所有 checkpoint 评测、official180/177敏感性汇总与归档。当前没有待恢复训练；监控已结束，空闲实验资源已收尾。

**结论：低学习率组在已训练偏好上学到明确的匹配记忆优势，官方 Base 初始化学习更快；本轮 dev 和 official 尚未显示可靠的新内容记忆收益。**

| 阅读目的 | 入口 |
|---|---|
| 最终指标、置信区间、限制 | [最终报告](FINAL-REPORT-20261013.md) |
| 本次每组每个 checkpoint 的匹配/错配、blank/text、诊断展开 | [完整表格](ALL-CHECKPOINTS-20261013.md) · [完整计数 CSV](ALL-CHECKPOINTS-20261013.csv) |
| 与旧全量 B730 比较、旧全部 checkpoint/问题族/控制 | [历史比较表](B730-COMPARISON-20261013.md) · [旧计数 CSV](B730-ALL-CHECKPOINTS-20261013.csv) |
| 旧B730在最新统一协议下的新增补测 | [2026-10-14阶段结果](../b730-latest-eval-20261013/README.md) · [固定终点同协议对比](../b730-latest-eval-20261013/evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.md)；历史V0/V1结果仍保留 |
| 总分结构的结论与机制分析 | [分析](ANALYSIS-20261013.md) |
| 完成情况、迁移及资源收尾 | [实验进度](PROGRESS-20261013.md) |
| 原始回读、冻结代码、数据、模型/图像指纹、验证命令 | [证据与复算](EVIDENCE-20261013.md) |
| 冻结科学条件 | [方案](PLAN.md) · [协议](PROTOCOL.json) |

88个汇总单元含4个由official180派生的official177，不是88次独立评测；60个诊断suite展开也不可与其父级重复计数。旧B730与本轮存在协议差异，比较不构成增强或初始化的跨代单因素因果实验。
