# 当前证据入口

性能分析先读取 `../experiments/registry.json`，只纳入`current_performance`目录，再按run核对协议。禁止递归扫描全部`reports/`混合汇总。

| 目录 | 角色 |
| --- | --- |
| `prefeval-k1-l0-l2-20260924/` | 当前K1、teacher/student、FM、曝光量与保持；各run分别报告 |
| `prefeval-multitarget-20260927/` | 当前多目标；各轮、配对baseline、链长分别报告 |
| `prefeval-official-alignment-20260923/` | 协议依据、固定数据导出与路线决策 |
| `official-alignment-results-20260913/`及`official-*` | 转折期工程审计、配对控制、初始化/bank依赖；非当前PrefEval baseline |

旧链路和旧自定义PrefEval数据已撤下。恢复索引与适用性见`../docs/ALIGNMENT_BOUNDARY.md`和`../docs/alignment-cleanup-manifest.json`。
