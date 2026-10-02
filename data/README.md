# 当前数据入口

固定官方 PrefEval 来源：`50795054b5ff5f418d2b768a331d71e480f93331`。

- 官方导出与 SHA 清单：`reports/prefeval-official-alignment-20260923/data/`。
- K1 划分、问法及登记：`reports/prefeval-k1-l0-l2-20260924/`。
- 多目标 bank 与登记：`reports/prefeval-multitarget-20260927/`。
- 当前读取实现：`scripts/experiments/prefeval_k1_data.py`。

官方 seed42 的820/180主题划分与项目730/90内部划分分别记录。完整对话、acknowledgment、最终答案mask和评分信息隔离遵循官方对齐入口；训练问法、替代Judge和去重视图单独声明。

旧seed2026自定义 PrefEval 数据、复述与人工排序实验已退出当前数据入口。历史生成器、独立文本baseline和通用schema保留兼容用途，其旧分数不自动成为当前baseline。原始官方数据不会因曾被旧实验使用而失效。

保留依据见 `docs/ALIGNMENT_BOUNDARY.md`。本次仓库清理没有删除外部共享盘、原工作树中的未跟踪文件或模型权重。
