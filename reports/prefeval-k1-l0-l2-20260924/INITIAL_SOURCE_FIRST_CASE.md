# 初写源图保持对照：固定 B 首例仍在第一次更新失败

2026-09-24。固定 `entertain_games:0001`、错配 `entertain_games:0007`，官方正确选项 D。
本轮固定首例读取在训练前已部署；报告 16 条完整记录，完整 64 条继续评测。

| B 模型、匹配图 | 初写 0 | 干扰 1 | 干扰 5 | 干扰 10 |
|---|---|---|---|---|
| 原递归源图保持，chain 0 | D，对 | C，错 | C，错 | C，错 |
| 原递归源图保持，chain 1 | D，对 | C，错 | C，错 | C，错 |
| 仅初写源图保持，chain 0 | D，对 | C，错 | C，错 | C，错 |
| 仅初写源图保持，chain 1 | C，错 | C，错 | C，错 | C，错 |

新对照在这四个端点的错配图均回答 C。16 条均无解析失败或截断。
已核对匹配/错配两噪声链的 44 张实际 PNG、40 次 RGB 递归转换；绑定本轮保持后权重。
新对照的 chain 1 初写已经失败，不能把该链后续错误全归为遗忘。

这排除了“改成有信息的原初写训练输入就足以修复此例”的简单解释。
但训练时的输入来自旧 write2048，而实际新链的初写来自保持后的 Writer；
二者的图片分布仍可能不同。该单例还不能区分一步保持未拟合与新输入适应失败，
更不能代替整体 L2 指标。

下一项[训练源图单步定位](RETAIN_TRAINING_SOURCE_PROBE.md)固定这一差异，不再新增训练。
该定位现已完成：[训练时见过的可读 PNG 在两种噪声下一步后也失败](RETAIN_TRAINING_SOURCE_RESULTS.md)，
因此新旧初写图的分布变化不是足够解释。

- [完整读回与递归核验](evidence/initial-source-first-B/initial-source-first-B-verification.json)
- [16 条原始记录](evidence/initial-source-first-B/B/first-case-T1/readback-0.jsonl)
- [原保持首例](FIRST_POSTRETAIN_DIAGNOSTIC.md)
