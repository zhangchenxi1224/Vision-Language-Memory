# 初写源图保持对照：固定首例未修复

2026-09-24。固定 `entertain_games:0001`、错配 `entertain_games:0007`，官方正确选项 D。
本轮固定首例读取在训练前已部署；报告 16 条完整记录，完整 64 条继续评测。
后续 A 的 16 条也已完成，以下分组报告；两组仍是同一个固定偏好，不扩大偏好分母。

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

## A 的完整首例补充

| A 匹配图 | 初写 0 | 干扰 1 | 干扰 5 | 干扰 10 |
|---|---|---|---|---|
| 仅初写源图保持，chain 0 | A，错 | C，错 | C，错 | C，错 |
| 仅初写源图保持，chain 1 | C，错 | C，错 | C，错 | C，错 |

A 错配初写 chain 0/1 分别为 C/B，后续端点均为 C，全部错误。
16 条无解析失败或截断，4 条完整链的 44 张 PNG、40 次 RGB 转换哈希核对通过。
A 的匹配回答序列与原保持对照在这些端点一致，但不声称图片本身相同。
由于两链初写均错误，该首例不能独立衡量 A 是否保存了原来可读的正确偏好。
[A 完整核验](evidence/initial-source-first-A/initial-source-first-A-verification.json)、
[A 原始读取](evidence/initial-source-first-A/A/first-case-T1/readback-0.jsonl)。

- [完整读回与递归核验](evidence/initial-source-first-B/initial-source-first-B-verification.json)
- [16 条原始记录](evidence/initial-source-first-B/B/first-case-T1/readback-0.jsonl)
- [原保持首例](FIRST_POSTRETAIN_DIAGNOSTIC.md)
