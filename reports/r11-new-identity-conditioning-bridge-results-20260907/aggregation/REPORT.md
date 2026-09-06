# R11_new identity-conditioning bridge 独立复算报告

## 结论

- 工程技术门：`true`
- Teacher replay：`true`
- 距离门：`false`
- Endpoint Reader transfer：`false`
- Bridge diagnostic：`false`
- Picture Memory 正式科学成功：`false`
- Phase 2：仍阻塞。
- 唯一改动是实际 conditioning 文本；source-only 初始化仍不接收 teacher。
- conditioning/初始化答案无关不等于 solver/writer 答案无关；优化仍由 canonical teacher 的 dense MSE 监督。

## Raw step 256

- MSE：`0.111322358251`
- MSE/M0：`0.503225995494`（门槛 <= 0.01）
- L2/M0：`0.709384236852`（门槛 <= 0.10）
- teacher-normalized RMSE：`0.509649225485`（门槛 <= 0.10）
- Reader accuracy：`0`（门槛 = 1.0）
- 决策：`distance_fail_reader_fail_preregister_one_discriminating_diagnostic`
- 次级条件诊断：`distance_fail_reader_fail_secondary_condition_improves`

## 证据边界

本报告从原始 tensor、receipt、checkpoint 和 logits 独立复算，不采信 trainer 的 PASS 字段。
唯一改动是把实际 conditioning 从原事件文本改为固定 `no changes`；x_T 初始化仍等于固定 source。
M0 是新 conditioning 经过完整四步链路后的未优化 endpoint，不能沿用父臂 M0。
次级诊断只有技术/teacher 通过且两个主门均失败时才评估，且须绝对 endpoint MSE 与 CE 同时严格优于父实验。
次级比较器始终是 source-only 初始化、原事件 conditioning 的直接父臂；teacher-matched 结果只作描述参考。
本轮至多提供该目标、dense-supervised 求解器下的 conditioning 敏感性证据，不能证明 identity mapping、事件到状态学习或普遍可达/不可达。
无论结果如何，都不证明 shared writer、ID/OOD、递归状态更新或 Picture Memory 正式成功。
