# R11_new teacher-matched initialization bridge 独立复算报告

## 结论

- 工程技术门：`true`
- Teacher replay：`true`
- 距离门：`false`
- Endpoint Reader transfer：`false`
- Bridge diagnostic：`false`
- Picture Memory 正式科学成功：`false`
- Phase 2：仍阻塞。
- 本轮使用 teacher 辅助初始化，不是答案无关的 writer，也不能补算 Phase 1A 通过。

## Raw step 256

- MSE：`0.0657004490495`
- MSE/M0：`0.354123233704`（门槛 <= 0.01）
- L2/M0：`0.595082543605`（门槛 <= 0.10）
- teacher-normalized RMSE：`0.391529363683`（门槛 <= 0.10）
- Reader accuracy：`0`（门槛 = 1.0）
- 决策：`distance_fail_reader_fail_change_one_solver_factor`
- 次级初始化诊断：`distance_fail_reader_fail_secondary_init_improves`

## 证据边界

本报告从原始 tensor、receipt、checkpoint 和 logits 独立复算，不采信 trainer 的 PASS 字段。
唯一改动是 x_T 初始化；M0 是新初始化经过完整四步链路后的未优化 endpoint。
次级诊断只有技术/teacher 通过且两个主门均失败时才评估，且须绝对 endpoint MSE 与 CE 同时严格优于父实验。
teacher 辅助初始化至多提供该目标、该求解器下的经验初始化敏感性证据，不能挽救主门失败或证明普遍可达/不可达。
无论结果如何，都不证明 shared writer、ID/OOD、递归状态更新或 Picture Memory 正式成功。
