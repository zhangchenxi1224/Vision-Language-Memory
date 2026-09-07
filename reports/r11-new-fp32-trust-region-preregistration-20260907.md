# R11_new FP32 trust-region：预注册

状态：activation-precision control 完成、归档并同步 GitHub 后，任何本轮 trust-region
DreamLite forward 之前锁定。

## 第一性原理问题

上一轮只证明了两点：同一 BF16-valued DreamLite 映射在 FP32 算术下具有宽得多的
teacher capture 区域，且 plateau 负梯度在 6/6 个有限差分尺度上方向正确。它尚未证明
多步更新能够稳定累积。因此本轮只问：固定 FP32 映射后，最简单的确定性一阶搜索能否
从同一 plateau 真正进入 fixed target basin？

## 固定输入与唯一算法

- 沿用完全相同的数据、模型 snapshots、source/conditioning、teacher target、plateau
  checkpoint、四步 scheduler 和同一次 BF16→FP32 无损提升。
- 参数化为 `student_xT = fixed_plateau_xT + residual_delta`；delta 初始为零。
- 每轮在当前点计算 endpoint-MSE 对 `student_xT` 的完整 FP32 梯度，方向固定为单位
  L2 负梯度。
- 必须评测全部固定半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`；选择 loss 最小者，
  完全相同时选更小半径。
- 相对改善至少 `1e-6` 才接受；无可接受候选即停止。formal 最多 32 轮；loss ratio
  `<=.01` 立即停止。
- 不使用 torch optimizer，不做 gradient clipping；单位化就是预注册更新规则。
- teacher `x_T` 只用于一次生成冻结 teacher endpoint，不得进入更新方向、候选构造或
  选择。Reader 不执行。

## 阳性控制与执行计数

第 0 轮 FP32 teacher、plateau、gradient、direction，以及半径 `.1/.01/.001` 的
candidate xT/endpoint hash 和 loss ratio，必须逐项复现上一轮 precision-control。

- technical preflight：1 个 BF16 teacher replay + 1 个 FP32 teacher + 1 个梯度 forward
  + 5 个 candidate forward + 1 个 final replay = 9 forward；1 backward；预期 1 次
  parameter update；0 optimizer/Reader。
- formal 若执行 N 轮：`3 + 6N` forward、N backward、accepted-count 次显式 parameter
  update；N 不超过 32，因此最多 195 forward。

## 预注册结论

- `fixed_target_trust_region_success`：最终 loss ratio `<=.01`、至少一次更新、接受序列
  严格单调、每个 current endpoint 与 final endpoint 独立 replay bitwise 一致。
- `strong_capture_reached_only`：最终 ratio `<=.1` 但未达到 `.01`。
- `monotone_progress_above_capture`：有严格进步但 ratio `>.1`。
- `trust_region_stalled`：无有效更新或最终不优于 plateau。

所有候选 endpoint、梯度、方向、current/final tensors、接受 checkpoint、代表图片、
日志和哈希必须落盘并由独立审计重算。

## 解释边界

即使得到 `fixed_target_trust_region_success`，也只证明单个自生成 endpoint target 在
FP32 frozen DreamLite 下可优化。它不是事件→状态 writer、不是 Reader 记忆准确率、不是
ID/OOD、不是多 seed，也不是最终 Picture Memory 成功；所以本轮始终
`formal_success=false`、`phase2_allowed=false`。成功后只能进入多目标/非 oracle writer
可行性实验，不能直接宣布主任务完成。
