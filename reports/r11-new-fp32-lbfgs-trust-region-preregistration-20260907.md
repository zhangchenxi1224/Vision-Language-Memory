# R11_new FP32 L-BFGS trust-region：预注册方案

本方案在任何 L-BFGS DreamLite forward 之前冻结。父实验是已完整交付并推送的 PRP+ 结果（训练 commit `de26e3a…876`，交付 commit `efc2a9c…9ae`）。

## 第一性原理诊断

PRP+ 在完全相同的 target、plateau、FP32 DreamLite、loss、候选半径和预算下，把最终 loss ratio 从单位负梯度父实验的 `0.05279661` 降至 `0.02859119`，相对降低约 `45.85%`。这证明历史方向信息确实缓解了一阶最速下降的长尾；但 64 次更新后五个候选均无法通过固定接受门槛，且仍高于预注册成功阈值 `0.01` 的 `2.86` 倍。

因此，本轮只回答一个问题：显式利用最近局部曲率构造 inverse-Hessian 方向，能否在同一 128-update / 771-forward 上限内跨过 `0.01`？若仍不能，本研究线停止继续微调局部优化器，转向“当前 x_T 像素级参数化是否本身病态”的最小判别实验。

## 唯一科学变量

- 父方法：第 1 步为单位负梯度，此后为 PRP+ 共轭方向。
- 本轮：第 1 步仍为同一单位负梯度；从第 2 步起使用确定性 L-BFGS 两环递推。
  - 每次已接受状态转移形成 `s_k=x_k-x_(k-1)`、`y_k=g_k-g_(k-1)`。
  - 仅当 `<s_k,y_k> > 1e-8 ||s_k|| ||y_k||` 时接纳曲率对。
  - 按时间顺序保留最新 10 个曲率对。
  - 两环递推的向量和标量计算均在 CPU float64 完成；raw direction 只在单位化前转换一次 FP32。
  - 初始尺度 `gamma=<s_last,y_last>/<y_last,y_last>`；无历史时 `gamma=1`。
  - 若方向非有限或不是下降方向，清空历史并确定性回退到 `-g_k`。
  - 候选方向为 FP32 `unit(p_k)`。

固定不变：同一固定 self-generated target、同一 plateau、同一模型快照与 FP32 lift、同一 endpoint MSE、同一五个候选半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`、同一全候选选择与 `1e-6` 相对改善接受规则、同一 128-update 上限、同一 `<=0.01` 成功阈值、无 torch optimizer、无梯度裁剪、Reader 禁用及全部解释边界。

## 运行与审计门槛

- technical preflight：1 次梯度、5 个候选、1 次更新；共 9 次完整前向、1 次反向、0 optimizer、0 Reader。首轮梯度、方向、五个候选、loss 与选择必须和 PRP+ 父实验逐字段精确一致。
- formal：最多 128 次更新、771 次完整前向、128 次反向。逐轮保存 current x/endpoint、gradient、曲率 `s/y`、接纳判据、历史索引、gamma、rho/alpha/beta、raw/unit direction、五个候选、checkpoint 和接受图像。
- 独立审计从保存张量重新形成每个曲率对并重跑两环递推，然后核验候选、loss、选择、接受链、checkpoint、计数和最终 bitwise replay；模型快照必须前后不变。
- 外部审计哈希绑定 PRP+ 的 result、manifest、terminal、inventory、trace、tensor bundle、最后 checkpoint、完整 archive 与父审计，并独立比较首轮全部共同字段。

## 预注册判定

1. `fixed_target_trust_region_success`：最终 ratio `<=0.01`。只说明单 target oracle 优化通过；下一步必须预注册 multi-target / non-oracle writer 可行性，仍不得称为 Picture Memory 成功。
2. `lbfgs_material_improvement_only`：未成功，但最终 ratio `<=0.0271616277`，即相对 PRP+ 再改善至少 5%。说明 inverse-curvature 有帮助但不足；交付后转向可控参数化判别，不追加事后优化器调参。
3. `lbfgs_no_matched_budget_advantage`：有进展但未达到上述 5% 改善。否定局部一阶/二阶方向选择是当前主因，转向可控参数化判别。
4. `lbfgs_stalled`：无可接受进展。否定本 L-BFGS 方向并转向可控参数化判别；不得事后修改 memory、曲率阈值、半径或成功门槛。
5. `technical_failure`：只修复违反的绑定、递推、重放、计数、精度或 inventory，在全新目录重跑；不得解释为科学结果。

无论单目标结果如何，本轮 `formal_success=false`、`phase2_allowed=false`：它仍使用单个 oracle endpoint，没有 Reader、共享 writer、多 target、多 seed、ID/OOD、SET/overwrite 或因果替图评测。
