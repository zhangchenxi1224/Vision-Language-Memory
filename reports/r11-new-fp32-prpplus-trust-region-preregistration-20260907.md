# R11_new FP32 PRP+ trust-region：预注册方案

本方案在任何 PRP+ DreamLite forward 之前冻结。父实验是已完整交付并推送的 horizon-128 结果（训练 commit `8efeabc1…65d`，交付 commit `88dddced…76c`）。

## 第一性原理诊断

父实验 128/128 次更新严格单调下降，loss ratio 从 `1.0` 降至 `0.05279661`，第 40 步进入 strong capture，但没有达到固定目标门槛 `0.01`。它同时呈现三个关键现象：

1. 梯度始终稠密、非零且可执行，所以根因不是断梯度、错误符号或 DreamLite 完全不可控。
2. 第 65–128 步连续选择最小半径 `0.001`；后 32 步平均相对改善仅约 `0.280%`，说明相同最速下降方向进入长尾。
3. 梯度范数后半程近乎恒定、loss 仍以小幅锯齿式改善，符合病态曲率下最速下降在高/低曲率方向间反复修正的可检验特征。

因此，本轮不再增加相同一阶预算，也不修改 target、loss 或成功门槛；只测试“保留历史方向是否能减少之字形并在相同预算下更接近目标”。

## 唯一科学变量

- 父方法：`d_k = unit(-g_k)`。
- 本轮：PRP+ 非线性共轭梯度。
  - `p_0 = -g_0`；因此第 1 步必须与父实验完全一致。
  - `y_k = g_k - g_(k-1)`。
  - `beta_raw = <g_k,y_k>/<g_(k-1),g_(k-1)>`，标量用 float64 计算。
  - `beta = max(0,beta_raw)`，不加 epsilon、不设上限。
  - `p_k = -g_k + beta*p_(k-1)`。
  - 若 `p_k` 非有限或 `<g_k,p_k> >= 0`，确定性重启为 `p_k=-g_k, beta=0`。
  - 候选方向为 FP32 `unit(p_k)`。

固定不变：同一 target、同一 plateau、同一模型快照与 FP32 lift、同一 endpoint MSE、同一候选半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`、同一全候选选择/接受规则、同一 128 更新上限、同一 `<=0.01` 成功阈值、Reader 禁用和解释边界。

## 运行与审计门槛

- technical preflight：1 次梯度、5 个候选、1 次更新；共 9 次完整前向、1 次反向、0 optimizer、0 Reader。梯度、方向、五个候选和选择必须与父实验第 1 步精确一致。
- formal：最多 128 次更新、771 次完整前向、128 次反向；每轮保存梯度、前一梯度、raw 共轭方向、单位方向、beta、重启原因、五个候选、checkpoint 和接受图像。
- 每个当前 endpoint 和最终 endpoint 必须独立 bitwise replay；模型快照必须前后不变。
- 独立审计从保存张量重新计算每轮 PRP+ 递推、方向、候选、loss、选择、接受链、checkpoint、计数与分类。
- 外部审计额外绑定 horizon-128 父实验的 result/manifest/terminal/inventory/trace/tensor/checkpoint/archive 哈希，并比较第 1 步全部共同字段。

## 预注册判定

1. `fixed_target_trust_region_success`：最终 ratio `<=0.01`。只允许进入多 target / 非 oracle writer 可行性实验；仍不得称为 Picture Memory 成功。
2. `prpplus_material_improvement_only`：未成功，但最终 ratio `<=0.0501567794`，即相对父实验至少改善 5%。结论是共轭信息有效但不足；下一项只测试 L-BFGS inverse-curvature。
3. `prpplus_no_matched_budget_advantage`：有进展但未达到上述 5% matched-budget 改善。否定 PRP+ 是本长尾的解决方案，下一项只测试 L-BFGS。
4. `prpplus_stalled`：没有可接受进展。否定该递推并转向 L-BFGS。
5. `technical_failure`：只修复违反的绑定、递推、重放、计数、精度或 inventory 后在新目录重跑，不解释为科学结果。

无论单目标结果如何，本轮 `formal_success=false`、`phase2_allowed=false`：它仍使用单个 oracle endpoint，没有 Reader、共享 writer、多 target、多 seed、ID/OOD、SET/overwrite 或因果替图评测。
