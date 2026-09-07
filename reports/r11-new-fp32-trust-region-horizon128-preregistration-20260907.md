# R11_new FP32 trust-region horizon-128 预注册

状态：在任何 horizon-128 DreamLite forward 之前冻结。

## 已观察事实

已交付的 32-update formal 在固定 FP32 target 上通过全部工程与独立审计：32 次更新全部接受，endpoint MSE ratio 从 `1.0` 严格单调降至 `0.12737637992938525`，最后一轮仍有 `2.4417%` 相对改善，停止原因仅为 `maximum_iterations`。该结果不是固定目标成功，也不是 Picture Memory 成功。

## 第一性原理假设

当前证据同时支持两种解释：

1. FP32 一阶方向正确，32 次预算不足；继续相同优化可以达到 ratio `<=0.01`。
2. 一阶方法虽持续下降，但会在 `0.01` 之前停滞；需要曲率感知方向，而非更多相同步骤。

本轮只增加迭代预算来区分二者，不改变模型或任务。

## 唯一科学变量

`maximum_formal_iterations: 32 -> 128`。

以下全部保持逐字段相同：固定 target、plateau、FP32 lift、DreamLite/VAE/文本编码器、source latent、prompt embedding、scheduler、MSE loss、单位负梯度、半径顺序 `(0.1,0.03,0.01,0.003,0.001)`、全候选评测、选择/接受规则、停止阈值 `0.01`、checkpoint 频率、最终重放、Reader 禁用和所有解释边界。

## 运行与审计

- technical preflight：1 update、9 full-chain forward、1 backward、0 optimizer、0 Reader。
- formal：从同一个 plateau 重新开始，最多 128 update、771 full-chain forward、128 backward、0 optimizer、0 Reader。
- 每个接受状态、梯度、方向、五个候选 endpoint 与 checkpoint 均保存并哈希。
- formal 分类只有在独立审计确认新轨迹前 32 条 iteration 与前 160 条 candidate **逐字节等于**已交付 parent 后才被接受。
- 若前缀不一致，视为技术失败；不得解释为科学结果。

## 预注册判定

- `fixed_target_trust_region_success`：最终 ratio `<=0.01`，accepted loss 严格单调、最终 replay 逐位相等且 parent prefix 审计通过。
- `strong_capture_reached_only`：最终 ratio `<=0.1` 但 `>0.01`。
- `monotone_progress_above_capture`：有接受更新且最终 ratio 仍 `>0.1`。
- `trust_region_stalled`：没有可接受候选而提前停止。
- `technical_failure`：任一配置、parent、前缀、计数、精度、snapshot、inventory 或 replay 约束失败。

## 决策边界

- 达到固定目标成功：只说明单 target oracle endpoint 可优化；下一步必须另行预注册 multi-target / non-oracle writer，不得直接宣称 Picture Memory、Reader、ID/OOD 或 overwrite 成功。
- 未达到：否定“128 步以内仅增加预算足够”，下一项才允许测试 FP32 曲率感知/共轭方向；不得回到 BF16 Adam，也不得修改 target、loss 或阈值追求阳性。

锁定配置：`configs/experiments/r11_new_fp32_trust_region_horizon128_target01.json`。
