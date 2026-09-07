# R11_new FP32 trust-region horizon-128：完整结果交付

## 核心结论

本轮工程执行、完整产物审计和 32 步父实验前缀复现全部通过。相同的一阶 trust-region 算法在 128 步内达到 **strong capture**，但仍未达到固定目标成功门槛，因而**不是 Picture Memory 成功**。

- 128/128 次更新全部接受，accepted loss 严格单调下降。
- endpoint MSE 从 `1.7977445168e-5` 降至 `9.4914815918e-7`，相对下降 `94.7203%`。
- loss ratio 从 `1.0` 降至 `0.05279661`；第 40 次更新首次达到 strong-capture 门槛 `<=0.1`，但未达到固定目标成功门槛 `<=0.01`。
- 前 32 条 iteration 与前 160 条 candidate 记录和已交付父实验逐字节一致，排除了“延长预算时算法或早期轨迹发生漂移”。
- 预注册分类为 `strong_capture_reached_only`；`fixed_target_optimization_success=false`、`formal_success=false`、`phase2_allowed=false`。

第一性原理判断：延长预算确实有效，因此 32 步上限是上一轮未达到 strong capture 的直接原因；但第 64→128 步 loss ratio 只从 `0.06439` 降至 `0.05280`，后 32 步主要选择最小半径 `0.001`，平均单步相对改善约 `0.280%`。这说明相同的一阶归一化方向已进入长尾区，128 步仍不足以完成精确固定目标优化。按预注册决策表，下一项最小判别实验应在同一目标、同一 FP32 映射和同一 loss 下测试曲率感知/共轭方向更新，而不是继续增加相同一阶预算或回到 BF16 Adam。

## 固定实验口径

- 与 32 步父实验使用同一个 self-generated target、同一个 plateau 和同一模型快照。
- 仅优化 FP32 `student_x_T` residual；DreamLite、VAE、文本编码器和 Reader 全部冻结。
- loss：FP32 DreamLite endpoint 与固定 FP32 teacher endpoint 的逐元素 MSE。
- 每轮计算单位负梯度，完整评测固定半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`，选择 loss 最小者。
- 不使用 torch optimizer、不做梯度裁剪、不调用 Reader。
- 唯一科学改动是正式更新上限从 32 增至 128；成功阈值、候选、接受规则和审计边界均不变。

## 结果摘要

| 项目 | 技术预检 | 正式实验 |
| --- | ---: | ---: |
| 完整前向 | 9 | 771 |
| 反向 | 1 | 128 |
| 参数更新 | 1 | 128 |
| optimizer step | 0 | 0 |
| Reader forward | 0 | 0 |
| 初始 loss ratio | 1.0 | 1.0 |
| 最终 loss ratio | 0.68736614 | 0.05279661 |
| 独立 inventory/audit | 通过，27 个产物 | 通过，281 个产物 |

正式实验选择半径次数：`0.03×6`、`0.01×11`、`0.003×23`、`0.001×88`；没有选择 `0.1`。关键断点的 loss ratio 为：第 32 步 `0.12737638`、第 64 步 `0.06438662`、第 96 步 `0.05775867`、第 128 步 `0.05279661`。

## 图表

- [loss 主图](figures/trust_region_loss.png)：黑线为接受轨迹，彩色点为每轮五个半径的全部候选；竖线标出父实验 32 步终点和首次 strong capture 的第 40 步。
- [优化动态](figures/trust_region_dynamics.png)：选择半径、梯度范数、单步相对改善以及离 plateau 的 residual L2。
- [阶段对比](figures/trust_region_horizon_comparison.png)：四个 32 步阶段的端点 ratio 与半径选择组成，直接展示后半程长尾。
- [图像演化](figures/trust_region_montage.png)：展示固定目标、FP32 teacher、多个接受状态和最终重放；肉眼相似不能替代连续张量 loss 与因果评测。

## 可复核产物

- [派生汇总](derived/summary.json)：完整数值、阶段统计、双层独立审计、父实验前缀复现和图像统计。
- [formal 指标与日志](formal/)：配置、环境、逐步/候选指标、结果、终止记录、日志及接受图像。
- [preflight 指标与日志](preflight/)：独立技术预检的同类材料。
- [父实验前缀审计](external_parent_prefix_audit.json)：远端独立审计原件；本地脚本还会使用父实验原始包重新计算一次。
- [交付清单](DELIVERY_MANIFEST.json)：GitHub 交付文件的字节数与 SHA-256。
- [可复现渲染/审计脚本](render_results.py)：无损重组原始包，校验每个 inventory、张量/checkpoint、候选选择、重放和父实验前缀，再生成图表与汇总。

完整原始包为 `616,876,521` bytes，SHA-256：

`d907382f050eb7e5a452fe8632f079c746f2d6cd1f9519a667c95dae9e740483`

为避免平台传输限制并满足 GitHub 单文件限制，包按文件名顺序无损切为 14 个 `raw/*.github-part-XX.bin`：前 13 片各 `47,185,920` bytes，最后一片 `3,459,561` bytes。按 `00` 到 `13` 顺序串联后必须得到上述字节数和完整 SHA；每片哈希记录在 `derived/summary.json` 与 `DELIVERY_MANIFEST.json`。

## 科学边界与下一步

本轮只证明：对**单个固定 oracle target**，FP32 DreamLite 映射上的可微一阶方向可以稳定进入 strong-capture 区间。它没有评测 Reader、事件到状态写入、共享 writer、多 target、多 seed、ID/OOD、SET/overwrite 或因果图像干预，因此不能支持 Picture Memory 成功主张。

下一轮应只替换优化方向为预注册的曲率感知/共轭方向，同时锁定当前目标、plateau、精度、loss、候选评测和 `<=0.01` 成功门槛。只有单目标门槛通过后，才进入多 target、非 oracle writer 与 Reader 的可行性验证。
