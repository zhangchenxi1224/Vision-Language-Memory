# R11_new FP32 trust-region：完整结果交付

## 核心结论

本轮工程执行与独立审计全部通过，但**没有达到固定目标优化成功，更不是 Picture Memory 成功**。

- 32/32 次更新全部接受，accepted loss 严格单调下降。
- endpoint MSE 从 `1.7977445168e-5` 降至 `2.2899018859e-6`，相对下降 `87.2624%`。
- 最终 loss ratio 为 `0.12737638`；未达到 strong capture 门槛 `0.1`，更未达到固定目标成功门槛 `0.01`。
- 预注册分类：`monotone_progress_above_capture`。
- `fixed_target_optimization_success=false`、`formal_success=false`、`phase2_allowed=false`。

第一性原理判断：FP32 DreamLite 映射上的梯度是可执行的，问题已经不是“完全没有梯度”或“优化方向错误”；本轮只因达到预注册的 32 次更新上限而停止，末步仍取得 `2.4417%` 相对改善。因此下一项最小判别实验应只增加迭代预算，保持目标、模型、FP32 映射、loss、半径集合与选择规则全部不变。

## 固定实验口径

- 固定同一个 self-generated target 和同一个已交付 plateau。
- 仅优化 FP32 `student_x_T` residual；DreamLite、VAE、文本编码器与 Reader 全部冻结。
- loss：FP32 DreamLite endpoint 与固定 FP32 teacher endpoint 的逐元素 MSE。
- 每轮计算单位负梯度，并完整评测固定半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`；选 loss 最小者。
- 不使用 torch optimizer、不做梯度裁剪、不调用 Reader。
- formal 最多 32 次更新；达到 ratio `<=0.01` 或没有可接受候选时提前停止。

## 结果摘要

| 项目 | 技术预检 | 正式实验 |
|---|---:|---:|
| 完整前向 | 9 | 195 |
| 反向 | 1 | 32 |
| 参数更新 | 1 | 32 |
| optimizer step | 0 | 0 |
| Reader forward | 0 | 0 |
| 初始 loss ratio | 1.0 | 1.0 |
| 最终 loss ratio | 0.68736614 | 0.12737638 |
| 独立审计 | 通过，27 个产物 | 通过，89 个产物 |

正式实验选择半径次数：`0.03×4`、`0.01×11`、`0.003×16`、`0.001×1`。末步梯度范数仍为 `2.0562e-5`，末步选择半径重新增至 `0.01`，没有出现“所有候选均无法下降”的停滞证据。

原 4×H200 实例在首轮 formal 第 18 次更新后被平台停止；该不完整轮没有 `terminal.json`，所以未被当作科学结果。它的前 18 条 iteration 和前 90 条 candidate 记录与本轮逐条完全一致，说明数值轨迹可复现，中断属于基础设施事件。

## 图表

- [loss 主图](figures/trust_region_loss.png)：黑线为每轮接受状态；彩色点为五个半径的全部候选；虚线分别是 `0.1` 与 `0.01` 门槛。
- [优化动态](figures/trust_region_dynamics.png)：展示选择半径、梯度范数、相对改善与离 plateau 的 residual L2。
- [图像演化](figures/trust_region_montage.png)：人眼下几乎不可区分，但哈希与连续 endpoint MSE 均发生变化；不能用肉眼相似代替数值/因果评测。

## 可复核产物

- [派生汇总](derived/summary.json)：完整数值、独立审计、资源中断前缀一致性与图像统计。
- [formal 指标与日志](formal/)：`iteration_metrics.jsonl`、`candidate_metrics.jsonl`、配置、环境、日志、结果、终止记录及全部接受图像。
- [preflight 指标与日志](preflight/)：独立技术预检的同类材料。
- [交付清单](DELIVERY_MANIFEST.json)：GitHub 交付文件的字节数与 SHA-256。
- [可复现渲染/审计脚本](render_results.py)：从原始包重新校验 inventory、张量、checkpoint、候选选择、loss、重放与图表。

完整 round02 原始包为 `167,727,523` bytes，SHA-256：

`e8b8630ac8177ef51ea4c812cd513ae295032db4f2b523a7805bfb7c3002397b`

因 GitHub 单文件限制，它按文件名顺序无损拆成两片：

1. `raw/r11-new-trust-bebeffd-20260907-round02.tar.gz.part-00.bin`，`83,886,080` bytes，SHA-256 `c34360c72f13fa1bacd75133314d6463b7e381489bffae83770da676a337761f`。
2. `raw/r11-new-trust-bebeffd-20260907-round02.tar.gz.part-01.bin`，`83,841,443` bytes，SHA-256 `5e40804a497b45f1abfbdb3e932e6877846cd6158f1ba3dbcf4d96589e3f40cc`。

按 `part-00`、`part-01` 顺序串联即可逐字节恢复原始 `.tar.gz`；恢复后必须核对上述完整包 SHA-256。首轮基础设施中断包也完整保存在 `raw/r11-new-trust-bebeffd-20260907-round01-interrupted.tar.gz`，SHA-256 `b71aa8fc8db6c046cd8571683682d9969ac129f9ba16bb98673e9be2ab9887de`。

## 科学边界与下一步

本轮只证明：对于**单个固定 oracle target**，FP32 下的归一化一阶方向可以持续降低 DreamLite endpoint MSE。它没有评测 Reader、事件到状态写入、共享 writer、多 target、多 seed、ID/OOD 或 overwrite，因此不支持 Picture Memory 成功主张。

下一轮只把 formal 上限从 32 增至 128，其他因素全部固定，并要求前 32 步逐条复现本轮。若 ratio 达到 `<=0.01`，才能进入多 target/非 oracle writer 可行性；若提前停滞或 128 步后仍明显高于门槛，则应否定“仅增加预算足够”，再测试曲率感知更新，而不是回到 BF16 Adam 或修改评测口径。
