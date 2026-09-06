# R11_new 可达 endpoint 优化正对照

## 核心结论

**工程链路完整，但预注册的可达目标正对照失败。** 冻结 DreamLite 自己生成了目标 endpoint，因此该目标按构造必然存在精确解；然而从既定 source-only `xT` 初始化出发，256 步 Adam 只把 endpoint MSE 从 `0.02440103` 降至 `0.01938320`（仅改善 `20.56%`），没有达到预注册要求的 `99%` 改善。Picture Memory 科学成功仍为 `false`，Phase 2 仍为 `false`。

这排除了“只是 canonical teacher 位于冻结生成器值域外”这一单一解释：即使目标来自同一个冻结映射，当前的全局反演过程仍无法从既定初始化找到它。当前主瓶颈应定位为 **四步非线性 DreamLite 映射下的优化吸引域/局部 Jacobian 病态**，而不是链路不回传或梯度裁剪。

## 固定实验设计

- 固定 target 1、灰色 source、实际 conditioning 文本 `no changes`，使用冻结 DreamLite 四步完整链路。
- 教师噪声由固定 seed 构造，`teacher_xT = source_xT + unit-RMS noise`；目标为 `F_identity(teacher_xT)`。
- 教师 target 生成与 replay 逐位一致，学生恢复到原 source-only `xT` 后才建立优化器。
- 仅训练学生 `xT`；Adam、256 步、前 128 步 LR `0.05`，后 128 步 cosine 衰减至 0；不裁剪梯度。
- Reader 全程 0 次 forward；本轮只诊断 writer 的可优化性，不测试语义记忆。
- 主 endpoint 固定为 raw step 256，禁止挑 best checkpoint。

## 结果

| Optimizer step | Endpoint MSE | MSE / M0 | L2 / initial L2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.02440103 | 1.0000 | 1.0000 |
| 64 | 0.01987309 | 0.8144 | 0.9025 |
| 128 | 0.01957356 | 0.8022 | 0.8956 |
| 192 | 0.01944431 | 0.7969 | 0.8927 |
| 256 | 0.01938320 | 0.7944 | 0.8913 |

预注册门槛要求同时满足：endpoint MSE `≤ 0.001`、MSE/M0 `≤ 0.01`、L2/initial L2 `≤ 0.1`。实际三项分别为 `0.01938320`、`0.79436`、`0.89127`，全部失败。

工程计数严格为 263 次完整 DreamLite forward、256 次 backward、256 次 optimizer step、0 次 Reader forward。所有 256 步梯度都有限且非零，最小梯度范数 `6.2489e-05`，最小非零元素比例 `99.9008%`；没有触发梯度裁剪。训练耗时 289.93 秒，5 个预注册 checkpoint 全部存在。

## 从第一性原理解释失败

已知精确解是 teacher `xT`，但学生 `xT` 到该精确解的 MSE 从 `1.0000` 反而增加到 `1.0671`；最终学生位移与正确 teacher 方向的余弦仅 `0.1186`。也就是说，梯度确实在改变 `xT` 并降低一部分图像端误差，却把参数带到了另一个局部方向，随后在约 `0.0194` 平台化。

因此：

- 不是“没有梯度”：梯度覆盖几乎所有 `xT` 元素且连续 256 步存在。
- 不是“总被裁剪”：本轮完全没有裁剪。
- 不是“目标不存在”：目标就是同一冻结映射从已保存 teacher `xT` 生成的，replay 逐位相同。
- 也不能简单归因于“步数太少”：主要下降发生在前 64 步，此后 192 步只从 `0.019873` 降至 `0.019383`，且既定 LR 已衰减到 0。增加相同步骤不能解决错误吸引域，但本轮也不声称数学上不可优化。

图像肉眼均呈近乎均匀的灰绿色，这是非语义几何正对照的预期外观；不能据此宣称模型学会了实体、preference 或 Picture Memory。

## 下一项最小判别实验

暂缓 LoRA、全量数据和 ID/OOD 评测。固定复用本轮 target artifact 与四个 tensor SHA-256，不重新采样；保持同一 optimizer、步数、损失和 gate，只比较两个预注册初始化：

- `x_init = source + 0.50 × (teacher - source)`：中距离 warm start；
- `x_init = source + 0.99 × (teacher - source)`：近邻正对照。

若 `0.99` 仍失败，优先检查局部 Jacobian、精度和优化器参数化；若 `0.99` 通过而 `0.50` 失败，说明吸引域很窄；若两者均通过，则 source-only 初始化位于可优化盆地之外。只有完成这一判别后，才决定是重参数化/continuation，还是最小 generator adaptation。

## 审计与可复现性边界

- 远端原运行环境（Linux，PyTorch `2.7.0a0+ecf3bae40a.nv25.2`）对 preflight 20 项与 formal 29 项 inventory 的独立 audit 均通过。
- 原始双阶段归档 SHA-256：`41741aaf7e85022d4a1440a7a0b07ed580151201fec01e10db82205427dc74fb`；下载后本地哈希完全一致。
- Windows/PyTorch 2.11 本地复审能够重算 inventory、张量哈希、构造公式、256 行 metrics 与 endpoint 指标；但按同 seed 重新生成并归一化的噪声与原张量最大相差 `2.3842e-07`（约 1 个 FP32 ULP），因此不能宣称跨 PyTorch/OS 逐位 RNG replay。存储噪声本身 RMS 为 `0.999999999845`，且 `teacher_xT = source + stored_noise` 逐位成立。
- 后续实验直接复用已哈希 target artifact，避免将不同运行时的 RNG 归一化误当作同一输入。

## 交付内容

- `preflight/`、`formal/`：结果、终态、完整 metrics、日志、环境、manifest、模型快照校验和 inventory。
- `images/`、`figures/`：实际 target/checkpoint 图像、loss/梯度/潜变量几何图和 montage。
- `derived/summary.json`：从原始 checkpoint 重新计算的结构化摘要及跨运行时审计记录。
- `raw/`：包含所有条件张量、target、checkpoint 和运行产物的完整原始归档。
- `render_results.py`：从原始归档验证并重建摘要、图表与 `DELIVERY_MANIFEST.json`。
