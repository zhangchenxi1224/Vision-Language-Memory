# R11_new canonical teacher 4↔7 因果换图审计

## 核心结论

**D0 通过：canonical R11 teacher 图具有强样本区分力，不是对任意问题都奏效的通用视觉触发器。** 但本轮没有训练参数，完整 Picture Memory 科学成功仍为 `false`，Phase 2 与共享训练仍未开放。

这一步回答的是：此前直接 VAE-latent 优化得到的“成功图片”，究竟真的分别携带 blue/green 信息，还是只让 Reader 更容易顺着问题猜答案。固定同槽位、同四个候选的两个样本后，交换图片会让预测随图片内容切换，空白图则稳定回到 `no active preference`，因此在这对样本上排除了通用触发与纯问题先验解释。

## 固定设计

- Target 4：正确答案 `blue`；Target 7：正确答案 `green`。
- 每条 query 读取三种图：自己的 teacher、另一条样本的 donor、固定灰图 reset。
- 每个条件使用 4 个 reverse-cyclic 候选排列，共 `2 × 3 × 4 = 24` 行、96 次冻结 Qwen3-VL 候选前向。
- 原 FP32 endpoint 经原 VAE BF16 解码后，必须与存档 RGB 逐位相同。
- 0 optimizer step、0 U-Net forward；不生成新数据，不调阈值。

## 有效 Round 02 结果

| Query target | 图像条件 | Mean listwise CE | 准确率 | 四个排列的预测 |
| --- | --- | ---: | ---: | --- |
| 4 / blue | own target-4 | 0.0002836023 | 4/4 | blue ×4 |
| 4 / blue | donor target-7 | 18.4689441 | 0/4 | green ×4 |
| 4 / blue | reset gray | 32.2447910 | 0/4 | no active preference ×4 |
| 7 / green | own target-7 | 0.0000795679 | 4/4 | green ×4 |
| 7 / green | donor target-4 | 14.2546263 | 0/4 | blue ×4 |
| 7 / green | reset gray | 34.6510420 | 0/4 | no active preference ×4 |

两个方向的 own-vs-donor 与 own-vs-reset 均满足：准确率差 `1.0`、4/4 排列 CE 严格改善、相对 CE 改善超过 `99.998%`。Teacher replay gate、distinguishability gate 和 D0 diagnostic gate 全为 `true`。

工程证据：提交 `d0aa3558073167b09f4ddc6087829c94ff32f436`；运行 35.45 秒；observed counters 精确等于 `{U-Net: 0, Reader: 96, optimizer: 0}`；独立远端 audit 与下载后本地 audit 均通过，22 项 inventory 哈希全部一致。

## Round 01 技术失败

提交 `08687bb57f59eb9246fbe3f448485a84fdcca67d` 的首轮生成了 24/24 receipts 和两张逐位一致的 teacher RGB，但 forward hook 挂在未被评分函数调用的外层 `reader`，计数误报为 0，按预注册守门得到 `technical_failed`。没有 `result.json`，不构成有效 D0。唯一修复是把 hook 移到实际执行的 `reader.model` 并在检查前保存计数；样本、结果口径和门槛均未修改。失败目录及 19 项已登记产物完整保留。

## 能证明与不能证明的内容

本轮能证明：这两张现成 teacher 图片对冻结 Reader 有方向正确、随图片交换而切换的因果读取效应，canonical 标签本身不是当前主故障。

本轮不能证明：冻结 DreamLite 能从任意事件生成这些图、共享 writer 能泛化、overwrite 等状态转移已学会，或 ID/OOD 科学门槛已通过。结合前序 identity-conditioning 仅带来约 0.4% 的误差改善，下一项最小判别实验应先用**由同一冻结 DreamLite 映射自身生成的可达 endpoint**做正对照：若同一优化器能拟合可达目标而仍不能拟合 canonical teacher，瓶颈才可收窄到 frozen generator 的可达集合，再讨论 LoRA；若可达目标也失败，应先修优化/Jacobian，而非贸然增加数据。

## 交付内容

- `round02-valid-d0/`：有效轮的原始 logits、结果、终态、manifest、计数、环境与 inventory。
- `round01-technical-failure/`：失败轮终态、完整日志、receipts、manifest 与 inventory。
- `images/` 与 `figures/`：三张实际 Reader 输入及指标/图片可视化。
- `raw/`：两个完整原始运行包。Round 01 SHA-256 `f7524b28c8a2e772a19d37dabe40f8f280e1e29e5a245c277bc6ea245544e0a5`；Round 02 SHA-256 `fdb41c09cf6de0e3cf7560973d8ce2141c396f411644d2a56fcaa3a33b49a01e`。
- `DELIVERY_MANIFEST.json`：本目录所有交付文件的大小与 SHA-256（清单本身除外）。
