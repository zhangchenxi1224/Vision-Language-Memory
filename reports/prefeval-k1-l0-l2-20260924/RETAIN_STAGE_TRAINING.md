# 固定保持 FM 已完成，转入最终 RGB 链生成

2026-09-24，北京时间 08:18 核验。A/B 各完成 2,048 次保持阶段更新，
从各自单写检查点接续，使用本组单写学生实际生成的 64 条 RGB 前缀。
官方 DreamLite FM、目标 latent、优化预算和原生推理设置均沿用登记。

| 核验项 | A | B |
|---|---:|---:|
| 有效更新 | 2,048 | 2,048 |
| 总样本抽取 | 8,192 | 8,192 |
| 初写 / 保持样本 | 4,096 / 4,096 | 4,096 / 4,096 |
| 每条偏好每个训练类型抽取 | 64 | 64 |
| 非有限损失或梯度步骤 | 0 | 0 |
| 零梯度步骤 | 0 | 0 |

轨迹步骤连续且无重复。每个保持位置 1–4 共 448 次，5–10 各 384 次；
即每偏好各位置分别出现 7 或 6 次，这是固定 2,048 步循环预算的实际分布。
保持目标哈希与该组原单写目标完全一致，父检查点也逐一匹配。
实际输入前缀的 128 条链、1,408 张 PNG 及跨步源/目标哈希已在
[完整保持前核验](PRERETAIN_FULL64_RESULTS.md)中保存。

## 训练误差与能力证据分开

以下为前 128 次与后 128 次更新内，各类样本 FM 速度 MSE 的均值：

| 组别 | 初写：前 → 后 | 保持：前 → 后 |
|---|---:|---:|
| A | 0.327243 → 0.230296 | 1.005135 → 0.943926 |
| B | 0.160580 → 0.139925 | 0.533090 → 0.463684 |

该损失描述对目标速度的拟合，不能直接当作偏好准确率。
两组目标分布不同，也不能直接用跨组 MSE 大小排列视觉记忆能力。
保持误差下降有限；是否恢复了可读信息，必须检查重新生成的链，不能由这里宣布成功。

## 最终权重及后续执行

- A：`473f9ec4b02af5693f4df5c74cc851144dc1ed742894f2fc6cdff5185440a01b`
- B：`66001224f511877cc55c750664a9439d888899cbd59103ccf7fe4e6971ca68b7`

上述为实际 `checkpoint-final.pt` 的 SHA-256，已与完成标记核对。
原 Writer 启动会话自动转入 `final-pilot`，新进程 A 3207341、B 3206761，
启动参数明确指向各自 `retain/checkpoint-final.pt`，未重新开始训练。
每条链从灰图开始，每步保存并重新读取 uint8 PNG，再 VAE encode；原生 28 步、CFG=1。
pilot 之后继续生成 dev，两组均为两条固定噪声链。

完整 T1 MCQ 等待任务和自然回答/五问法流水线继续执行。
当前只有训练完成和最终生成已启动的证据，尚无最终保持准确率。
O1/O2 不用于选择检查点或下一轮参数。

原始证据：[完整核验与哈希](evidence/retain-training/retain-training-verification.json)、
[A 轨迹](evidence/retain-training/writer/A/retain/optimization.jsonl)、
[B 轨迹](evidence/retain-training/writer/B/retain/optimization.jsonl)、
[A manifest](evidence/retain-training/writer/A/retain/manifest.json)、
[B manifest](evidence/retain-training/writer/B/retain/manifest.json)。
