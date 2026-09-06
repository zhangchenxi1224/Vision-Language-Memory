# R11_new：Teacher-matched initialization 真实结果

训练锁定提交：`5bffcb9a1605c7cdef77df1126daca6f9578bdda`。固定 target 1，唯一干预是使用 canonical teacher 反解 initial xT；完整四步 DreamLite 保持冻结，只优化 FP32 xT。

## 结论边界

工程门：**通过**；teacher replay：**通过**；距离门：**失败**；Reader 门：**失败**；本轮 bridge 诊断：**失败**。

本轮使用答案相关 teacher 辅助初始化。无论诊断结果如何，`formal_success=false`、`phase2_allowed=false`。Phase 1A 仍为 **6/8**；不能将本轮 target 1 或 canonical R11 结果拼入主线通过数，也不能声称长期 memory/shared writer 已学会。

## 实际门槛与结果

| raw256 指标 | 本轮真实值 | 固定门槛 |
| --- | ---: | --- |
| MSE / 新 M0 MSE | 0.3541232337 | ≤ 0.01 |
| L2 / 新 M0 L2 | 0.5950825436 | ≤ 0.1 |
| RMSE / teacher population std | 0.3915293637 | ≤ 0.1 |
| Reader accuracy | 0/4 | 4/4 |
| Reader mean CE | 10.97591446 | 原样报告，不新增 endpoint CE 成功阈值 |

四视图是同一个 target 的固定排列，并非四个独立样本。仅 raw256 判主门；中间 checkpoint、最低 train loss、二级改善均不能挽救主门失败。

## 真实训练轨迹

![距离轨迹](distance_trajectory.png)

Receipt 第 u 行的 loss 在更新前测量，横坐标为 u−1；checkpoint 保存更新后 endpoint，横坐标为 u。新 M0 是新初始化完整跑完四步后、零更新的 endpoint，不是初始 flow state 或 teacher。父实验具有不同 M0，因此只画父实验绝对 MSE 参考线，不作同分母比值对比。

| 已完成更新数 | checkpoint MSE | MSE / 新 M0 | teacher NRMSE |
| --- | ---: | ---: | ---: |
| 0 | 0.1855299026 | 1 | 0.6579412686 |
| 64 | 0.07660387456 | 0.4128923343 | 0.4227713394 |
| 128 | 0.07006178796 | 0.3776307053 | 0.4043158601 |
| 192 | 0.06672922522 | 0.359668303 | 0.394582855 |
| 256 | 0.06570044905 | 0.3541232337 | 0.3915293637 |

![优化器与梯度](optimizer_diagnostics.png)

真实 receipts 共 256 行。Adam 前 128 次学习率 0.05，之后按锁定 cosine 曲线衰减；第 256 次学习率为零，但仍须有有限非零梯度与第 256 次 Adam 计数。图中零值保持为零。

![Reader 结果](reader_transfer.png)

![真实 checkpoint 图像](checkpoint_images.png)

图像仅展示保存的真实 RGB；人类可读性不是本实验优化目标或门槛。

## 二级审计与下一项决策

主决策：`distance_fail_reader_fail_change_one_solver_factor`。

二级决策：`distance_fail_reader_fail_secondary_init_improves`。资格 eligible=`true`，通过 passed=`true`。

二级仅在技术/teacher 有效且两主门均失败时参与解释，要求 endpoint MSE 严格小于父值 `0.09553645551204681`，且 Reader CE 严格小于父值 `25.536474171257463`。只改善一项或相等均不通过，二级不是成功门。

| 距离 / Reader | 另行预注册的候选下一项 |
| --- | --- |
| 通过 / 通过 | 答案无关 learned initializer / inverse writer |
| 通过 / 失败 | teacher 邻域 Reader 鲁棒性 |
| 失败 / 通过 | 原始 QA 目标诊断 |
| 失败 / 失败且二级通过 | 隔离答案无关初始化或 conditioning |
| 失败 / 失败且二级未通过 | reachable-teacher control，区分优化失败与 teacher/flow 不匹配 |

这些是需新预注册的候选，不是此配置内追加实验许可。初始化接近 teacher 不证明全程局部可控，也不证明初始化是主导根因；失败不证明数学不可达。

## 证据与复核范围

Preflight 为一次完整四步前向、一次反向、零更新，不评判 bridge 结果。源目录 inventory、聚合 comparison/RAW 及所有绑定文件字节已检查；本地另从原始 checkpoint tensor 复算距离并按既定容差核验，从原始四选项 logits 复算 CE/正确率，并对原始聚合指标重算主次决策。

所有表格、图表和展示字段统一使用已哈希绑定的原始 record/聚合统计。FP32 reduction 在不同平台或线程实现下可能出现末位差异；本地 CPU 复算值仅用于原有严格容差校验，单独保存在 `checkpoints[*].local_cpu_recomputed_distance_statistics`，不替换远端原值、不改变科学门槛。机器可读文件同时保留两组明确标注的数值。

CUDA 初始化与每 checkpoint 起点闭包由独立 CUDA aggregator 核验。本渲染器仅核对其证据文件字节，不在 CPU 上伪装重跑 CUDA 逐位算术。初始化接近阈值只约束 step 0，其余起点由各自当前 xT 决定；不要求旧 Gaussian M0 或旧优化前缀相等。

逐行学习率的精确合同由原 Linux 运行环境的独立 aggregator 验证并以文件哈希绑定。Windows/Linux 的 cosine 库可产生极小末位差异，渲染器不以另一平台的重算替换原始学习率，也不放宽原有门槛；图表保留原 receipts 数值。

机器可读数据：[training_diagnostics.json](training_diagnostics.json)；逐行原始 Reader 数据和带明确前后坐标的 receipts/checkpoints 均在其中。交付文件哈希：[DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json)。
