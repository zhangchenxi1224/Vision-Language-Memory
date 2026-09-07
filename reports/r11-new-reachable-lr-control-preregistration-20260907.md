# R11_new 固定可达目标低学习率判别：预注册

状态：在 `alpha=0.99, Adam lr=0.05` 的真实过冲结果完成、交付后，且任何低学习率 DreamLite forward 之前锁定。

## 已知事实与研究问题

同一固定可达 target 存在能逐位重放 teacher endpoint 的精确 `teacher xT`。上一轮从
`source + 0.99 × (teacher-source)` 出发时，初始 endpoint MSE 仅为
`0.0001353121`，但 Adam 第一步把 `xT` 移动 `12.1878`，是剩余精确距离
`2.559998` 的 `4.7609` 倍；一步后 loss 放大 `29.0926` 倍，raw step 256
MSE/M0=`4.36391`。这与 Adam 首步近似按参数维数给出固定尺度更新的机制一致。

本轮只回答：**降低 Adam 的 base LR，是否能消除已测得的局部首步过冲，并在同一
alpha=0.99 初始化上拟合同一 teacher endpoint？**

## 唯一自变量

两个 condition 严格按下列顺序独立运行：

1. `lr-005`：base LR=`0.005`；由上一轮线性缩放预测首步 `xT` L2 更新约
   `1.21878`，为剩余距离的 `0.47609`；
2. `lr-001`：base LR=`0.001`；预测首步更新约 `0.243756`，为剩余距离的
   `0.095217`。

每个 condition 都新建 Adam 并从完全相同的 alpha=0.99 `xT` 开始。禁止共享
optimizer state；禁止改变 target、初始化、loss、步数、schedule 形状、checkpoint、
阈值、模型、数据或精度。

## 固定证据链与训练口径

- 直接复制 parent basin formal 的 `fixed_parent_target.pt`，文件 SHA-256：
  `5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；
- source `xT`：`719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`；
- teacher `xT`：`9a5cde0f0f93ba29dc74b2b78878e64dde49a6ec9168cb60f57ad48cdfb31f66`；
- teacher endpoint：`9e627ec840af943d29e3b40a9a8ab796ce23bf07861cb18801360157c4bea0e9`；
- stored noise：`a320afe2438b0ee725de4702b7d7bd7a213c08ad306ea05eef699b292f7b4377`；
- 同一 target 1、灰色 source、conditioning=`no changes`、冻结 DreamLite 四步完整链路；
- 唯一可训练对象是一个 condition-specific FP32 `xT`；Reader 必须 0 forward；
- loss 是学生 endpoint 与固定 teacher endpoint 的 FP32 elementwise MSE；
- 每组 256 step：1–128 使用各自 base LR，129–256 按同一 cosine 形状衰减到 0；
- weight decay=0，无 gradient clipping；checkpoint=0/64/128/192/256；
- 主结果只看 raw step 256，禁止挑选 best checkpoint。

## 技术 preflight

共 5 次完整 forward、2 次 backward、0 optimizer step：1 次 teacher replay，随后每个
condition 各 1 次 M0 forward 与 1 次 gradient probe。除模型冻结、target 重放、M0
`>=1e-8`、梯度有限且非零比例 `>=0.99` 外，两组还必须满足：

- 初始 student `xT`、初始 endpoint 及 M0 逐位一致；
- gradient FP32 tensor SHA-256、loss、gradient norm、非零比例逐位一致。

任一条件不满足只记技术失败，不解释训练结果。

## Formal 计数与门槛

两个 condition 合计 523 次完整 forward、512 次 backward、512 次 optimizer step、
0 次 Reader forward。每组 raw step 256 必须同时满足：

- endpoint MSE `<=0.001`；
- MSE/M0 `<=0.01`；
- L2/initial L2 `<=0.1`。

## 预注册解释

- 两组都通过：确认存在稳定低 LR 区域；选择 `0.005` 进入新的 source-init alpha=0
  可达控制；
- 仅 `lr-005` 通过：`0.005` 足够，`0.001` 在 256 步内欠拟合；选择 `0.005`；
- 仅 `lr-001` 通过：需要更保守尺度；选择 `0.001`；
- 两组都失败：单纯缩小 LR 不足，下一轮优先判别 Adam 符号归一化、有限差分
  Jacobian、数值精度或 residual 参数化，而不是直接训练 generator。

本轮仍是几何/优化诊断：`formal_success=false`、`phase2_allowed=false`。不得声称
事件到状态学习、共享 writer、Reader 读取、ID/OOD 或 Picture Memory 科学成功。
