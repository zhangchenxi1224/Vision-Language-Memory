# R11_new 固定可达目标吸引域判别：预注册

状态：在 parent `alpha=0` 真实失败完成并交付后、任何 warm-start DreamLite forward 前锁定。

## 已知事实与唯一问题

Parent 可达目标按构造存在精确 teacher `xT`，但 source-only `alpha=0` 经 256 步 Adam 后，endpoint MSE 仅从 `0.02440103` 降至 `0.01938320`，MSE/M0=`0.79436`，预注册 gate 失败。全程梯度稠密、无裁剪、256 次 optimizer step 完整，因此下一步不增加数据或 LoRA，只回答：**同一优化器在靠近已知精确解时能否收敛？**

## 唯一自变量

固定复用 parent target artifact：

- 文件 SHA-256：`5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；
- source `xT`：`719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`；
- teacher `xT`：`9a5cde0f0f93ba29dc74b2b78878e64dde49a6ec9168cb60f57ad48cdfb31f66`；
- teacher endpoint：`9e627ec840af943d29e3b40a9a8ab796ce23bf07861cb18801360157c4bea0e9`；
- stored noise：`a320afe2438b0ee725de4702b7d7bd7a213c08ad306ea05eef699b292f7b4377`。

禁止重新采样、重新归一化或生成 target。只运行两个固定初始化，顺序不可改：

1. `alpha-050 = source + 0.50 × (teacher-source)`；
2. `alpha-099 = source + 0.99 × (teacher-source)`。

## 不变训练口径

- 同一 target 1、灰色 source、conditioning=`no changes`、冻结 DreamLite 四步完整链路；
- 每个 condition 独立新建 Adam，只有该 condition 的学生 `xT` 可训练；
- 每个 condition 256 步，LR 1–128 为 `0.05`，129–256 cosine 衰减至 0；weight decay=0；无 gradient clipping；
- loss 仍为学生 endpoint 与固定 teacher endpoint 的 FP32 elementwise MSE；
- checkpoint 固定为 0/64/128/192/256；主结果只看 raw step 256，禁止挑 best checkpoint；
- Reader 0 forward；模型权重必须前后哈希一致；两个 optimizer 间状态必须重置。

## 技术 preflight

1 次 teacher endpoint replay，加上每个 condition 1 次 M0 forward 与 1 次 gradient probe，总计 5 次完整 forward、2 次 backward、0 optimizer。每个 condition 必须满足：M0 `≥1e-8`、梯度有限且范数大于 0、非零元素比例 `≥0.99`。任一失败仅按技术失败处理，不读取 partial outcome。

## Formal 计数与门槛

两个 condition 总计 523 次完整 forward、512 次 backward、512 次 optimizer step、0 次 Reader forward。每个 condition 独立同时满足：

- raw256 endpoint MSE `≤0.001`；
- raw256 MSE/M0 `≤0.01`；
- raw256 L2/initial L2 `≤0.1`。

门槛与 parent 完全相同，不因 M0 大小或结果修改。

## 预注册解释

- `alpha-099` 失败：即使只距精确解 1%，当前优化仍不能收敛；优先诊断局部 Jacobian、精度或 `xT` 参数化，继续暂缓 generator adaptation。
- `alpha-099` 通过、`alpha-050` 失败：存在窄吸引域；下一步测试 continuation/重参数化，不直接扩数据。
- 两者都通过：局部及中距离优化正常，source-only 初始化处于可用盆地之外；下一步预注册 continuation 或最小 generator adaptation。
- `alpha-050` 通过但 `alpha-099` 失败：非单调异常，停止科学解释并审计实现/数值。

无论哪种结果，本轮均为几何诊断：`formal_success=false`、`phase2_allowed=false`，不得声称事件到状态学习、共享 writer、canonical teacher 生成或 ID/OOD 成功。
