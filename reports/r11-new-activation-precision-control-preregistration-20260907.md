# R11_new 激活精度对照：预注册

状态：上一轮 oracle terminal-capture 结果完成并交付后、任何本轮 precision-control
DreamLite forward 之前锁定。

## 第一性原理问题

上一轮已证明 teacher `x_T` 确实存在，但 BF16 frozen DreamLite 映射只在
`remaining L2 <= 3e-5` 时保留相同 endpoint；到 `1e-4` 时 loss ratio 立即跳到
`1.52669`。同时 plateau 负梯度沿有限差分多数不下降。这留下两个不同根因：

1. FP32 `x_T` 在 sampler 边界被量化为 BF16，局部可解单元与 autograd surrogate
   因此失真；
2. 即使消除该量化，四步 DreamLite 映射自身仍具有不连续/病态 Jacobian 几何。

本轮只区分这两个根因，不训练、不调参、不使用 Reader，也不宣称 Picture Memory
成功。

## 唯一变量

- `bf16-baseline`：原样重放已交付的 frozen BF16 数值映射。
- `fp32-lifted`：完成 BF16 条件后，将同一内存中已加载的 BF16-valued UNet、VAE、
  source latents 与固定 prompt embeddings 无损提升为 FP32 算术；不重新加载模型、
  不重新编码条件、不恢复不存在的原始 FP32 权重。

两条件固定相同数据、target、plateau `x_T`、teacher `x_T`、scheduler、四步轨迹与
模型文件。FP32 条件改变的是数值映射，所以必须生成自己的 teacher endpoint，并仅以
自己的 plateau MSE 归一化。

## 固定测量

- 路径点取上一轮索引 `(0, 8, 9, 11, 13, 15, 16)`，对应 remaining L2
  `(0, 3e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2.0512112034)`。
- 每个条件按 teacher→plateau、plateau→teacher 各重放一次，共 14 个路径 endpoint；
  重复点必须 bitwise 相同。
- 每个条件在同一 plateau 计算一次 endpoint-MSE 的 FP32 `x_T` 梯度；沿单位负梯度
  在半径 `(1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1)` 上评测正负两侧，共 12 个 endpoint。
- formal 总计 54 full-chain forward、2 backward、0 optimizer step、0 Reader forward。
- technical preflight 总计 4 forward、2 backward、0 optimizer/Reader。

## 预注册判据

- FP32 恢复 capture：连续强捕获前缀达到 `remaining L2 >= 1e-3`，且宽度至少为
  BF16 的 10 倍。
- FP32 梯度可操作：负梯度正向扫描的最佳 loss ratio `<= 0.9`。
- BF16 路径必须复现上一轮 7 个 endpoint hash 与 loss ratio；否则停止解释 FP32。

分类固定为：

- `fp32_restores_capture_and_gradient`
- `fp32_widens_capture_without_actionable_gradient`
- `fp32_gradient_only`
- `precision_lift_insufficient`
- `bf16_reproduction_failure`

无论分类为何，本轮 `formal_success=false`、`phase2_allowed=false`。只有第一类结果才允许
下一轮预注册 FP32 residual-centered trust-region 优化；其余结果按配置中的固定决策表
继续最小纠因实验。

## 完整性边界

配置、代码、测试、Git commit、parent raw archive、模型 snapshots、数据、环境变量、
所有输入/endpoint/gradient/direction tensor hash、图片、日志、计数器和 artifact inventory
均须落盘并独立重算。任何缺失、非有限值、计数漂移、Reader 调用、optimizer step、
snapshot 改写或后验修改半径/阈值均按技术失败处理。
