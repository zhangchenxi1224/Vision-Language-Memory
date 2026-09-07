# R11_new Oracle terminal-capture：结果交付

## 结论先行

本轮技术执行与独立审计全部通过，但仍是固定 target 数值诊断，不是 Picture Memory
成功。预注册分类为：`microscopic_partial_code_capture`。

最关键结果非常明确：

- 从精确 teacher 沿 plateau 方向离开，直到请求 `delta=3e-5`，endpoint 仍与 teacher
  **逐位完全相同**、loss=0；此时 65,536 个 BF16 坐标中已有 3 个不同；
- 到 `delta=1e-4`，约 17 个 BF16 坐标不同，endpoint loss 突然跳到 plateau loss 的
  `1.526692×`；
- 因而强捕获连续前缀的最大宽度仅为 `3e-5` L2，小于预注册的 `1e-4` 下界；
- 两个相反扫描顺序的 17/17 endpoint 全部逐位一致，排除了顺序、缓存或状态污染；
- `formal_success=false`，`phase2_allowed=false`。

这把上一轮“2.0 到精确 2.051211 之间发生什么”解析清楚了：不是平滑下降，而是一个
极窄的 BF16 离散捕获单元。当前 FP32 参数经 BF16 cast 后的 surrogate gradient，很难从
距离 `O(1)` 的初始化稳定命中宽度 `O(1e-5)` 的目标单元。

## 实验口径与计数

固定复用同一 target、同一 `lr=.001 raw256` plateau 和同一冻结 DreamLite。17 个候选
`xT` 的请求剩余 L2 为：

`[0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 2.0512112034]`。

网格在运行前只根据 parent tensor 的 BF16 code 边界选择，每个候选 FP32 tensor/hash
均预先锁定。formal 先按 teacher→plateau，再按 plateau→teacher 重放：34 scan forward；
加独立 teacher replay，共 35 forward、0 backward、0 optimizer step、0 Reader。

| 请求 delta | 不同 BF16 坐标数 | endpoint/teacher 逐位相同 | loss / plateau loss |
|---:|---:|---|---:|
| 0 | 0 | 是 | 0 |
| 1e-8 | 0 | 是 | 0 |
| 3e-8 | 1 | 是 | 0 |
| 1e-7 至 1e-5 | 2 | 是 | 0 |
| 3e-5 | 3 | 是 | 0 |
| 1e-4 | 约 17 | 否 | 1.526692 |
| 3e-4 | 约 61 | 否 | 1.565838 |
| 1e-3 | 约 191 | 否 | 1.602286 |
| 3e-3 | 约 503 | 否 | 1.441883 |
| 1e-2 | 约 1,358 | 否 | 1.468686 |
| 5e-2 | 约 5,107 | 否 | 1.597844 |
| plateau 2.051211 | 约 49,792 | 否 | 1.0 |

“不同坐标数”由相同率乘以 65,536 后取整，仅用于解释；原始相同率、BF16 L2、FP32 L2、
endpoint tensor 与哈希均保存在 metrics/归档中。

## 第一性原理归因

冻结映射实际接收的是 `BF16(xT)`。因此 FP32 参数空间中的大多数微小连续移动，要么不
改变 BF16 code、endpoint 完全不动；要么跨过少数离散边界，造成不成比例的 endpoint
跃迁。本轮直接观察到：

1. `delta<=3e-5` 的多个不同 FP32 tensor 映射到相同 endpoint；loss 平台完全平坦；
2. 从 3 个 changed code 增加到约 17 个时，loss 一步从 0 跳到 `1.5267×`；
3. 离开终端 cell 后，loss 在 `1.44×–1.60×` 间非单调波动，并不提供通向 teacher 的
   连续斜坡；
4. 这与上一轮“autograd 解析方向和真实有限差分失配”相互印证，而不是多训练 step 能
   自动解决的问题。

所以当前主根因已从“Adam 学习率不合适”推进到更基础的接口问题：**优化变量是 FP32，
但冻结 DreamLite 的第一道有效接口是 BF16 量化；反向给出的连续 surrogate 与真实离散
前向几何不一致。**

## 下一步主线

不再继续扫 Adam 或盲目加 step。下一轮做同一 target/path 的精度-参数化判别：

- BF16 条件保持当前真实基线；
- 将相同 BF16 权重值与固定 conditioning 提升到 FP32 activation compute，移除 `xT`
  首层 BF16 cast，但不引入新模型知识；
- 对每种精度分别重放自己的 teacher endpoint，比较 terminal path 是否从离散 cliff
  变成宽且近似连续的 capture basin，并检测负梯度的真实有限差分；
- 若 FP32 显著恢复连续可下降几何，下一轮采用 FP32/residual-centered trust-region；
  若仍无改善，则根因不只在输入 cast，要重新审视 DreamLite latent 参数化，而不是优化器。

该对照仍是 oracle 固定 target 诊断。只有可达性真正解决后，才恢复事件→状态 writer、
Reader 因果读取、固定数据、多 seed、SET/overwrite 与 ID/OOD 最终门槛。

## 交付与复现

- 实现 commit：`0262545a91788f59c96a420bc27528cbbed6d0c0`；
- 原始归档：`raw/r11-new-terminal-0262545-20260907-round01.tar.gz`；
- archive SHA-256：
  `9c94ff86eb2649c049a763e93660fa6393e53117ad88e0d753ec588efbc75892`；
- preflight inventory：22 项，SHA-256
  `ed1879beac9a947c2e6a0ada2f15929a2b2107cccc9985d8d2f117f2e8853e14`；
- formal inventory：56 项，SHA-256
  `376958a52709d637ee4daeb1170c8b75214fb7d26b16826b1ebb28bc2c79bdb7`；
- formal result SHA-256：
  `0ccc5a80967327643d45bfc0a6a498e37544a1d03bb4972891afbcf0fbd8a814`；
- `render_results.py` 会验证 archive、两个 inventory、78 个清单内 artifact，并在临时
  目录运行独立审计器后生成 `derived/summary.json` 与三张图；
- 原始包包含全部 34 个 endpoint tensor、34 张图片、target/checkpoint、metrics、日志、
  环境、模型快照、config、manifest 和哈希清单；精选目录不替代原始归档。

所有生成图片仍是人类不可读的近均匀灰色纹理；视觉外观没有被当作成功证据。
