# R11_new 激活精度对照：结果与结论

## 一句话结论

在数据、模型文件、BF16-valued 权重数值、conditioning、teacher/plateau `x_T`、四步
DreamLite 路径全部固定时，仅把 UNet/VAE 与激活算术从 BF16 无损提升到 FP32，就把
强 oracle capture 宽度从 `3e-5` 扩至 `0.05`（`1666.67x`），并使 plateau 负梯度
达到 loss ratio `0.81522` 且 6/6 个有限差分尺度方向一致。因此，当前训练失败的主要
近因确实是 BF16 输入/激活量化造成的离散 cliff 与 surrogate-gradient 失真，而不是
“DreamLite 全链路绝对不可导”。

这仍是 oracle fixed-target 数值诊断，`formal_success=false`、`phase2_allowed=false`；
它没有证明事件→记忆学习、Reader、ID/OOD 或共享 writer 成功。

## 固定实验

- Git commit：`27530943dba93d457e0ba0560201cd58dd81ca4b`。
- 条件 A：原始 `bf16-baseline`。
- 条件 B：同一次加载中将 1209 个、共 392,413,451 个 BF16 UNet/VAE 数值提升到
  FP32；不 reload、不重新生成 conditioning，text encoder 不执行。
- 每条件：14 个双向 path endpoint + 1 个 plateau-gradient forward + 12 个双符号
  gradient-scan endpoint。
- formal：54 forward、2 backward、0 optimizer step、0 Reader forward；58.57 s。
- preflight：4 forward、2 backward、0 optimizer/Reader；34.62 s。
- BF16 的 7 个路径 endpoint hash/loss ratio 逐项复现上一轮；两遍路径全部 bitwise
  相同，模型 snapshot 文件前后不变。

## 核心结果

| 指标 | BF16 baseline | FP32 lifted |
|---|---:|---:|
| plateau endpoint MSE | `1.63641e-5` | `1.79774e-5` |
| 强 capture 连续前缀宽度 | `3e-5` | `0.05` |
| 相对 BF16 capture 宽度 | `1x` | `1666.67x` |
| 负梯度最佳正向 loss ratio | `0.95938 @ r=.001` | `0.81522 @ r=.1` |
| 有限差分下降符号数 | `1/6` | `6/6` |
| 梯度非零比例 | `0.999603` | `1.0` |

FP32 路径在 `remaining L2=.001/.01/.05` 时的 loss ratio 分别仅为
`2.76e-7 / 2.59e-5 / 6.46e-4`；BF16 在 `.0001` 已跳到 `1.52669`。这不是小幅
数值改善，而是从“离散不可搜索”切换到“宽且局部可优化”的连续映射。

## 图与图片观察

- `figures/activation_precision_path.png`：相同 `x_T` 路径下 BF16 cliff 与 FP32
  连续低损失区的直接对照。
- `figures/activation_precision_gradient.png`：BF16 与 FP32 双符号有限差分；只有 FP32
  负梯度在全部六个尺度上符号一致，并跨过 `<=0.9` 门槛。
- `figures/activation_precision_montage.png`：两种精度的 teacher、中间点和 plateau
  图片。它们仍近似均匀、肉眼不可读，符合“像素承载模型可读 latent code”方向；但本轮
  未调用 Reader，所以不能由外观推断记忆已可读。

## 科学解释与下一步

FP32 结果支持如下因果链：

`FP32 x_T → BF16 sampler cast/activations → 极窄离散 code cell → 有解析梯度但有限差分失真 → Adam 无法稳定进入 teacher basin`。

按运行前决策表，下一轮只允许预注册一个最小 FP32 residual-centered normalized-gradient
trust-region 优化：仍使用同一 target/plateau 和 endpoint-MSE，先验证从 plateau 能否
真实、单调、可重复进入 teacher capture；不得直接外推为全量 Picture Memory 成功。
只有该优化在固定 checkpoint、重复运行和独立 replay 下通过，才进入非 oracle writer
与 Reader 因果评测。

## 完整交付

- 原始归档：`raw/r11-new-precision-2753094-20260907-round01.tar.gz`，40,511,899 bytes，
  SHA-256 `b4a34fa3e9d3d7f40949b4b6534978e00e552f1baaf2903554dc5ee112def659`。
- formal inventory：79 项，SHA-256
  `7be7b1d219d3f91c5c85ef8428013d94692a185720ecd365cc6f67ea03de5673`。
- preflight inventory：25 项，SHA-256
  `1eee2d0fb87cb6cd29660b34b388563b02ab9a2a9c51f6e295c6c71cc98c904f`。
- `derived/summary.json` 保存本地独立审计、精度提升证据、完整指标与图片统计；
  `DELIVERY_MANIFEST.json` 对本目录所有交付文件逐项记录大小和 SHA-256。

原始归档中的 endpoint tensors、gradient/direction tensors、全部 52 张扫描图片、配置、
环境、日志、父产物副本、snapshot 证明和 inventory 均已保留；仓库中另展开关键元数据、
指标、图和代表图片以便审阅。
