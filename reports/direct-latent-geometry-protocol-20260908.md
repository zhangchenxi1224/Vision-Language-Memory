# Direct final-latent 多分布主实验：前瞻协议

研究问题：对于同一道填空题，从不同分布与大小的初始 final latent 出发，哪些轨迹能得到 Reader 严格读对的终点？成功终点是否具有可重复的几何结构？

训练路径：`z_init → 可训练 z → 冻结 FP32 VAE → RGB → 冻结 BF16 Qwen → answer CE + EOS CE`。只更新一个 `1×4×128×128` FP32 参数张量，Adam LR 0.05，256 次更新。U-Net、condition encoder 和 scheduler 不加载、不执行。EOS 使用模型实际 assistant 模板的结束 token，并核对 generation 的停止 token。答案为 ambient；第一批仍是同一道题的机制实验，不是多题训练结果。

## 新起点及固定条件

所有 96 条训练均从重新生成的初值开始，不复用旧训练终点作为初始化。统一中心为常量 RGB 127/255 经冻结 FP32 VAE posterior mean 编码的 reference。公式：

`z_init = reference + 0.1 × scale × RMS(reference) × epsilon`

| 研究 | 设置 | 实际新增训练 |
|---|---|---:|
| 分布 | Gaussian、Uniform、Sphere、Rademacher、heavy-tail，各 seed 0–7，scale=1 | 40 |
| 大小 | Gaussian，scale=0.25、0.5、1、2、4，各 seed 0–7，复用 scale=1 单元 | 32 |
| Gaussian 更密采样 | Gaussian scale=1，补充 seed 8–31 | 24 |

Uniform 为 U(-√3,√3)，Rademacher 为等概率 ±1，heavy-tail 为 Student-t(df=5)×√(3/5)，Sphere 为高斯方向归一到固定 RMS=1。前四种非 Sphere 分布只匹配总体方差，不逐次归一，保留真实抽样半径；Sphere 固定半径。相同 Gaussian seed 的 scale 对照复用方向。随机生成器为 NumPy PCG64。记录每个起点的均值、RMS、峰度、原始张量及 SHA。

这里研究的是共享 reference 周围、不同分布形态和五档大小的起点；不能称为整个 VAE 空间的穷尽搜索。FP32 VAE 为本轮前瞻数值设置，不能与旧 BF16 R11 结果作为只有一项变化的因果对照。

## 填空损失与换问法工程修正

`L = mean(answer token CE) + EOS CE`，EOS 权重固定为 1；保持完整答案序列。训练只用 original 问句，以隔离起点变量；4 个改写仅用于新训练轨迹的终点评测，不参与选种子或训练。

每个 prompt 严格三行，第一行保留相同模板前缀、实体、属性和时间语义，仅改问句；后两行逐字固定：

```text
[提问句]
Use the memory image to answer.
Answer with a short phrase only.
```

主指标为固定 step256、原问句、matched 图片、raw greedy generation（32 tokens）的 Exact Match。另报答案前缀正确率、过度续写率、五问句全部正确率，不能把前缀命中当作完整回答正确。blank 与固定异答案 donor 均评估五问句。donor 只作图像依赖对照，不作为新训练初值或成功目标。此次不新增旧终点 replay。

每条轨迹保存全部 257 个 latent；在 0/1/2/4/8/16/32/64/128/192/256 保存 optimizer/RNG、PNG 和原问句 raw generation。每条末尾另保存 15 条原始 generation（五问句×matched/blank/donor）。梯度重复性必须 bitwise 一致；VAE/Reader 始终冻结，源代码、数据、模型快照绑定不可变。

## 算力、断点和后续工作

部署到用户现有 `dl-base-h200x4-20260907`，GPU 0/1 和 2/3 各运行一条独立轨迹。启动前确认四卡型号、显存及无既有计算进程。到实例截止时间前，按已观测耗时预留至少 15 分钟，在完整轨迹边界暂停。已完成且校验通过的轨迹续跑时跳过；中途被强停的残缺目录保留并报错，必须审计后单独恢复或重跑，绝不静默覆盖。

全部结束后输出按分布/scale/seed 的成功率、终点与初值矩阵、成功终点及位移的 PCA 谱，并给出相同样本数的各向同性参照。PCA 的秩受样本数限制，不能仅凭低秩图宣布存在低维流形；本轮还不足以判断跨题类别。

这些结果构成候选监督库。后续分别建立 Direct 成功终点库和 Frozen DreamLite 可达成功终点库，验证标签与稳定性，分析分布后，再做 U-Net 监督对照。不要直接平均多个成功 latent：平均点可能失去可读性。共享 U-Net 最终还须在多题及留出数据上验证，单题 oracle 成功不等于 Writer 成功。此脚本不自动启动 U-Net 训练。

当前另一个 `vlm-oracle-geometry-h200x4-20260908-r02` Job 的历史协议是 MCQ、仅优化 xT。其数据保留独立标签；它尚未切换到本次 EOS 填空协议，也不与新 Direct EOS 成功率作严格配对比较。
