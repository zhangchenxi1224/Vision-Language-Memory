# DreamLite 官方训练对齐审计与实验记录

分支：`codex/dreamlite-official-alignment-20260913`，起点 `f68bf06`。附件为问题线索，以下以实际源码和可追溯产物为依据。

## 官方依据

2026-09-13 读取官方仓库 HEAD，仍为锁定版本 `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。

- [LoRA 编辑训练源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/lora/train_edit_lora.py)
- [Mobile 推理源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/dreamlite/pipelines/dreamlite/pipeline_dreamlite_mobile.py)
- [LoRA 条件编码源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/dreamlite/pipelines/dreamlite/pipeline_dreamlite_lora.py)

## 第一性原理

令目标为 y，源图为 s，独立高斯噪声为 e。官方 FM 定义 x_sigma=(1-sigma)y+sigma*e，速度为 e-y。网络输入为宽度方向拼接 [x_sigma,s]，再以事件条件和时间进行预测，损失只计算目标半边。sigma 从 1 积分到 0，故 dx/dsigma=e-y 且初态为纯噪声。固定 y/e/sigma 时改变 s，不应改变目标侧状态与速度标签。

历史桥为 x_sigma=(1-2sigma)y+sigma*s+sigma*e，速度 s+e-2y，sigma 最大 0.5。它在数学上是可定义的另一条条件流；错误在于将改变起始分布、时间域、速度目标和少步轨迹后的任务视为官方预训练微调。不能仅改一个公式，也不能据公式直接断言其是全部失败的唯一原因。

## 本次成套修复

| 项目 | 历史实验 | 新默认 |
|---|---|---|
| FM 状态与标签 | 源图参与桥接 | 目标与独立噪声；source 只作条件 |
| 训练 sigma | 0–0.5 | torch.rand 覆盖 0–1；不施加推理 shift |
| 训练 timestep | 连续 1000*sigma | 官方 floor(1000*sigma) |
| 生成初态 | 0.5*source+0.5*noise | 纯噪声，并逐位校验实际初态 |
| 四步输入 sigma | 强制有效 0.5/.375/.25/.125 | 原始 1/.75/.5/.25 交给 scheduler |
| 有效 sigma | 反解 shift 以强制旧表 | 读取实际 scheduler.sigmas 并记录；不能把 raw 当 effective |
| 提示词 | Mobile diptych 包装 | 官方 LoRA 原始 event；同一条件用于训练和评测 |
| 条件图像 | VAE 解码回原图 | 原始灰色 PIL 条件；bank 的 FP32 source latent 保持独立核验 |
| LoRA | rank4 | rank16、alpha16；相同 attention projections |
| 更新 | lr1e-4，weight decay0，无累积 | lr5e-5，weight decay1e-4，累积4，clip1 |
| 身份与恢复 | v1 | v2 绑定协议、条件哈希、raw/effective 时间表、完整 optimizer/RNG |

历史源码通过 Git 保留；共享诊断 helper 的旧调用仍复现 anchored 路径。新 CLI 默认为 `--flow-protocol official`，旧比较臂必须明确选择 `legacy_anchored`。不同协议不得混用旧 checkpoint。

## 必须公开的剩余区别

1. 官方 LoRA 示例训练 DreamLite-base，本仓库默认加载 Mobile（蒸馏后的四步模型）。本轮先对 Mobile 做官方 FM 目标适配，不能称为官方 base 的完整复现。后续应使用现存 base 快照建立独立对照。
2. 官方 LoRA 条件编码用 512×512 图像，Mobile 用 256×256；不得混淆两种模型。官方训练传 raw prompt，Mobile 公共推理包装 diptych；本实验为保持条件一致，两端采用同一明确配置。
3. 官方从训练 RGB 编码 target，本实验使用已验证的 FP32 model-space oracle endpoint。重新 decode/encode 会改变这些经搜索验证的目标，故保留并明确这一实验设计。灰图 source 的 FP32 编码及 PIL 量化差异亦不能掩盖。
4. 官方示例 bf16，本实验保留已验证的 FP32 DreamLite/latent 与 bf16 Reader，确保 teacher 重放精度。训练步数由各实验显式记录；短程 pilot 不冒充官方 3500 步预算。
5. 单题成功不能证明按事件写入或泛化；需进一步反事实事件和多题检查。

## 以往证据核验

- 新三训练问法 Direct bank 有 96 个已成功终点，属于同一道 ambient 题。不能把它当 96 道题；未对终点求平均。
- 旧 512 步参考轨迹蒸馏新噪声 0/8，带 bank 的参考算法 8/8；参考算法推理访问成功库，不能归功于 U-Net。
- 2026-09-13 从共享盘读取 `runs/unet-learnability/fd07eb1-20260909-r01`：旧 anchored 单目标 rank16×512 和 rank4×2048 都没有通过训练输入及 8 个新噪声的功能门槛。因此增加 rank 或延长旧训练不是已验证的修复。

## 当前验证与下一步

本地公式、采样器、评测与恢复测试第一轮 33 passed。另加直接提取并执行锁定官方训练源码的状态、标签和 loss/gradient 数值对照；GPU 上还要运行真实官方 pipeline 逐步轨迹 parity。

新建 `dl-align-cpu-20260913` 读取共享盘和同步固定提交。首次 H200 请求因原资源组无可用卡而停止，改为 `dl-align-h200x2-20260913-r2`，同镜像 CUDA12.8、2 H200、40 CPU、400 GiB。设置本轮 8 小时平台上限，避免无人监控空耗；需要更多训练时按证据接续。

先以哈希确定的单一训练 teacher 检查官方 FM 微调是否可学，普通训练噪声独立于评测噪声；再决定多目标及 base 对照。保留原问、四个改写、blank/donor、原始 greedy 32-token generation、EOS、几何统计。训练完成不等于功能通过；实验产物到齐前不声称已得到可用版本。

## 后续实测与Base实现

Mobile单目标512步已完成，原问及四种改写全部0/8；原始结果和实际生成图片见 [实验记录](official-alignment-results-20260913/README.md)。45项相关测试通过，严格确定性FP32的实际官方Mobile轨迹逐位一致。

已实现Base独立运行时：直接加载官方 `DreamLitePipelineLoRA`，训练raw event、512像素conditioner、全时间域FM；评测直接调用官方28步CFG代码。因此上文“本仓库默认Mobile”的限制仅适用于Mobile臂，不能用于描述现在新增的Base臂。Base评测遵循官方原样的diptych提示词包装，与其训练示例raw event的差别完整保留并记录。

Base历史下载缺失text_encoder权重，已经按官方HF revision的内容SHA补全；完整27文件封存通过，且Base/Mobile VAE权重字节相同。Base `ac34ab2-base-single-20260913` 已完成512步，五問法均0/8；同checkpoint的CFG1仍均0/8。Oracle邻域诊断表明其有一定可读范围，见实验记录。已预注册并启动相同代码的3500步预算对照，功能目标仍active。新增评测汇总严格EOS和完整问法检查后，相关测试46 passed。
