# DreamLite 官方训练对齐审计与实验记录

分支：`codex/dreamlite-official-alignment-20260913`，起点 `f68bf06`。附件为问题线索，以下以实际源码和可追溯产物为依据。

当前结论（09-14 15:55）：官方FM公式、完整时间域、纯噪声初态、原生scheduler已成套实现并验证，真实实验使用官方Base原生28步推理。03原生条件模型完成4832步，开发1510/1510、功能1736/1800，连续链480/480但历史改写仍失败64条。b9历史表达增强完成：开发1510/1510、原功能1780/1800、另一套表达1740/1800，PNG读取3520/3600。4f从03继续训练也已完成：开发1510/1510、原功能1790/1800、同一套已观察表达1750/1800；全量PNG读回3540/3600。上述各轮完整原始证据和PNG均已本地复核；当前仍60条连续清除及随后保持错误，尚不支持完整可用。

针对b9新增的连续清除退化，完整32任务/320读数的来源图与权重交叉诊断中，03权重配两种来源均80/80，b9权重配两种来源均60/80，两个对角组合逐项复现原结果。它支持在这些观察案例中排查参数更新造成的退化，不能据此推断所有来源分布都无影响。4f从03已训练参数开始，以fresh AdamW1e-5保留完整历史表达增强和4832额外更新；完整302图、轨迹和raw的初始化复现已实际通过。两套全量逐格比较均相对b9修复10条、没有新增错误，但原注册链仍相对03退步10条。初始化和学习率共同改变，不能作单因素归因。全部训练draw分析确认清除样本与sigma高半区实际被覆盖，FM损失降低也未保证功能全对。

09-14 15:55：固定03的完整已观察表达对照已结束并全部本地复核，1721/1800（360、440、452、469）。全1990 raw、360生成PNG与真实CLI六写三十读齐全。相对03，4f保留1711、修复39历史、新增10链错误、40链仍错；b9保留1701、修复39、新增20、40仍错。全部负对照保持。两类错误并存，不能把4f50条全归于遗忘。完整配对与独立哈希见[03对照报告](official-alignment-results-20260913/03-observed-wording-baseline-review.md)。训练目标未完成；没有新训练已启动。

详见[4f原注册结果](official-alignment-results-20260913/clear-retention-registered-review.md)、[4f表达回归](official-alignment-results-20260913/clear-retention-observed-wording-review.md)、[全量PNG](official-alignment-results-20260913/clear-retention-png-validation-review.md)、[实际训练损失分析](official-alignment-results-20260913/clear-retention-training-loss-review.md)、[03对照固定计划](official-03-observed-wording-baseline-plan-20260914.md)及[实际运行记录](official-alignment-continuation-20260913.md)。两套已观察表达均属于回归集。以下早期记录按实验阶段保留。

16:38进展：完整24张独立生成源图及120条原始读取均通过，全部分块、张量、原生噪声重放及本地复核已齐全。source仅作条件的生成图变化训练实现已在ef163b2准备并部署，额外4832更新，新的4H200实例已创建、仍在资源排队；尚未开始新优化。见[源图完整证据](official-alignment-results-20260913/generated-source-pool-review.md)及[下一轮固定训练合同](official-generated-source-training-plan-20260914.md)。当前4f功能得分和未达可用的结论保持原样。

## 官方依据

2026-09-13 读取官方仓库 HEAD，仍为锁定版本 `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。

- [LoRA 编辑训练源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/lora/train_edit_lora.py)
- [Mobile 推理源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/dreamlite/pipelines/dreamlite/pipeline_dreamlite_mobile.py)
- [LoRA 条件编码源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/dreamlite/pipelines/dreamlite/pipeline_dreamlite_lora.py)

## 第一性原理

令目标为 y，源图为 s，独立高斯噪声为 e。官方 FM 定义 x_sigma=(1-sigma)y+sigma*e，速度为 e-y。网络输入为宽度方向拼接 [x_sigma,s]，再以事件条件和时间进行预测，损失只计算目标半边。sigma 从 1 积分到 0，故 dx/dsigma=e-y 且初态为纯噪声。固定 y/e/sigma 时改变 s，不应改变目标侧状态与速度标签。

历史桥为 x_sigma=(1-2sigma)y+sigma*s+sigma*e，速度 s+e-2y，sigma 最大 0.5。它在数学上是可定义的另一条条件流；错误在于将改变起始分布、时间域、速度目标和少步轨迹后的任务视为官方预训练微调。不能仅改一个公式，也不能据公式直接断言其是全部失败的唯一原因。

条件一致性是另一个独立问题。令 c=C(s,event)，训练实际拟合 v_theta(x_sigma,t,c) 到 e-y；如果训练用 C_raw 而生成用 C_native，即便FM状态、标签和积分器完全正确，也是在另一组条件输入上调用同一网络。官方LoRA示例的raw指令与原生Base的diptych三分支编码确实不同，不能仅因它们都来自官方代码就推断在本任务上可互换。修复条件分布不需要把source混入加噪公式，也不需要缩短训练时间域。

09-14的固定bb34092模型三臂对照覆盖全部302格：真实native首步、相同sigma1的raw条件首步、sigma0.999/整数999的真实训练样本。全部302个native首步逐位重现原轨迹，全部速度PT和906个MSE经CPU复核。10个开发失败格的native平均速度MSE为0.16410525，raw为0.000681887；这支持检验条件差异，不能替代完整28步生成与Reader测试，也不能把差异只归因于文本模板而忽略批处理和padding。新训练取官方完整三分支编码的第三行及mask，推理原样保留；属于明确的训练配方变化，详见[固定计划](official-native-condition-training-plan-20260914.md)。

## 成套修复与早期默认设置

| 项目 | 历史实验 | 新默认 |
|---|---|---|
| FM 状态与标签 | 源图参与桥接 | 目标与独立噪声；source 只作条件 |
| 训练 sigma | 0–0.5 | torch.rand 覆盖 0–1；不施加推理 shift |
| 训练 timestep | 连续 1000*sigma | 官方 floor(1000*sigma) |
| 生成初态 | 0.5*source+0.5*noise | 纯噪声，并逐位校验实际初态 |
| 四步输入 sigma | 强制有效 0.5/.375/.25/.125 | 原始 1/.75/.5/.25 交给 scheduler |
| 有效 sigma | 反解 shift 以强制旧表 | 读取实际 scheduler.sigmas 并记录；不能把 raw 当 effective |
| 提示词 | Mobile diptych 包装 | 训练用官方 LoRA raw event；Base 主评测保留官方 native 包装，另设一致条件对照 |
| 条件图像 | VAE 解码回原图 | 原始灰色 PIL 条件；bank 的 FP32 source latent 保持独立核验 |
| LoRA | rank4 | rank16、alpha16；相同 attention projections |
| 更新 | lr1e-4，weight decay0，无累积 | lr5e-5，weight decay1e-4，累积4，clip1 |
| 身份与恢复 | v1 | v2 绑定协议、条件哈希、raw/effective 时间表、完整 optimizer/RNG |

历史源码通过 Git 保留；共享诊断 helper 的旧调用仍复现 anchored 路径。新 CLI 默认为 `--flow-protocol official`，旧比较臂必须明确选择 `legacy_anchored`。不同协议不得混用旧 checkpoint。

### 当前实训路径逐项核对（09-14）

当前4fbc857沿用03/b9的显式full_unet/native_base/FP32/CFG1配置，所以上表中的LoRA和raw event是早期默认设置，不是当前训练范围与条件。4fbc857另明确更改初始化与lr1e-5。核对实际调用链：

- `flow_microbatch()`在official分支只调用`official_flow_bridge(noise, target, sigma)`。该函数没有source参数；返回`(1-sigma)*target+sigma*noise`和`noise-target`，不会经旧anchored桥。
- `balanced_draw()`在official分支取完整`torch.rand`区间；`predict_velocity(..., integer_timestep=True)`要求1000单位并取整。source仅在宽度维拼接给U-Net，损失裁回目标半边后使用FP32均方误差。这与官方训练源码的相应操作一致。
- Base评估的`NativeBaseEditSampler`把实际初态固定为独立Gaussian，调用锁定官方pipeline完成全部28步；同时核对官方source编码、实际29个状态、有效sigma从1至0。推理时间没有错误地套用训练的整数化，也没有把Mobile的原始四步表当作Base有效时间表。
- 当前训练条件取官方Base三prompt完整编码批次的第三行及mask，保持真实padding。历史9表达仅改变此条件输入，不改变source/teacher/noise/sigma流或31逻辑条件的权重；此项是已公开的训练配方变化，不宣称等同官方raw LoRA示例。
- 官方示例调用`unet.train()`，本实验固定`unet.eval()`并仍对明确选中的参数反向传播。09-14直接检查当前固定Base快照的实际`unet/config.json`，其`dropout`为0.0；官方自定义模型源码中未发现BatchNorm或显式`self.training`条件分支。此配置观察用于解释模式差异，不能代替数值等价证明，也不据此改变正在运行的训练。

单次与历史前缀的已有评分读FP32解码像素，实际部署读量化PNG；因此另外固定[3980条完整PNG读取验收](official-png-readback-acceptance-20260914.md)。源码协议对齐、开发集通过和CLI重放一致都不能代替该完整功能证据。

## 必须公开的剩余区别

1. 官方 LoRA 示例训练 DreamLite-base，历史主路径加载Mobile（蒸馏后的四步模型）。最早Mobile适配不能称为官方Base复现；后续已建立固定Base快照的独立训练与原生28步对照，目前候选使用Base。
2. 官方 LoRA 条件编码用512×512图像，Mobile用256×256；不得混淆两种模型。此前Base训练用raw event、推理保留官方diptych包装。两种条件的CFG1对照仅在早期三状态集合通过，不能据此推断扩展任务条件等价。全302格首步结果已推翻这种广泛推断；当前明确用native_base训练条件做同预算对照，官方raw训练选项仍保留。条件编码的批处理/padding也属于该变量。
3. 官方从训练 RGB 编码 target，本实验使用已验证的 FP32 model-space oracle endpoint。重新 decode/encode 会改变这些经搜索验证的目标，故保留并明确这一实验设计。灰图 source 的 FP32 编码及 PIL 量化差异亦不能掩盖。
4. 官方示例 bf16，本实验保留已验证的 FP32 DreamLite/latent 与 bf16 Reader，确保 teacher 重放精度。训练步数由各实验显式记录；短程 pilot 不冒充官方 3500 步预算。
5. 单题成功不能证明按事件写入或泛化；需进一步反事实事件和多题检查。
6. 后续全U-Net容量对照改变了官方示例的LoRA训练范围。这是根据配对失败证据做的显式实验选择，保持官方FM/条件/采样协议，不称为原样复现官方LoRA配方。
7. 官方示例将batch prompts替换为同一个default_prompt，属于固定风格示例；本任务学习多个事件条件是额外实验任务。不能假定官方示例本身已证明多状态记忆更新。
8. 当前候选CFG1是根据固定零更新诊断得到的显式推理选择，默认官方CFG7.5的失败端点保留。早期三状态属于同一个实体、同一道语义问题；当前151条件对应17道已见语义问题。新噪声、事件改写、RGB连续更新和更广范围的功能仍需分别验证，不能将表达数量当作未见问题数量。

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
