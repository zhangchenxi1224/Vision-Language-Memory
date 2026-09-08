# Direct 成功而 U-Net 失败：因果排查

日期：2026-09-09，北京时间。范围仅限第一轮 Direct 单题 `ambient` 的成功 bank 与后续两组 U-Net，不能外推为 Frozen 路线结论或多题泛化结论。

## 问题与第一性原理

Direct 的优化变量是 final latent；损失是冻结 Reader 对正确答案与结束 token 的负对数概率。成功说明存在 `Reader(VAE(z)) = ambient` 的 latent。

后续 U-Net 更新的是 rank-4 LoRA，以成功 latent 集合为目标做 source-anchored conditional flow matching。其损失约束的是插值状态上的生成速度，不直接约束完整四步生成图片的问答结果。成功 latent 的存在，不保证有限参数、有限训练预算和四步推理能够生成可读 latent。

第一轮 bank 有 96 个同题正确终点，其中 77 个参与训练、19 个留出作目标坐标诊断；不是 96 道题。两组 U-Net 各更新 512 步，原问在 8 个新噪声上均为 0/8，五问法均为 0/40，原问输出均为 `No music preference recorded.`。

## 已完成的零更新对照

代码：`9b2b1a2451aea855f8d446a4f3a05c2ded0b504d`。

原始证据：[controls.json](controls.json)。本次只加载第一组旧 U-Net checkpoint，未更新任何参数。

| 检查 | 真实结果 | 能说明什么 |
|---|---|---|
| 抽查 3 个成功 teacher，重新 VAE 解码并问原题 | 3/3 输出 `ambient`；第一个解码图与 Direct 原图逐位一致 | teacher 读取、latent 编解码尺度及 Reader 链路可工作 |
| 已训练 LoRA 加载 | 所有可训练参数与 checkpoint 逐位一致 | 不是漏加载已训练权重 |
| 重新执行原四步 sampler | 输出 latent 与旧评测保存值逐位一致 | 旧失败可复现，不是读错评测产物 |
| 训练用 velocity 前向手工四步积分 vs 正式 sampler | 最大误差 2.46e-6 | 已检查的拼接、时间单位、积分方向和裁切一致 |
| 用已知正确 teacher 构造精确速度，四步积分 | 终点误差 2.38e-7，输出 `ambient` | 积分符号与比例正对照通过；这不是模型学会的结果 |
| 77 个训练 teacher 取平均 | 输出 `No music preference` | 正确目标平均后不保证正确 |
| 抽查两个正确 teacher 的中点 | 输出 `indigo desk train 001123 prefers jazz.` | 该对 teacher 之间存在不正确的插值点 |
| 同一模型、同一新噪声：4 / 16 / 64 步推理 | 均输出 `No music preference recorded.` | 此样本不能仅靠增加积分步数修好 |
| 错误生成结果向正确 teacher 插值 | teacher 权重 0.25 仍错，0.5 / 0.75 / 0.9 均答对 | 该方向存在可恢复的正确区域，不支持“只有一个精确点才正确”的断言 |

不同起点的 teacher 平均值不正确，不等于已经证明 U-Net 发生均值塌缩。本次模型输出与平均值有明显距离；不能把这两个命题混为一谈。条件 flow matching 的 MSE 是状态相关的向量场回归，理论上能表示多模态分布，并不必然把所有终点平均成一个点。参见原始论文 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 固定样本的损失复测

同一个噪声、相同 4 个训练 teacher，比较训练前后的速度 MSE；另 4 个留出 teacher 结果方向一致。此处使用固定输入，避免将原日志里不同 sigma/不同目标的首尾 loss 当作可比曲线。

| sigma | 训练前 MSE | 512 步后 MSE |
|---|---:|---:|
| 0.500 | 0.4040 | 0.4046 |
| 0.375 | 0.7308 | 0.6545 |
| 0.250 | 1.5824 | 0.9871 |
| 0.125 | 2.4845 | 1.2906 |
| 0.050 | 2.7215 | 1.2408 |

模型确实学到一部分插值状态上的速度，但仍有较大残差，自行生成的轨迹未进入能被 Reader 正确读取的区域。这把问题定位到成功 bank 向 Writer 行为的转移；尚不能仅凭这些对照区分训练预算、LoRA 容量与目标设计各自贡献。

## 有限预算干预

独立脚本 `scripts/analysis/probe_unet_objective.py`，修正后的锁定代码 `f0ed1c4a475104c99c001e0ad3e1b9ae54c86e8a`。

- 两组从同一份旧 512 步 LoRA 开始，均重置为相同的新 AdamW 状态，LR=1e-4、rank=4、裁剪 1，各追加 64 个 optimizer updates。
- A 继续原 flow matching；B 通过真实四步 DreamLite → VAE → Reader，使用答案 + EOS 损失，只更新 U-Net LoRA。
- 同一对 H200 顺序执行；相同新训练噪声、相同 8 个留出评测噪声；训练只用原问，最后评测原问与四个未训练改写，固定两行指令、greedy/32 tokens。
- 两组优化步数相同，计算量不相同：B 每步包含完整生成、VAE 和 Reader 反传。结果不能解释为等 FLOPs 优劣比较。
- 原始 run `r01` 在评测记录缺少 question_id 时退出，0 次参数更新；已保留失败记录。`r02` 修正记录字段后从原 checkpoint 重新运行。
- 执行实例：`vlm-r11-open-h200x2-20260907`。原 Direct 与 Frozen campaign 源码、bank 及旧结果均保持独立。

两组均已完成，最后一组结束于北京时间 **01:02:33**。没有训练或评测异常，VAE、Reader、condition encoder 与 U-Net 基础权重冻结审计均通过。

| 从相同旧 512 步权重开始 | 追加更新 | 原问 EM | 四种改写合计 EM | 原问答案 token prefix |
|---|---:|---:|---:|---:|
| A：继续 flow matching | 64 | 0/8 | 0/32 | 0/8 |
| B：完整四步生成后的答案 + EOS | 64 | 0/8 | 0/32 | 0/8 |

两组原问均输出 `No music preference recorded.` 或 `No music preference`。全部 blank/donor 对照也不回答 `ambient`。当前失败包含答案本身错误，不能归因于正确答案后忘记 EOS。

复核两组第 0 步的评测记录完全一致、64 个训练噪声逐项一致。两个固定评测噪声上的 answer CE：旧权重 **10.6356**，继续 flow 后 **10.0859**，QA+EOS 后 **7.0424**。说明 QA 反馈确实提高正确答案的模型概率，但仍不足以改变 greedy 生成的答案。该概率变化不等价于系统已经会答题。

相对旧权重的 LoRA 参数变化 L2：A **0.7941**、B **1.3505**。两组均有有限非零梯度；B 的裁剪前梯度范数在 **52.04–1803.94**，没有发现梯度断开，不能仅凭梯度大就诊断数值错误或断言降低学习率能修好。训练段耗时 A **29.86 秒**、B **164.12 秒**，再次说明这是等更新步数而非等计算量对照。

原始结果：[flow-continuation.json](flow-continuation.json)、[qa-eos-continuation.json](qa-eos-continuation.json)；配对复核：[paired-summary.json](paired-summary.json)。最终 LoRA/optimizer 与八个新噪声生成的 latent/RGB 已保存在各自远端目录。

## 当前结论与尚未证明的部分

1. **已定位失败阶段**：Direct 找到正确可读终点之后，当前 rank-4 U-Net 学习方案未能从普通新噪声生成可读终点。成功 teacher 的解码、旧权重加载、旧输出复现、已检查的训练与推理计算路径均通过。
2. **已证实监督差异与残余拟合误差**：原 U-Net 学习向量场，其固定样本损失虽改善但仍有残差，问答效果没有随之成功。成功端点平均值/抽查中点会读错，说明坐标近似是否保留答案必须实测。
3. **尚未确定唯一根因**：追加 64 步原损失没有修复；改成 QA+EOS 追加 64 步也没有修复。不能宣称只是漏加 QA、只是步数少、只是 rank 太低，或已经证明平均值塌缩。两个短对照也不能证明这些路线最终不可行。
4. **当前首要瓶颈不是换问法或停止**：原问、答案首 token 已失败；更换问题模板和多题泛化还不是本次 0/8 的直接解释。

下一项最有区分力的实验应先过最小可学性门槛：固定一个普通输入噪声、一个已复测正确的 teacher，直接约束实际四步生成终点拟合该 teacher，并同时检查训练输入上的原始 QA。若这一步失败，再分别改变训练预算与可训练模块/rank；若成功，再放开噪声、加入多 teacher。这样才能把“连一个目标也未拟合好”与“多个目标的联合学习失败”分开。此项为后续建议，**本次未执行，不计入已完成结果**。

第二轮三问法 Direct 和其他 Frozen campaign 属于独立实验；本报告没有更改其数据、源码或将本次失败覆盖为成功。诊断脚本与报告已实现，但仍不能把 U-Net 记为功能通过。

## 远端证据位置

共享根目录：`/inspire/ssd/project/exploration-topic/czxs26210936`。

旧 bank/Writer：`runs/oracle-to-unet/a43b2e4-direct-trust-seed20260908` 与 `seed20260909`。

零更新诊断：`runs/unet-diagnosis-20260909/controls-r01/diagnosis.json`。

追加两组对照：`runs/unet-diagnosis-20260909/objective-{flow,qa_eos}-r02/`；调度记录 `objective-paired-r02-dispatch.json`。
