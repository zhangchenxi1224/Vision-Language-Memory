# 官方对齐：真实实验结果与当前状态

目标仍在进行。实现偏差已修复并有数值证据，但目前还没有通过功能验收的 Writer。

## 已完成：Mobile 官方 FM 单目标试验

- 训练代码：`106c8eb62c4da4b752a60866fe74bae880ce1120`。
- 实例：`dl-align-h200x2-20260913-r2`，新建的双 H200；原始镜像 PyTorch `2.7.0a0+ecf3bae40a.nv25.02`、CUDA12.8、Diffusers0.39.0、Transformers4.57.3。
- 成功库来自同一道 ambient 题的 96 个三训练问法 Direct 终点。训练 teacher 按哈希从训练成员中预先选定，未根据测试输出选择。使用真实原问重新解码验证，输出 token 为 `ambient` 后立即 EOS。
- rank16、alpha16、lr5e-5、AdamW weight_decay1e-4、累积4、clip1；从预训练 Mobile 新建 LoRA，512 次更新、2048 个独立训练噪声。
- 实际 sigma 范围 `[0.00142616, 0.99921983]`，1025 个样本大于0.5。目标侧无 source 混合。
- 生成从纯噪声开始。原始 sigma `[1,.75,.5,.25]` 对应实际有效 sigma `[1, .904530764, .759510934, .512844145]`，最后一步到0。这是 scheduler 的实际偏移结果，不能用原始序列冒充有效序列。
- 严格确定性、FP32 条件下，实际官方 Mobile pipeline 与本地采样器初态和四步轨迹逐位一致：最大绝对误差0。
- 结果 JSON 与最终 checkpoint SHA256 已从共享盘重新计算核验通过。

| 相同8个评测噪声 | 原问 | 改写1 | 改写2 | 改写3 | 改写4 |
|---|---:|---:|---:|---:|---:|
| 未训练 Mobile | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| 官方 FM 微调512步 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

各阶段50条真实 greedy generation：8×5 matched，加 blank/donor 各5条。对照图像在训练前后相同；blank/donor 未答对。teacher 的成功不是 U-Net 的成功。仍是单题机制检验，不能推断事件条件写入或未见题泛化。

训练 loss 首64步均值0.752814，末64步均值0.439556；使用不同随机样本，属于未配对训练统计。LoRA 参数变化 L2=12.637723。损失和参数确实变化，但40个 paired matched 输出没有任何错误转正确，因此不得称为可用。

![相同噪声训练前后真实输出](mobile-preview.png)

上排是训练前，下排是训练后。训练后主要产生纹理，仍没有形成 Reader 可读的目标信息。这是图像与原始答题记录支持的现象；不能仅据该现象断言唯一原因是容量、预算、蒸馏或 teacher 目标的鲁棒性。

原始完整文本证据压缩包：[mobile-evidence.tgz](mobile-evidence.tgz)，重新核验摘要：[verified-summary.json](mobile/verified-summary.json)。原始生成记录：[训练前](mobile/train/baseline/generations.jsonl)、[训练后](mobile/train/trained/generations.jsonl)。

## Base 官方训练对象对照

Base 臂直接使用锁定的官方 `DreamLitePipelineLoRA`，原始事件提示词训练、512像素官方 conditioner，评测直接执行官方28步和三路CFG，未重新实现Base采样循环。保留FP32已验证raw latent目标；VAE权重与Mobile逐字节相同，并检查VAE架构及latent缩放约定。

旧 Base 下载目录只有26个文件，缺少text_encoder权重，首个Base尝试在0次更新时退出。已补全并新增组件完整性检查：HF官方 revision `a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa` 明确要求4,255,140,312字节权重，SHA256为 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。现有Mobile缓存权重与该官方Base文件的SHA256完全相同，故复用相同内容补全，未用近似模型替换。见 [官方文件元数据](base-official-file-metadata.json)。

补全后27文件全部验证HF内容哈希，新封存payload SHA256为 `fc70dd11ccc652805388e0b0e687383c1441fbf9ea34c5384c8d7a241c7f7a72`。Base运行目录为 `P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`，代码 `ac34ab20b50fae82d56d1aef13fc995ee35c302a`，已完成512次更新。

Base训练前和训练后，原问和4个改写问法均为 **0/8**，严格答案加立即EOS同样为0/8；全部40个matched单元都没有错误转正确。首末64步未配对训练loss均值为0.684343/0.436569，adapter delta L2为13.174183。结果与checkpoint重新计算SHA核验通过，result SHA为 `afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4`。

![Base原生28步生成对比](base-preview.png)

Base预训练输出也包含列车，训练后出现纹理及残留列车。与事件标识中的train词存在语义联系，但这只是观察，不足以认定唯一原因。原始证据：[base-evidence.tgz](base-evidence.tgz)、[核验摘要](base-summary.json)。

## Oracle可读范围诊断

固定代码 `1bb0d28`，原始Base512输出seed0与同一个预先选定teacher；不选新teacher、不做优化。48条真实生成记录及其SHA均已复核。仅测试原问和预先固定的paraphrase_4，结果不得替代Writer准确率。

| 加到teacher上的等方差噪声RMS | 原问正确且立即EOS | paraphrase_4正确且立即EOS |
|---|---:|---:|
| 0（原始teacher） | 1/1 | 1/1 |
| 0.001 | 3/3 | 3/3 |
| 0.003 | 3/3 | 3/3 |
| 0.01 | 3/3 | 3/3 |
| 0.03 | 3/3 | 3/3 |
| 0.1 | 3/3 | 2/3 |
| 0.3 | 0/3 | 0/3 |

固定3个扰动方向，幅度之间复用方向做配对。沿teacher到真实Base512 seed0的方向，插值比例0.01、0.03、0.1、0.3均在两个问法上通过；纯Writer端点失败。纯Writer端点距teacher的RMS为0.444540，因此比例0.3对应约0.133362。

这排除了“本teacher只有精确坐标才可读”的强假设，但样本少，不能证明任意方向都鲁棒。结果支持先区分拟合程度与推理引导影响，再决定训练容量；不支持宣称模型已经可用。证据：[neighborhood-evidence.tgz](neighborhood-evidence.tgz)、[完整raw generations](neighborhood/generations.jsonl)、[完成与哈希](neighborhood/complete.json)。

固定CFG=1对照已完成，使用同一Base512 checkpoint、8个噪声、28步原生pipeline、source、prompt和Reader，只改变官方支持的guidance_scale。50条生成文本哈希已核验，五问法均0/8；因此降低引导并未救回这个检查点。训练不变，也不以这个诊断回写既有结果。证据：[guidance1-evidence.tgz](guidance1-evidence.tgz)。

已据此启动3500步官方预算对照，沿用Base512固定提交和所有超参数，重新初始化相同LoRA与噪声序列，只更改明确绑定的预算和输出目录。见[调度前预注册](../official-base-3500-preregistration-20260913.md)。当前8个评测噪声未参与优化，但已在研究迭代中观察；任何正结果须追加全新噪声和反事实事件检验。

`P=/inspire/ssd/project/exploration-topic/czxs26210936`。模型根为 `/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`。

## 验收与接续

修复后的源码及46项相关测试已通过，含CFG参数、异常恢复、答案正确但缺EOS及缺失/重复评测问法的测试。功能目标未完成。Base原始generation和checkpoint已复核，应继续依据诊断结果决定下一轮预算、可训练模块或功能监督实验；不能将某个进程启动、loss下降或训练完成当作目标达成。

当前运行的进程与源码目录保持固定提交；后续改动用新的checkout和输出目录。原始失败尝试、旧模型和teacher证据保留。最终确认可用还需新噪声、EOS、改写、blank/donor以及反事实事件/多题检验。
