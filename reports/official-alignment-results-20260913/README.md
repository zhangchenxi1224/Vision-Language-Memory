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

补全后27文件全部验证HF内容哈希，新封存payload SHA256为 `fc70dd11ccc652805388e0b0e687383c1441fbf9ea34c5384c8d7a241c7f7a72`。新的Base运行目录为 `P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`，代码 `ac34ab20b50fae82d56d1aef13fc995ee35c302a`，预算512次更新。此处在最终结果返回前不填入成功率。

`P=/inspire/ssd/project/exploration-topic/czxs26210936`。模型根为 `/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`。

## 验收与接续

修复后的源码及45项相关测试已通过；功能目标未完成。必须继续读取Base的原始generation和checkpoint，依据结果决定下一轮预算、可训练模块或功能监督实验；不能将某个进程启动、loss下降或训练完成当作目标达成。

当前运行的进程与源码目录保持固定提交；后续改动用新的checkout和输出目录。原始失败尝试、旧模型和teacher证据保留。最终确认可用还需新噪声、EOS、改写、blank/donor以及反事实事件/多题检验。
