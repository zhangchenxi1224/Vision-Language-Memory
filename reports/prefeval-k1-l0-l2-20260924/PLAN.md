# PrefEval 单信息视觉记忆 L0–L2：执行登记

2026-09-24。本轮基于 3fd9958 的 A/B 方案，范围仅 K1，不做 K2/K4。
这是视觉记忆扩展实验，不声称复现官方文本 SFT 模型。

## 核心判断

一张图答对一题并不足以证明共享可写记忆。依次分开检验：
1. L0：固定训练偏好的目标 PNG 能支持官方应用题及换问法读取。
2. L1：同一共享 Writer 写入不同单偏好状态；进一步用同主题新偏好替换旧偏好，
   仍只有一个有效偏好。覆盖事件和评价另行登记为扩展；清除没有现成官方四选项 gold，
   不把任一官方选项冒充“无偏好”，不能仅凭保持实验宣称删除完成。
3. L2：每次输入只有上一步实际 PNG 和当前完整 user/assistant 交换，
   每次都执行原生生成并保存/重读 PNG。对 0/5/10 干扰截面读取，跨步不传 latent 或文本历史。

优先诊断顺序：教师目标可读性 → 单写蒸馏落差 → 未训练偏好泛化 → RGB 连续保持。
失败时保留完整分母，定位后只改变对应环节。不能将教师结果替代 Writer 结果。

## 配对方案

沿用官方 topic split，730 train / 90 internal-dev / 180 official-eval；既定 64 pilot 不重抽。
A：官方 response_to_q 的全部答案 token 含结束符平均 CE。
B：原官方 MCQ 内容、官方提示与解析函数，完整 choice 短回答含结束符平均 CE。
每 12 步均衡三训练问法 × 四正确位置，错项顺序也打乱。
同一状态只优化一个 latent，T1/T2/T3 循环；O1/O2 不进入梯度或模型选择。

问法由逐题人工语义拆解（场景、请求对象、动作）生成；不读取偏好来添加条件。
保留实体、时间、范围和否定。家族使用一致句法，因此只检验这两个固定家族的措辞迁移，
不宣称自由自然语言分布、全新语义用途或相对单问法的因果收益。
原题官方格式成绩单列；同一冻结 PNG 回答全部问题。

执行后、首次 OOD 推理前，修正 travel_hotel:0052、lifestyle_beauty:0038、
lifestyle_health:0040 的 O1 冠词语法；T1/T2/T3/O2 和答案未变（逐条比较验证）。
目标训练固定于 5153444 的原清单，评估使用修正清单并保存实际 query。
90 条 internal-dev 的单独问法表也在推理前完成，不产生任何可优化的教师目标。

目标阶段每组 64 × 288 Adam 更新，lr=.05；原生灰图 VAE 初始化、冻结 FP32 Tiny VAE
及 BF16 Qwen3-VL-4B Reader。RGB 前向量化等价 uint8，反向 STE；最终从磁盘 PNG 评价。
不使用旧 answer_mean_CE + EOS_CE 加权。技术 smoke 每组同样 1 样本 × 12 步，独立目录，
只验证运行与梯度，不用于选参数；正式两组均从灰图 fresh optimizer 开始。

FM 两组从 4fbc857 的最终 checkpoint 开始，各 2,048 次更新、effective batch=4、
AdamW lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=1e-4, clip=1。
复用 official_flow_bridge、predict_velocity 和 native 条件编码：source 只在条件侧，
target/noise 桥覆盖完整 sigma，整数训练 timestep；推理原生 28 步、CFG=1、纯高斯起点。
保持阶段各 2,048 步；同组单写模型生成 64 条十交换真实 RGB 前缀，
1/2 初写、1/2 保持，目标仍为真实偏好；冻结终点后重新生成完整评测链。

评测包括官方原题自由回答与 MCQ，空图、错配图、完整文本参照及两条固定噪声链。
除每张图的准确率，另报同一偏好的两条噪声链都正确的比例，分母为偏好数；
任一链缺失或尚未评分时，该偏好不能计为完整结果，不把重复噪声链当独立偏好样本。
自由回答用官方四项 judge 提示和聚合，不按训练答案逐字匹配；若 judge 不可用，
保存原始回答并明确该指标未完成，不用自定词匹配冒充官方得分。
先以内部开发 T1 定位；官方 180 只在流程冻结后跑。研究中的旧主题暴露历史沿用原方案披露。

## 资源与产物

禁止使用 dl-clear-retain-h200x4-20260914（用户已分配其他实验）。
新申请 prefeval-k1-l012-h200x4-20260924，训练区-H200-1号机房-cuda13.2，4 H200，
80 CPU / 900 GiB，NGC 25.02，128 GiB shm。该规格平台只允许 priority=1，可能被抢占；
每 24 个目标更新保留 latent+Adam 状态，正式预算不因续跑增加。
GPU 0/1 为 A 两分片、GPU 2/3 为 B 两分片；后续 FM 各占一张卡也可保持有效 batch。
代码分支 codex/prefeval-k1-l0-l2-20260924，独立共享盘运行根 runs/prefeval-k1-l0-l2-20260924。
现有 CPU 实例只用于 Git 同步和共享盘文件准备。不覆盖既有模型、环境及其他实验目录。

初次技术启动因遗漏既有确定性环境变量而在梯度前失败，日志保存在
smoke-env-failed-61ce035；5153444 修正环境后，A/B 各 12 步及 PNG 回读通过。
官方自由回答 judge 暂无可用入口，本机已有 AI Gateway 鉴权返回 401；
已询问用户可用评分服务，不影响教师/FM/MCQ 推进。评分入口复用官方四份提示、
两个 XML parser 及 analyze_errors，官方 100-token/temperature=0 配置；
无有效 judge 结果前不得填报自由回答遵循准确率。

## L1 单有效偏好覆盖探针（模型冻结后执行）

使用续训后 Writer 的真实 10 干扰 PNG 作为旧状态，按不依赖成绩的固定哈希在同主题中
配对一条不同旧偏好。输入新 user 交换为 `My earlier preference no longer applies.
My current preference is: <new official preference>`，assistant 保留新偏好的官方 acknowledgment。
这是一条自然语言覆盖请求，没有 oracle SET/RETAIN/CLEAR 标签；Writer 看不到未来问题/选项/答案。
只生成一张新 PNG，再用新偏好对应的官方原题/答案读取。旧与新始终各只有一条有效偏好，
不涉及多槽选择性编辑。报告为本项目覆盖扩展，不能冒充官方原始 benchmark。
与同一最终 Writer 从灰图写入相同新偏好的结果配对，判断既有图像状态是否妨碍改写。
若覆盖失败，再登记下一轮包含训练侧覆盖转移的 FM 预算；本轮不提前加入其训练样本。

## 最终 180 条官方主题评估的输入

冻结 Writer 后使用 `benchmark_dataset/explicit_preference` 的 180 条 eval-topic 记录。
先由同一冻结 Qwen Reader 对官方偏好生成 acknowledgment，输入只有偏好，
使用官方 `You are a helpful assistant.` 和 300-token 上限、greedy decoding。
两组共享这一份 acknowledgment；它不含未来问题、选项或发布的 SFT 回答。
十轮干扰使用官方 benchmark context pool 的前十个完整交换，
不沿用内部 pilot/dev 使用的 SFT context pool。Writer 每步仍只接收前一张 RGB 和当前交换。

`official-paraphrase-requests.json` 逐题只从原问题抽取请求，保留地点、时间、用途、
数值及原题自身明确给出的条件；`official-question-forms.json` 保存最终五问法。
T1 为官方原文，其余为直接问句、指令、情境陈述和条件请求四种固定家族。
这些是措辞扩展，不是新的语义任务。官方原题若已经包含部分偏好提示，保留原文，
并用空图/错配图/全文参照判断视觉记忆是否有额外贡献。
当前仅准备输入和脚本，尚未生成 acknowledgment、优化这些偏好的 latent 或读取评估成绩。

最终读取继续使用本实验固定的 Qwen 图像+问题接口；它是视觉记忆适配，
不声称逐 token 复现官方各文本模型的聊天模板。自由回答保留官方 300-word 提示和
300-token 生成上限；MCQ 保留官方内容及 XML parser，但给 32 token 以容纳 Qwen 的完整 XML，
单独披露区别于官方配置的 5-token 上限。所有截断输出仍纳入固定分母并单列截断率。

## 730 条训练规模的问法准备

`train-paraphrase-requests.json` 补齐 pilot 之外的 666 条逐题请求描述，仅依据官方原问题，
保留实体、地点、时间、数量、用途及原题明示的条件。它不读取偏好、选项或目标答案来补题。
`python scripts/experiments/prefeval_k1_data.py --split train` 生成 `train-question-forms.json`：
64 条 pilot 的五问法逐字复制既有冻结文件；其余 666 条使用与官方主题评估相同的
直接问句、指令、情境陈述及条件请求家族，T1 始终保留原题。
因此这些是新条目的未训练措辞，不声称句式家族或语义用途全新。

730 条覆盖检查、原题保留、pilot 一致性、与 90 条内部开发及 180 条最终评估的
偏好 ID 隔离均通过；原题中的数字也逐条保留。该材料已备齐，但扩容训练尚未启动。
当前 GPU 仍执行原冻结 64 条实现，先完成该轮教师、FM 和真实 RGB 保持诊断，
再依据内部 T1 结果确定扩容与纠因迭代。新增文件不改变当前运行的训练或检查点选择。
