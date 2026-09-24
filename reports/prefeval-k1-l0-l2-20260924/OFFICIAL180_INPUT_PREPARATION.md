# 提前准备并封存官方 180 条 acknowledgment

2026-09-24，执行前登记。沿用原计划的冻结 Qwen3-VL-4B-Instruct Reader、官方显式偏好、
`You are a helpful assistant.`、greedy decoding、300-token 上限和现有 acknowledgment 脚本。
两组共享同一份输出，不读取未来问题、选项或 SFT 的 assistant 答复。

调整的是调度顺序：在 Writer 长训练继续时先生成 Reader acknowledgment，
减少最终评估的等待。这不依赖任何 Writer 权重，不优化 latent 或 FM，
不计算官方留出集的视觉记忆成绩，也不利用 acknowledgment 内容筛选 Writer。
输出完整 180 条后封存，保留所有截断答复；不按答复内容重采样、改写或删记录。

执行已有 `scripts/experiments/prefeval_k1_benchmark_ack.py`，代码冻结 `912335d`。
仅检查完整 ID 集、偏好原文一致性、生成 token 与预算、固定 binding 和上下文拼接，
不对 acknowledgment 的语义好坏评分。正式推理前继续固定模型与读取方案，
并披露 730 train 与官方 180 中已有的 3 条文本重复。

使用本任务 Reader 实例的现有 GPU1；原训练、完整递归评测、开发集评分继续。
官方历史中十轮干扰仍来自 benchmark context pool，不混用内部 SFT context pool。
本文件记录输入准备，不代表最终 180 条视觉记忆评测已经运行。
