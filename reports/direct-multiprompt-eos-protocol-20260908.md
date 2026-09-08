# Direct 三问法轮换训练：与原 96 起点配对

此独立分支实现用户指定的问法覆盖修正。原基线为提交 `0f704178137ee4beac952224f3175a3923d5e438`，结果根目录为 `/inspire/ssd/project/exploration-topic/czxs26210936/runs/direct-latent-geometry/0f70417-20260908-r01`。历史 `direct-latent-geometry-protocol-20260908.md` 描述原问法基线；本文件描述新实验。

- 训练：`original_open`、`paraphrase_1`、`paraphrase_2`，按零起始 update index 的 `step % 3` 轮换。
- 测试：`paraphrase_3`、`paraphrase_4`，只在固定 step256 终点评测。禁止用于梯度、早停、checkpoint 或种子选择。
- 256 次 optimizer 更新中，三个问法各使用 86、85、85 次，每次只计算一个问法的损失并更新一次。每步 prompt ID 进入 `metrics.jsonl`，断点校验核对完整轮换序列。
- 原有重复梯度预检和只读 checkpoint/终点评测保留。因此“256次”指训练更新对应的 forward/backward；整轮还包含原来就有的预检与评测计算。
- 保持答案 CE + EOS CE、EOS 权重 1、Adam LR 0.05、FP32 VAE、BF16 Reader、固定指令后缀、贪心生成 32 tokens、评分标准及 96 个初值设置不变。VAE/Reader 冻结，只训练 final latent，不增加其他损失或解码技巧。
- 每次均从原方法重新生成同一初始 latent，不从旧 endpoint 接着训练。使用独立源码和输出目录，旧结果不覆盖。

## 评测与解释

每条仍保存 257 个原始 latent、11 个原问法 checkpoint 评测和 15 个终点评测（5问法 × matched/blank/fixed donor）。保留历史 original `qa_pass` 方便比较，另外分别报告三个训练问法全对、两个保留问法全对、全部五问法全对及每问法正确数。答案前缀正确不能替代完整答案正确。

这是固定总更新预算的对照；原问法曝光量从256次降到86次，不能表述为同原问法曝光量对照。循环优化与每步三个梯度平均在 Adam 下不数值等价，是否改善由结果判断。

q3/q4 的旧基线表现已经被观察过，属于不参与此次优化的保留问法，不是研究者从未看过的盲测集。这仍是同一事实的不同问法实验，不能声称跨事实/多题泛化。

本轮固定训练结束后生成独立 summary 和几何文件，并在同一个 `dl-base-h200x4-20260907` 实例自动接续两组各512步的新 U-Net 训练与生成答题评测。原 U-Net 已完成结果及其输入目录保留。接续只以原问法完整答对决定成功latent入库，不使用保留问法分数选择目标或调整参数。当前部署见 [同实例接续记录](direct-multiprompt-unet-dl-base-20260909.md)。
