# 03 固定终点的已观察表达对照：完整本地结果

2026-09-14 15:55：四路生成、收集及真实 CLI 已结束；所有关闭归档均按独立远端 SHA256 下载，并完成本地全量复核。固定 probe 为 `ff862dfcc5139921db3d4b8ff4b546ef1197a694`，固定模型为 03 的4832步终点，checkpoint SHA256 `d473825a403ad5c219681a09fc9e8a4270a63133393fc8ee30c0baac4c48d539`。本轮零优化步，远端 suite 完成时间为1789371674.9871664。

| 完整矩阵 | 03 | b9 | 4f |
|---|---:|---:|---:|
| 单次更新 | 360/360 | 360/360 | 360/360 |
| 六步 RGB 连续链 | 440/480 | 420/480 | 430/480 |
| 历史前缀0 | 452/480 | 480/480 | 480/480 |
| 历史前缀1 | 469/480 | 480/480 | 480/480 |
| 合计 | **1721/1800** | **1740/1800** | **1750/1800** |

完整配对读取每个模型1990条原始回答，保留190条负对照，使用原始 token 与立即 EOS 判据。新03归档还包含全部360张生成PNG；四个本地 verifier 均重新核验父终点6040条raw和19328个真实训练draw。真实CLI六写三十读逐项重放一致，但功能错误仍在；parity通过不等于功能通过。

相对03，4f保留1711条正确、修复39条历史前缀错误、新增10条链错误、40条链错误保持失败。b9保留1701条正确、同样修复39条历史错误、新增20条链错误、40条链错误保持失败。全部负对照不变。

在这套固定表达/噪声上，03已失败的40条来自 sequence0 和 sequence1 各 repetition0/3 的 step4/5：清除及随后的保持未生效，仍读出20次jazz、20次ambient。4f另退步10条，来自 sequence1 repetition2 的 step4/5，仍读出ambient。故4f的50条表达链错误中，40条是03已有薄弱点，10条才是相对03新增退步。原注册矩阵的03链480/480及4f链470/480保持原结论，不能跨不同表达/噪声矩阵混算。

这排除了“4f的50条错误全是训练遗忘”的解释。下一轮应同时覆盖原有清除薄弱点和新增退步。训练清除表达本来就包含“has been withdrawn”，不能宣称withdraw完全未见；当前表达样式和噪声没有独立交叉，不能把错误只归因于某个词。较低FM训练损失也不保证清除功能。

所有比较均是已观察回归集，没有新holdout或未见实体证据。4f同时改变初始化和学习率，本表不能作单因素因果结论。当前仍未达到完整可用标准；尚未据此启动下一轮训练。

证据：

- [完整登记](03-observed-wording-baseline-preregistered.json)，SHA256 `5bc4d57752f770f6201cfc78b96da5884850b4af4c5b41de73715ba883d4335b`。
- [全部逐格配对](03-observed-wording-paired-comparison.json)，包含03→b9和03→4f全部矩阵及全部案例。
- [独立远端完成状态与归档哈希](ff862df-completed-artifacts.json)。五个完整`.tgz`均保存在本目录，没有省略归档或只保留摘要。
- [单次本地复核](ff862df-fresh-wording-confirmation-local-verification.json)、[连续链本地复核](ff862df-fresh-wording-chains-local-verification.json)、[历史0本地复核](ff862df-fresh-wording-prefix0-local-verification.json)、[历史1本地复核](ff862df-fresh-wording-prefix1-local-verification.json)、[CLI本地复核](ff862df-fresh-wording-cli-local-verification.json)。
- [早期关闭raw配对](ff862df-closed-raw-paired-summary.json)保留原字节及其较窄证据范围；正式结论以完整归档和上述本地复核为依据。大体积模型PT在远端经过校验，本地归档不包含完整权重。
