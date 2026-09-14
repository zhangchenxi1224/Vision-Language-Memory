# 历史表达增强后的完整旧案例验证

b9f90e9 的固定 4832 更新终点完成开发 1510/1510；7b82309 四路验证和独立 CLI 已全部结束。完整旧案例功能为 **1780/1800**，仍有 20 条错误，当前模型不能判定可用。

| 完整矩阵 | 原生条件训练 03f8467 | 历史表达增强 b9f90e9 |
|---|---:|---:|
| 单次写入 | 360/360 | 360/360 |
| 六步真实 RGB 连续写入 | 480/480 | 460/480 |
| 完整链通过 | 16/16 | 14/16 |
| 历史前缀 lane0 | 416/480 | 480/480 |
| 历史前缀 lane1 | 480/480 | 480/480 |
| 全部匹配答案 | 1736/1800 | 1780/1800 |

全部 1990 条原始回答、360 张生成 PNG 与 190 条负对照均保留并完成本地复核。完整[逐格配对](historical-wording-paired-raw-comparison.json)使用同一案例、问题、事件、噪声及严格 token+立即 EOS 规则：保留 1716 条正确，修复 64 条历史错误，同时新增 20 条连续写入错误。负对照的图像哈希和 token 未变化。这是已观察案例上的回归诊断，不是新的未见测试集。

20 条失败全部输出 `jazz`，期望均为 `no active preference`，具体来自 sequence-0 的 repetition-2 和 repetition-3。两次 step-4 清空后仍读出 jazz；随后 step-5 的保持不变操作也保留 jazz。四张图各五问均失败；同组 repetition-0、repetition-1 的清空和保持不变通过。完整链图中 92/96 张五问全对。此证据定位了首次失败和后续传播，尚未区分清空表述敏感与生成源图分布变化的因果贡献，不能据总分改善掩盖退步。

训练唯一变化及完整开发证明见[开发报告](historical-wording-development-review.md)。推理保持同一官方 Base 28 步、CFG1、纯 Gaussian、FP32，每次连续写入都携带真实生成 RGB，经官方 VAE 重新编码。源图只作条件，未混入 FM 的目标侧加噪或速度标签。单次及历史前缀仍读取解码 FP32 图；其 PNG 部署结论必须等待独立的[完整 PNG 读回](../official-png-readback-acceptance-20260914.md)。

| 本地完整证据 | 独立远端 SHA256，已与本地完整归档一致 |
|---|---|
| [单次归档](7b82309-logical-confirmation-evidence.tgz) / [复核](7b82309-logical-confirmation-local-verification.json) | `64aa242ca29aa087ba34487ab7dd7028e92255fd424e402f26745be8c05fdd8f` |
| [连续链归档](7b82309-logical-chains-evidence.tgz) / [复核](7b82309-logical-chains-local-verification.json) | `259aba3a18cb4423212f7e520aa1de7ca8c80b63d53c712320803153f444beb0` |
| [历史0归档](7b82309-logical-prefix0-evidence.tgz) / [复核](7b82309-logical-prefix0-local-verification.json) | `9e5d422dc7ebc9b6d91373789ce2ef3370398207b147577333c24aba8ef9ff8b` |
| [历史1归档](7b82309-logical-prefix1-evidence.tgz) / [复核](7b82309-logical-prefix1-local-verification.json) | `620f2ea493da5be360201b7588fb7d42ecdf6bbc11711de46c285c51f4604927` |

本地 verifier 重算全部原始回答、PNG 身份、训练 19328 次抽样以及父实验绑定；远端 collector 另实际检查源 PT、Gaussian、29 个有限 FP32 状态和 Reader 像素。参数与大张量仍在远端共享盘，本地不冒充持有或重算这些大张量。三个大归档因传输停顿改用 8 MiB 分块；最终组装字节仍与原归档的独立 SHA256 完全一致，未裁剪证据。

[CLI 归档](7b82309-logical-cli-evidence.tgz)为 6447693 字节，SHA256 `7336c24849dffed5b19e5e2342e57b291c3942c93bc559c2ecac68dd83b83047`。[本地复核](7b82309-logical-cli-local-verification.json)检查完整 36 条命令/结果、30 条读取、6 张写入 PNG 及最终持久图，`parity_pass=true`。`reference_functional_pass=false`取自原连续链验证的整体结果；它不是对重放第一条链的单独评分。四路总体验收也为 false，工程重放一致不能替代功能正确。

新表达与新噪声的 56b56a8 验收已独立运行，随后执行全部旧/新 PNG 读回。它们尚未完成本地全量复核，不能拼接两个模型或挑选各自成功案例来宣布一个可用版本。后续因果排查可固定同一清空事件与噪声，交叉使用 03/b9 权重和各自生成的 step-3 源 PNG，并纳入同组四次重复的清空与后续保持不变；此诊断尚未执行。
