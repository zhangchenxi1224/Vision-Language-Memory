# 冻结权重、统一raw条件后的完整功能结果

5ef8aa8 完整验证冻结 bb34092 的最终权重，仅改为与训练一致的 raw 事件条件推理，没有任何参数更新。全部四路、实际导出包和独立 CLI 均已完成。单次写入和连续链全部通过，历史仍有56条失败，完整功能为1744/1800，尚不能交付为可用版本。

| 完整矩阵 | 同一bb权重，native条件 | 同一bb权重，raw条件 |
|---|---:|---:|
| 单次写入 | 280/360 | 360/360 |
| RGB连续链 | 390/480 | 480/480 |
| 完整六步链 | 7/16 | 16/16 |
| 历史前缀lane0 | 428/480 | 438/480 |
| 历史前缀lane1 | 467/480 | 466/480 |
| 历史图五问法全对 | 166/192 | 170/192 |

全部1990条raw按case、condition、噪声和query逐一配对，query/gold/event及固定负对照输出、图像均核对，见[完整配对结果](raw-condition-paired-raw-comparison.json)。单次修复80条、链修复90条，没有退步。历史lane0保留402条正确、修复36条、新增26条错误；lane1保留457条、修复9条、新增10条错误。合计修复45条却新增36条，不能用净增9条掩盖退步。

56条历史失败包括：应答juice的3条、应答linen的20条、应答约定no active preference的33条。实际错误包含silver、aluminum、未指定、未存储，以及no drink、no preference、no active material等非约定回答。严格原gold token后立即EOS的评分保持，未做同义词归一化或截断。它们仍集中于改写条件；原始历史开发1510/1510并不能替代该测试。

完整归档、独立远端SHA及本地重算如下。全部1990raw和360PNG已本地验证，远端collector实际检查原始Gaussian、全部29个有限FP32状态、latent/Reader像素和PT文件；大PT及权重仍留共享盘。单次/历史读取的是解码FP32像素，连续链则实际携带uint8 RGB并在每次写入重新经官方VAE；不把单图PNG预览等同于全部真实连续写入验证。

| 完整证据 / 本地复核 | archive SHA256 |
|---|---|
| [单次](5ef8aa8-raw-condition-confirmation-evidence.tgz) / [重算](5ef8aa8-raw-condition-confirmation-local-verification.json) | `52d47028cc3d523ec3f406082ce596dd9aa66f5e50aabc360c92b8032ab6af2d` |
| [链](5ef8aa8-raw-condition-chains-evidence.tgz) / [重算](5ef8aa8-raw-condition-chains-local-verification.json) | `271663d048d3716f4bb9b77ce1ebded06ec1a3406dadb982266a2335dfb79493` |
| [历史0](5ef8aa8-raw-condition-prefix0-evidence.tgz) / [重算](5ef8aa8-raw-condition-prefix0-local-verification.json) | `eaaa134881c901c6dc329c3a370962144c9568b0fb6897118f0081f10978f401` |
| [历史1](5ef8aa8-raw-condition-prefix1-evidence.tgz) / [重算](5ef8aa8-raw-condition-prefix1-local-verification.json) | `a40e95c510c0f18ec3aa4837d19b3d2fe518d684771c4745358e28cb0dbed7b2` |

raw策略导出使用v2 schema，显式记录training_raw，旧v1加载器拒绝静默回退。实际CLI六写三十读的36条命令/结果、30个原始读取、6个写入PNG与最终持久PNG已独立本地复核，parity_pass和reference_functional_pass均true；后者仅是第一条链，完整suite仍false。[CLI归档](5ef8aa8-raw-condition-cli-evidence.tgz) SHA256为`e9b5db3031144060162195203d6ed01ddc5681dd3a7c6b230135d4e0e3083f89`，见[本地复核](5ef8aa8-raw-condition-cli-local-verification.json)。

原生条件训练03与冻结raw推理两条路径均修复音乐清除和链错误，但历史改写分别仍64/56失败。不能从不同候选里挑成功案例拼成一个版本。下一轮b9f90e9保持03原生条件、原始初始化和全部source/teacher/noise/sigma抽样，仅扩大历史训练表述覆盖；固定计划见[历史表述对照](../official-historical-wording-training-plan-20260914.md)。它目前在四卡运行，尚无终点结果。
