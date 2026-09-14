# 继续训练后的完整原注册功能评估

4fbc857终点的e372f3c原注册功能评估已完成全量本地复核：**1790/1800**。全部1990条raw、360张生成PNG、190条负对照完整保留。另有实际CLI的6次写入/30次读取一致性验证；仍有10条连续清除错误，尚不可用。

| 功能分区 | 03原生条件 | b9表达增强 | 本轮4f继续训练 |
|---|---:|---:|---:|
| 单次写入 | 360/360 | 360/360 | 360/360 |
| 连续RGB写入 | 480/480 | 460/480 | 470/480 |
| 历史前缀0 | 416/480 | 480/480 | 480/480 |
| 历史前缀1 | 480/480 | 480/480 | 480/480 |
| 合计 | 1736/1800 | 1780/1800 | 1790/1800 |

[完整逐格配对](clear-retention-registered-paired-comparison.json)核对相同case、condition、noise、query、事件语义及原始token/EOS；所有负对照图像哈希与token未改变。相对03保留1726条正确、修复64条历史表达、退步10条链读取；相对b9保留1780条、修复10条、仍错10条，没有新增退步。配对覆盖全部1800格，非只看总分或成功子集。

剩余错误是`sequence-0-rep-2-step-4`清除及`sequence-0-rep-2-step-5`保持不变，各5种问题应答`no active preference`，实际均为`jazz`。详见[链与CLI完整证据](clear-retention-chain-review.md)。初始化与学习率共同变化，不能将改善单独归因于降低学习率。

单次和链的完整归档、原摘要、本地复核见上述链报告。历史两分路已完整分块重建并本地验证：

- [历史0全部分块与重建方法](e372f3c-logical-prefix0-evidence.tgz.parts/README.md)、[560条raw/96PNG本地复核](e372f3c-logical-prefix0-local-verification.json)。原归档105641188字节，SHA`20c7078016c20ea73bed6367cc83da4833d54199ac318e0288ded1be1f07457b`。
- [历史1全部分块与重建方法](e372f3c-logical-prefix1-evidence.tgz.parts/README.md)、[560条raw/96PNG本地复核](e372f3c-logical-prefix1-local-verification.json)。原归档105553263字节，SHA`f665810884ac21bff74184decbe1870bb362155b341b393af9020a50d6aeba1b`。

完整分块保留每个归档字节；没有为了GitHub限制省略PNG或原始读取。大PT和checkpoint在远端实际检查，本地归档没有这些大张量，不能声称本地重读了全部PT。

另一套[已观察表达回归](clear-retention-observed-wording-review.md)已完成全部本地复核1750/1800；[本轮0c4d893完整PNG读回](clear-retention-png-validation-review.md)也完成3980 raw、796张PNG本地复核3540/3600，无答案翻转。Goal active，当前仍须修复连续清除问题。
