# 全部旧/新案例的实际 PNG 读取

38531f0 完成两套全部 **3980 条原始读取、3600 条匹配答案、796 张实际输入 PNG**。八路归档、原始源归档及所有 PNG 像素均已独立本地复核，结果 **3520/3600**。PNG 路径没有新增或修复匹配答案错误：原来 3520 条正确全部保留，80 条连续链错误仍在。模型仍不可用。

| PNG 验证 | 单次 | 连续链 | 历史0 | 历史1 | 合计 |
|---|---:|---:|---:|---:|---:|
| 原有案例 | 360/360 | 460/480 | 480/480 | 480/480 | 1780/1800 |
| 固定新表述 | 360/360 | 420/480 | 480/480 | 480/480 | 1740/1800 |

输入逐张执行 PIL RGB PNG → uint8 数组拷贝 → NCHW FP32/255，与实际 CLI 相同。matched 使用原验证封存的 720 张 PNG；其余 76 张 blank/donor 由原封存 PT 按固定规则舍入后保存、重新打开。冻结 Reader 先盲生成，之后按原 token+立即 EOS 计分，没有先执行带目标答案的 CE。两套连续链全部 960 行的实际像素和生成 token 与原始链结果一致。

该实验验证 PNG 存取和 Reader 读取路径；原 FP32 PT 及轨迹仍只在远端实际检查，本地不冒充持有这些大张量。单次与历史图虽然是从 FP32 量化到 PNG，在此次完整匹配矩阵中没有造成正确/错误翻转。这不能推出任意图片或后续模型都不受量化影响，也不能解释或抹去已有清空失败。

[完整本地汇总](png-readback-local-aggregate.json)含八路逐项计数、配对变化及像素检查；[独立远端哈希记录](png-readback-observed-artifacts.json)含每个归档与摘要的 SHA256。[原摘要与终态归档](38531f0-png-readback-summaries-evidence.tgz) SHA256 为 `2c0bef56a292598ee2a956c3f0ce90f8691a63cab9f65bd35f84af5cf10564f2`；八份原始摘要和终态按原字节提取，每份独立哈希与完整本地重算一致。

| 套件 | 完整归档与本地实际像素复核 |
|---|---|
| 原单次 | [归档](38531f0-png-readback-registered-confirmation-evidence.tgz) / [复核](38531f0-png-readback-registered-confirmation-local-verification.json) |
| 原连续链 | [归档](38531f0-png-readback-registered-chains-evidence.tgz) / [复核](38531f0-png-readback-registered-chains-local-verification.json) |
| 原历史0 | [归档](38531f0-png-readback-registered-prefix0-evidence.tgz) / [复核](38531f0-png-readback-registered-prefix0-local-verification.json) |
| 原历史1 | [归档](38531f0-png-readback-registered-prefix1-evidence.tgz) / [复核](38531f0-png-readback-registered-prefix1-local-verification.json) |
| 新单次 | [归档](38531f0-png-readback-fresh_wording_v1-confirmation-evidence.tgz) / [复核](38531f0-png-readback-fresh_wording_v1-confirmation-local-verification.json) |
| 新连续链 | [归档](38531f0-png-readback-fresh_wording_v1-chains-evidence.tgz) / [复核](38531f0-png-readback-fresh_wording_v1-chains-local-verification.json) |
| 新历史0 | [归档](38531f0-png-readback-fresh_wording_v1-prefix0-evidence.tgz) / [复核](38531f0-png-readback-fresh_wording_v1-prefix0-local-verification.json) |
| 新历史1 | [归档](38531f0-png-readback-fresh_wording_v1-prefix1-evidence.tgz) / [复核](38531f0-png-readback-fresh_wording_v1-prefix1-local-verification.json) |

相应完整源 PNG 在 [7b 旧案例证据](historical-wording-validation-review.md)和 [56 新表述证据](fresh-wording-validation-review.md)中。可支持范围仍为已见实体与语义问题、预先固定的新事件表述；后续应针对连续写入中的清空失败做源图与权重交叉诊断，再验证实际修复。
