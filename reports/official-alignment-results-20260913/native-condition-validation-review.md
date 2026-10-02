# 原生条件训练后的完整功能验证

03f8467 完成相同初始化、4832 次更新和 19328 次抽样的训练；9e27050 完整四路验证及实际 CLI 重放均已结束。开发集 1510/1510，完整功能矩阵 1736/1800，仍有 64 条严格答案/EOS 失败，不能认定当前版本可用。

| 完整矩阵 | 原 bb34092 | 原生条件训练 03f8467 |
|---|---:|---:|
| 单次写入 | 280/360 | 360/360 |
| 六步 RGB 连续写入 | 390/480 | 480/480 |
| 完整链通过 | 7/16 | 16/16 |
| 历史前缀 lane0 | 428/480 | 416/480 |
| 历史前缀 lane1 | 467/480 | 480/480 |
| 历史图五问法全对 | 166/192 | 170/192 |

这是同一组已观察案例的配对诊断，不是新的未见测试集。全部 1990 条原始回答（含负对照）和 360 张生成 PNG 保留；查询、事件、噪声种子和评分没有改变。完整原始输出的[逐格比较](native-condition-paired-raw-comparison.json)显示：单次新增 80 条正确、连续链新增 90 条正确，两者没有退步。历史 lane0 保留 413 条正确、修复 3 条，但新增 15 条错误；lane1 保留 467 条并修复 13 条。因此历史总分净增 1 条掩盖了实际退步。

64 条失败都来自 lane0 的两种改写，原始历史表述通过。饮品更新目标 juice 失败 36 条，材质更新目标 linen 失败 17 条，清除目标 no active preference 失败 11 条。输出包括旧值 coffee、物体颜色 silver、未指定/未记录回答和非约定的 no preference，不能通过同义词归一化补记为正确。完整事件和各原始回答计数见 [lane0 摘要](9e27050-logical-prefix0-summary.json)。这支持对历史表达敏感的诊断，尚不能证明唯一原因。

验证直接执行官方 Base 28 步、CFG1、纯 Gaussian 初态和 FP32 轨迹；每次写入使用当前实际 RGB 图，经官方 VAE 编码，Reader 查询不改写记忆。远端 collector 实际检查所有 PT、Gaussian、29 个有限 FP32 状态及 Reader 像素；本地依据独立观察的远端 SHA256 重算全部 raw、身份绑定、PNG 和原训练抽样。大型 PT 与参数权重仍在共享盘。

| 完整本地证据 | 远端观察并与本地核对的 archive SHA256 |
|---|---|
| [单次](9e27050-logical-confirmation-evidence.tgz) / [复核](9e27050-logical-confirmation-local-verification.json) | `27f91692a1ba02c3ff23c0ea41344c8ad23a0b79945478b4b65ead33fd133af3` |
| [连续链](9e27050-logical-chains-evidence.tgz) / [复核](9e27050-logical-chains-local-verification.json) | `e546637691a54c250ed8562321a78c1cfdb2e75bd18e973b782b7210f7156c20` |
| [历史0](9e27050-logical-prefix0-evidence.tgz) / [复核](9e27050-logical-prefix0-local-verification.json) | `7d7588c7d18f63715803991d108d9206f7adfd0e9666dfe7640205c2f9e5a545` |
| [历史1](9e27050-logical-prefix1-evidence.tgz) / [复核](9e27050-logical-prefix1-local-verification.json) | `4e77e7a216a4df7dd648be53db464f03502d6cf230e8b1fa0fd7d9f152d039d0` |

导出包完成独立的六写三十读 CLI 推理；本地重算全部 36 条命令/结果、30 条原始读取、6 张写入 PNG 及最终持久 PNG，`parity_pass=true`、`reference_functional_pass=true`。后者取自原连续链验证的整体通过标志；CLI 本身只重放第一条链。完整四路 suite 的 `functional_all_registered_correct=false`，不能据 CLI 一致或连续链通过宣布可用。见 [CLI 完整证据](9e27050-logical-cli-evidence.tgz) 和 [本地复核](9e27050-logical-cli-local-verification.json)，archive SHA256 为 `0de0462d52c05d2582e39936d38f581d73018e33329bf1556379ec1cfb9f349f`。

冻结旧 bb 权重的 raw 条件开发对照同样为 1510/1510，其完整功能实验已在新四卡运行；必须单独等待全部案例和真实导出包结果，不能把两种条件或两个模型的通过案例拼接为一个可用版本。
