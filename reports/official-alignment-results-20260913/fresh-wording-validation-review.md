# b9 固定新表述与新噪声完整验证

56b56a8 对同一 b9f90e9 固定终点完成四路验证及独立 CLI，完整功能 **1740/1800**，仍有 60 条连续链失败。全部 1990 条原始回答与 360 张生成 PNG 已下载并通过本地全量复核，不能据单次和历史前缀全对认定模型可用。

| 固定矩阵 | 严格答案+立即 EOS | 五问全对图 |
|---|---:|---:|
| 单次写入 | 360/360 | 72/72 |
| 六步真实 RGB 链 | 420/480 | 84/96 |
| 历史前缀 lane0 | 480/480 | 96/96 |
| 历史前缀 lane1 | 480/480 | 96/96 |

完整链为 **10/16**。60 条失败来自 sequence-0 和 sequence-1 的 repetition-0、2、3，各在 step-4 清空时首次出现，随后 step-5 保持不变仍保留旧值。期望均为 `no active preference`，实际 30 条回答 `jazz`、30 条回答 `ambient`，每张失败图五问均错。原文、事件、种子、全部 token 和其他成功/负对照均保留在[链归档](56b56a8-fresh-wording-chains-evidence.tgz)和[完整摘要](56b56a8-fresh-wording-chains-summary.json)。

该计划在 b9 终点产生前固定：[计划原字节](fresh-wording-preregistered-validation.json) SHA256 `ba77e8c2ab8742bdcda703ae12071b504ea99c33173d0ebad314ddc10292fb6b`。所有新事件文本与原银行、历史九种训练表达及旧验证无交集；噪声与训练、开发及旧验证无交集。实体、语义问题及五种 Reader 提问仍已见。此结果支持新事件表述测试的失败结论，不代表未见实体或任意多事实泛化。与旧案例的 1780/1800 不能逐格配对，因为事件文本和噪声不同。

| 完整本地归档 / 复核 | 独立观察且本地核对的 SHA256 |
|---|---|
| [单次](56b56a8-fresh-wording-confirmation-evidence.tgz) / [复核](56b56a8-fresh-wording-confirmation-local-verification.json) | `57bac3c27e408fe181585c9e0d5d3082ca8557d35e4e475f3ceff314726414cf` |
| [连续链](56b56a8-fresh-wording-chains-evidence.tgz) / [复核](56b56a8-fresh-wording-chains-local-verification.json) | `46e90a17acf52b001b4b31e1eabf93094fc7208d2905593e44929c247e771fe4` |
| [历史0](56b56a8-fresh-wording-prefix0-evidence.tgz) / [复核](56b56a8-fresh-wording-prefix0-local-verification.json) | `5c5899ea6700dd0373154c45b54f00e1fc201bd668f8d3d66d3183fd4be05eea` |
| [历史1](56b56a8-fresh-wording-prefix1-evidence.tgz) / [复核](56b56a8-fresh-wording-prefix1-local-verification.json) | `c882c2c4bd72545c5cb2474570b5b0ad2c0d8f99bd5d5952556a8f576f6fe197` |

四路均以相同 b9 父实验完整证据重放 19328 次训练抽样，并显式使用 `fresh_wording_v1` 的完整计划检查事件、问题、种子和原始评分。四份摘要从验证归档提取原字节后，分别与独立远端摘要 SHA 核对。原参数、PT、Gaussian 和全部 FP32 轨迹在远端实际审计；本地持有完整 raw 与 PNG，不声称本地持有大张量。

[独立 CLI 归档](56b56a8-fresh-wording-cli-evidence.tgz)为 6235501 字节，SHA256 `97ccb236df0ed87149b20cfc299d8a083aa8b577ef0f9c315981783669cef020`。[本地复核](56b56a8-fresh-wording-cli-local-verification.json)重算 36 条命令/结果、6 次写入、30 次读取及持久 PNG，`parity_pass=true`。`reference_functional_pass=false`来自原完整链验证，四路功能也失败。部署还需独立完整 PNG 读回；不能用工程一致性抹去清空失败。
