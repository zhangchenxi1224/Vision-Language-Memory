# 冻结旧模型，只改变推理条件：完整开发同样全通过

1f86d56完整28步raw条件对照已完成，独立CPU与本地全部原始记录复核通过。固定bb34092最终checkpoint、零优化更新，仅将推理条件改成训练时的raw事件embedding与mask：开发从 **1460/1510提升至1510/1510**，302张图五问法全部通过。原1460条正确回答全部保留，原50条错误全部修复，全部1510条blank/donor对照的像素与生成token保持不变。

| 固定开发集合 | 同一checkpoint原生条件推理 | 同一checkpoint raw训练条件推理 |
| --- | ---: | ---: |
| 音乐状态/表达 | 1300/1350 | 1350/1350 |
| 历史完整前缀 | 160/160 | 160/160 |
| 全部匹配回答 | 1460/1510 | 1510/1510 |
| 图像五问法全部通过 | 292/302 | 302/302 |

固定checkpoint SHA256为737735c4d7d3483b38be2f88d8c48fbe3050b40c8f90c0d49b21e30336455616；本对照始终冻结UNet、VAE、条件编码器和Reader，不进行optimizer更新。使用原151组bank、两种固定noise、五问法和全部负对照，保留官方source分支、纯高斯起点、28步时间表、CFG1/image CFG1与原始严格答案+立即EOS评分。唯一改变是将每组训练缓存的单条raw条件embedding/mask复制至native三分支，替代native包装编码。

该结果补上了此前首步速度MSE诊断无法证明的完整生成/Reader结论：在当前固定开发任务上，条件不一致确实影响结果；保持旧权重也能通过一致条件恢复全部开发失败。但它不证明raw策略是官方Base默认，也不证明任意新输入都等价。模板、编码batch和padding属于这次共同改变的条件变量。

远端CPU实际重新打开新旧604份PT，核验实际checkpoint SHA、每份PT文件SHA、302个条件/noise的完整矩阵、真实CPU高斯起点、全部29个有限FP32状态、最终latent与Reader实际像素；两组6040原始回答逐条按固定gold token IDs+EOS重算，负对照逐对比较。随后完整下载归档，本地再次重算6040条原始记录和所有便携seal；PT/checkpoint仍在共享盘，本地归档未包含它们。

证据：[完整CPU摘要](1f86d56-raw-condition-verified-summary.json)、[本地重算](1f86d56-raw-condition-local-verification-summary.json)、[便携归档](1f86d56-raw-condition-verified-evidence.tgz)。归档603818字节，SHA256为1e5d85d7b80ed0ddaea03b4cf3a4b2d6092b15ad5c1cc84dc90bc9cb2433b9a0；摘要SHA256为fd6845f568d87ae895ee5aa665802ae501e1e4f896b5ad4cef3f93535c8f501f；probe complete SHA256为a2c1e60c27b360d5a9d4a046a4258818fd90abc57a6764fa378696af8a33bcd3。

另一个独立训练臂03f8467已用原生条件训练达到同样1510/1510，并保持官方Base原生推理，见[原生训练终点](native-condition-development-review.md)。两者不是同一模型，也不能共用功能验收结论：9e27050正在验证新03模型；35741bd将验证冻结bb权重的raw策略，逐次从真实当前RGB和事件重新编码条件，随后实际v2参数包/CLI重放。全部功能计划在raw开发结果产生前已固定，见[计划](../official-raw-condition-functional-validation-20260914.md)。当前两套方案均未完成全部连续写入和独立表达验收，goal保持进行中。
