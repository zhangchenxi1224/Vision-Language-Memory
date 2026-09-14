# 历史表达增强：完整训练及开发集证据

b9f90e956eea7bda15f638c8877919941ce4fec5 已完成固定4832次更新及完整终点评估。开发集1510/1510，302张图的五种问法全部通过；相较本轮初始化，保留1273条正确回答并修复237条，没有开发集退步。此前03原生条件训练也达到1510/1510，因此这个总分本身不能证明历史表达增强有效；原回归、新表达及PNG部署验收仍是必要条件。

| 开发分区 | 初始化 | 本轮终点 |
|---|---:|---:|
| 音乐状态转换 | 1230/1350 | 1350/1350 |
| 16个历史前缀条件 | 43/160 | 160/160 |
| 图像五问全部通过 | 250/302 | 302/302 |

训练保持原151条件bank、31逻辑条件权重、原d153参数初始化、fresh AdamW、全U-Net、官方target/noise FM与原生28步FP32/CFG1推理。只增加历史训练事件表述：16个条件各9表达，每个表达实际抽到69–70次，历史部分共9975次draw。全部19328次source/teacher/noise/sigma与实际表达/embedding/mask绑定已重放检查；原编码seal为`bfdcfacc5925a42f80aec0203b1b4a2bd197391525e5f7ff25162288051b0467`。CPU审计检查绑定，没有重新运行条件编码器。

实际训练sigma最小0.00006788969039916992，最大0.9999235272407532，9583次位于0.5以上。最终4rank参数逐位一致，参数SHA为`29c9f6e07c4eeec6c6d32fd7f236de79efe457b8f18d4a7543d7e4f544eb02b8`。参数更新阶段3896.7965808808804秒；根实验terminal于1789343971.9516258记录completed。

独立7b82309 collector在远端实际检查完整终点文件、训练抽样和原初始化基线对应，再独立打开全部302个最终PT，检查实际Gaussian初态、每个29个有限FP32状态、末态latent和Reader像素。最终checkpoint文件SHA为`34f33ea7cd9b209d80c460c91bc102c06b48ad40a54b6fbb224f5f1637529348`；result SHA为`f939f80c79fcd625eceab04551e57fe5c2a571957e521fad74be108929b7047f`。checkpoint与PT留在远端，未声称在本地读取这些大张量。

本地已完整下载以下两套归档，按独立远端观察的SHA校验：

- [完整终点归档](7b82309-logical-endpoint-evidence.tgz)，5779807字节，SHA`5d6e92f5154241b5a03b42e3a267540c4386ceacf84b196f4e64fb7895eb228d`。本地重算前后6040条raw及全部19328次抽样，见[完整复核记录](7b82309-logical-endpoint-local-verification.json)。[远端summary](7b82309-logical-endpoint-summary.json)从归档提取原始字节，SHA`8e58122436ecac69d5784f2c941e099166d1f47c2163583a34cdcac343fe9733`。
- [独立最终PT审计归档](7b82309-logical-final-tensors-evidence.tgz)，386911字节，SHA`e2ac6ac74e6f8366e63b1449aa73eee363679e1bfb57170cdf392102552f5d38`。本地重算3020条raw及302个审计绑定，见[本地复核](7b82309-logical-final-tensors-local-verification-summary.json)。[远端summary](7b82309-logical-final-tensors-summary.json)的SHA为`7a0da3611bc1fb8118185bcc63d2a9c01fcef59a27ae9e409f56a365d975f5de`。

这证明已见17道语义问题的开发集保留情况与训练协议执行。7b完整旧案例回归正在运行，56新表达和385全量PNG读取随后执行。尚不判定可用，不能用开发集通过替代这些实际功能证据。
