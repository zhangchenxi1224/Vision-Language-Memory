# Direct EOS 首批真实结果快照

更新：原实验已完成96/96条，完整训练记录、原始回答、初值与终点数据及图表见[完整结果报告](../direct-latent-geometry-complete-results-20260909/README.md)。下文保留当时18条快照。

抓取时间：2026-09-08T11:11:16.970767+00:00（UTC）。这是一份运行中快照，不是全量结论。

完成 18/96，原问句正确 18/18，五问法全正确 13/18。所有结果来自本轮重新初始化与256步训练；每条endpoint_raw.pt已复算文件SHA。

| 分布 | 当前完成 | 原问句正确 | 五问法全正确 |
|---|---:|---:|---:|
| gaussian | 4 | 4/4 | 2/4 |
| uniform | 4 | 4/4 | 3/4 |
| sphere | 4 | 4/4 | 4/4 |
| rademacher | 3 | 3/3 | 3/3 |
| heavy_tail | 3 | 3/3 | 1/3 |

尚未覆盖全部seed和scale，不能按当前不同样本量判定分布优劣。训练原问句与评测改写共享完全一致的两行短答案指令。出现原问句正确但改写续写说明问句稳定性仍有差异。

部署：`dl-base-h200x4-20260907`，commit `0f704178137ee4beac952224f3175a3923d5e438`。远端输出：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/direct-latent-geometry/0f70417-20260908-r01`。

同目录 snapshot.json 保存每条已完成轨迹的11个checkpoint原始回答、5问法终点回答、blank/donor结果和训练用时。源码已有13项CPU检查通过，H200实际EOS结束符为151645，VAE/Reader冻结且重复损失和z梯度bitwise一致。此阶段不是U-Net训练结果。
