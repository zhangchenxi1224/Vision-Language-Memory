# 新四卡开发实例并行运行记录

更新时间：2026-09-08 16:38（Asia/Shanghai）。

用户指定的开发实例 `dl-base-h200x4-20260907` 已于16:36:50启动多题Open/EOS实验。
这项实验推进此前方案P4，回答“停止监督的收益能否在其他真实问题、不同答案长度上复现”。
它与Frozen DreamLite几何主线并行，不重复原ambient单题实验，不训练共享Writer。

## 实际资源与代码

- Workspace：分布式训练空间；Project：前沿课题探索。
- 开发实例：`dl-base-h200x4-20260907`；节点`qb-prod-gpu2309`。
- 四张NVIDIA H200，每卡143771MiB，80CPU，900GiB RAM，128GiB共享内存。
- 镜像：`ngc-pytorch:25.02-cuda12.8.0-py3`。
- 分支：`codex/r11-open-eos-multiquestion-20260908`。
- 固定训练提交：`2c0e41c899910bf0641f16ee724f85bbe3491a7e`；远端源码不随报告更新。
- target配置SHA256：`d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672`。
- 启动器PID246163；两个训练进程PID246188、246189。

## 工作划分和实际进展

| 资源 | 内容 | 16:38实际状态 |
|---|---|---|
| GPU0/1 | 每个固定层的第一题，共8题×4seed×A/B=64次训练 | target000-seed00-A已保存step32检查点 |
| GPU2/3 | 每个固定层的第二题，共8题×4seed×A/B=64次训练 | target001-seed00-A已保存step32检查点 |

四张卡实际显存占用为2491/10317/2491/10317MiB，均观察到非零利用率。
模型加载、真实source数据溯源、donor图像的成功证据及哈希核验已通过；两条训练链路确已更新并保存检查点。
完整科学结论仍需等待128次训练和固定raw生成评测结束。

原四卡主线是训练任务（Job）`vlm-oracle-geometry-h200x4-20260908-r02`，
位于分布式训练空间／前沿课题探索项目的训练任务列表，不属于开发实例列表。
其节点为`qb-prod-gpu738`，16:38仍正常运行；A1已完成5条，最后一条到40/256步。
本次未取消、重建或修改该运行作业。

## 产物位置

```text
代码：
/inspire/ssd/project/exploration-topic/czxs26210936/repos/r11-open-eos-multiquestion-20260908

输出根：
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos-multiquestion/paired-2c0e41c-20260908-r01

启动记录：
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos-multiquestion/notebook-launch-2c0e41c-20260908-r01/launch.json
```

输出根中：`status.json`记录总进度，`lane-0.log`和`lane-1.log`记录两路训练；
`lane-0/runs`及`lane-1/runs`保存每条轨迹的7个检查点、256步指标、27条原始生成和SHA目录。
两路完成后写合并`summary.json`与`terminal.json`，预期128条训练、32768次更新、3456条raw生成。
汇总包含按问题、答案实际token长度、问法和换图对照的指标，长度分层仅为描述性比较。

## 验证与运行边界

- 本轮19项runner/选择/调度测试通过，包括真实train_one的256步CPU演练、完整保存与校验复用。
- 原EOS回归已由子智能体复验，Linux advisory lock也已实际验证。
- 原始16题来自固定train文件中的16个不同semantic groups，覆盖5类任务，排除旧8题。
- GPU0/1与2/3分配没有题目交集；既有完整结果校验后可复用，partial不覆盖。
- 原实例设置的自动停止仍有效；本启动器另设2.5小时上限，不以此上限当完成时间承诺。
- 预计耗时需要以首批完整A/B的实际耗时校准；不能把CPU测试或平台RUNNING当作GPU实验完成。
