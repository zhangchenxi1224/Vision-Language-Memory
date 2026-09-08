# 17:25 中途结果：仅统计完整配对题

时间：2026-09-08 17:25（Asia/Shanghai）。这不是最终结果，两个lane均仍在训练。
17:24:39核查时已完成71/128条训练；lane0完成36条，lane1完成35条。
仅前8题（target0–7）已完整覆盖4seed×A/B，因此以下比较统一使用8题×4seed=32组配对，
不把较早完成的单个A或B混入分母；不依据结果挑题。

| Raw完整输出评测 | A：仅答案CE | B：答案CE+EOS |
|---|---:|---:|
| 原题EM | 28/32 | 32/32 |
| 固定改写EM，未训练 | 30/32 | 32/32 |
| 另一全新改写EM，未训练 | 28/32 | 31/32 |
| 原题正确答案前缀 | 32/32 | 32/32 |
| 原题多余续写 | 4/32 | 0/32 |

这些数字来自每个完整run的step256、matched图、原始greedy32token生成。
换成blank图，两种方法三种问法均0/32；固定ambient donor图对应三种问法为4/32、8/32、0/32，
两种方法一致。donor正确记录是重复使用同一图的对照结果，不是独立donor样本。

## 逐题原题EM

| target | 字段/答案 | 实际答案token数 | A | B |
|---|---|---:|---:|---:|
| 0 | color / green | 1 | 4/4 | 4/4 |
| 1 | color / green | 1 | 4/4 | 4/4 |
| 2 | drink / juice | 2 | 4/4 | 4/4 |
| 3 | drink / juice | 2 | 4/4 | 4/4 |
| 4 | music / jazz | 2 | 4/4 | 4/4 |
| 5 | music / jazz | 2 | 4/4 | 4/4 |
| 6 | material / linen | 2 | 3/4 | 4/4 |
| 7 | material / linen | 2 | 1/4 | 4/4 |

当前原题改善主要来自linen两题，不能说EOS对每道题都有增益。
green/juice/jazz的原题A本来已满分。B在一个jazz全新改写下仍有答案本身错误，
因此停止监督不等于解决所有问法鲁棒性问题。
`no active preference`的三token问题尚未完成，不能提前报告三token总成绩。
全部问题都是独立优化latent；8个问题是独立任务数，32组配对不是32个独立问题，
也不代表共享Writer或未见问题泛化成功。

## 同时运行的几何主线

`vlm-oracle-geometry-h200x4-20260908-r02`仍为job_running。
A1的6次重复运行完成，完整bitwise tensor/trajectory gate通过。
主体多起点A2已完成seed0–5；同一道target1难题严格成功仅seed4，因此当前为1/6。
seed6/7正在运行。不同起点已经出现成功和失败，不能宣称当前配置稳定成功。
整个主体清单当前完成12/158（含6条A1），分布/尺度/局部几何全量结论仍待后续实验。

## 原始文件位置

多题输出：
`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos-multiquestion/paired-2c0e41c-20260908-r01`

上述多题指标由`lane-0/runs`和`lane-1/runs`内完成run的`raw_generations.jsonl`重算。
多题固定训练commit：`2c0e41c899910bf0641f16ee724f85bbe3491a7e`。
几何输出：
`/inspire/ssd/project/exploration-topic/czxs26210936/runs/frozen-oracle-geometry/3c8cf7d-20260908-r02`
几何1/6来自6个A2完成run的`summary.json`之`qa_pass`，不是训练loss阈值。
