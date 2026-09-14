# 03 原生 Writer 继续训练及清除保留试验

训练源码：`4fbc85725d78427235757ace2661d086b896a97f`。新实例 `dl-clear-retain-h200x4-20260914`，节点 `qb-prod-gpu911`，4×H200、80 CPU、900 GiB RAM、128 GiB shared memory。原实例对象保留；上一轮 `dl-official-exp-h200x4-20260914` 已自然到期 STOPPED。

2026-09-14 11:24 已启动，driver PID 476384，pilot 477729，四个训练 rank 为 478724–478727。11:28 实际观察四 rank 存活、GPU 已加载约44 GiB/rank。四 rank 初始参数逐位相同，SHA256 均为 `4a41876c30d6e8d8b5de5ac71af91fee97ae23f77a1299863dde1d32572fce90`，与03最终参数证据相符。梯度预检和完整基线尚在运行，尚无本轮训练结果。

固定训练输入和预算：

- 父模型为03已训练检查点 `d473825a403ad5c219681a09fc9e8a4270a63133393fc8ee30c0baac4c48d539`，参数包 manifest `ef4d4a91b99d0b05a1561e05875b2d7a7d772267e81c3bf9174dc861f52a8314`。
- fresh AdamW，lr=1e-5，4832 次额外更新、global batch 4、19328 draws、seed 20260915。保留原151条件、31逻辑类、历史9表达增强、全部训练时间范围和原生28步/CFG1推理。
- 训练前实际重算完整302图、全部轨迹和3020 raw，必须与03 trained阶段一致。检查点文件校验使用结果绑定的 `checkpoint-final.pt`。只允许 raw 中 baseline/trained 的阶段标签不同；失败必须传播给全部 rank，优化前退出。
- 这是同时改变初始化和学习率的修复试验，不做单因素因果归因。依据是已完整复核的320读数来源图/权重交叉诊断。

远端 run：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832`。独立只读训练源码目录：`repos/dreamlite-clear-retention-20260914`。训练 deadline 为14:15，之后的验证须在本实例租期内安排。

训练计划完整本地文件：`official-alignment-results-20260913/clear-retention-preregistered-training.json`，185617字节，SHA256 `70854e73d122a0976ff41bfb1397badf5fbe69034bfd9e4ef258b10caa4e1b7b`，已与实际 driver 登记匹配。

评估固定为完整1510开发格、1800原注册功能格、1800已观察表达回归格、实际CLI重放，以及3980 raw/3600 matched/796图的PNG复读。保持全部失败、无检查点筛选。已观察表达的完整登记：`clear-retention-observed-wording-regression.json`，193639字节，SHA256 `d64d1410f1c4026993f4523b9b0c869858da16416f001bd81bd020d9a881ca3f`。其中全部事件、噪声、语义和评分条件与原56套件一致，明确降为回归集，不再声称新holdout。

本地验证：初始化/逻辑采样/原生条件18项通过；最终checkpoint绑定修正后10项通过；完整功能、表达、raw及终点证据29项通过；PNG原有11项和新增谱系登记1项通过。CPU测试只验证实现和拒绝不符输入，不能代替实际GPU训练及功能评估。

部署操作：终止了一次确认存活但重复传输大量历史归档的旧Git fetch，确认进程退出后改为独立 shallow/promisor sparse checkout；实际HEAD和clean状态通过。未改动训练中的旧源码目录或删除任何实验结果。后续评估使用另一个固定源码目录，不能 checkout 正在训练的目录。

目标仍 active。只有完整评估有证据通过，才可讨论当前任务范围内的可用版本；本计划不是成功报告。

11:34更新：四卡真实梯度预检通过（relative L2 4.047132680232797e-08，relative max 9.02379502557885e-08，固定阈值2e-6），完整基线正在重算，已生成104/302个PT。实际训练identity确认lr1e-5及03-trained参考。验证源码e372f3cf7330ffbfbfe6570a75cd9b80c6011dbc在独立目录部署成功；原注册、表达回归、PNG驱动640212/640213/640214均存活并按依赖等待，统一deadline17:20。尚无本轮训练后的效果结果。
